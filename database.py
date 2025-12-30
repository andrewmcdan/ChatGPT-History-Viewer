import json
import os
import sqlite3
import threading
import sys
from array import array
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


MessageTuple = Tuple[str, Optional[str], Dict[str, Any]]

NON_SEARCHABLE_CONTENT_TYPES = {
    "user_editable_context",
}

EMBEDDING_METADATA_TABLE = "embedding_metadata"
EMBEDDING_MAP_TABLE = "message_embedding_map"
EMBEDDING_VECTOR_TABLE = "message_embeddings"
VECTOR_BACKEND_ENV = "CHAT_HISTORY_VECTOR_BACKEND"
VECTOR_EXTENSION_ENV = "CHAT_HISTORY_VECTOR_EXTENSION"
DEFAULT_VECTOR_BACKEND = "vec"
EMBEDDING_MAX_CHARS_ENV = "CHAT_HISTORY_EMBEDDING_MAX_CHARS"
DEFAULT_EMBEDDING_MAX_CHARS = 8000


class ChatHistoryDatabase:
    """
    SQLite-backed store for ChatGPT conversations and messages.

    The database keeps a flattened representation of each conversation to make
    UI rendering and text search straightforward.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        # check_same_thread=False lets us import JSON on a worker thread.
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._vector_extension_loaded = False
        self._ensure_schema()
        self._purge_non_searchable_text()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _ensure_schema(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    create_time REAL,
                    update_time REAL,
                    default_model_slug TEXT,
                    raw_metadata TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    project_id TEXT,
                    title TEXT,
                    create_time REAL,
                    update_time REAL,
                    default_model_slug TEXT,
                    is_archived INTEGER,
                    is_starred INTEGER,
                    raw_metadata TEXT,
                    FOREIGN KEY (project_id)
                        REFERENCES projects(project_id)
                        ON DELETE SET NULL
                )
                """
            )
            self._ensure_column("conversations", "project_id", "TEXT")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    parent_id TEXT,
                    position INTEGER NOT NULL,
                    author_role TEXT,
                    content_type TEXT,
                    text_content TEXT,
                    create_time REAL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, position)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_project
                ON conversations(project_id)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_text
                ON messages(text_content)
                """
            )
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {EMBEDDING_METADATA_TABLE} (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {EMBEDDING_MAP_TABLE} (
                    embedding_id INTEGER PRIMARY KEY,
                    message_id TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL,
                    create_time REAL,
                    FOREIGN KEY (message_id)
                        REFERENCES messages(message_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                )
                """
            )
            self.conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_embedding_map_conversation
                ON {EMBEDDING_MAP_TABLE}(conversation_id)
                """
            )
            self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        existing = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        if not any(row["name"] == column for row in existing):
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _purge_non_searchable_text(self) -> None:
        if not NON_SEARCHABLE_CONTENT_TYPES:
            return
        placeholders = ",".join("?" for _ in NON_SEARCHABLE_CONTENT_TYPES)
        with self._lock, self.conn:
            self.conn.execute(
                f"""
                UPDATE messages
                SET text_content = ''
                WHERE content_type IN ({placeholders})
                  AND text_content IS NOT NULL
                  AND text_content <> ''
                """,
                tuple(NON_SEARCHABLE_CONTENT_TYPES),
            )

    def _vector_backend(self) -> str:
        backend = os.environ.get(VECTOR_BACKEND_ENV, DEFAULT_VECTOR_BACKEND).strip().lower()
        if backend not in {"vec", "vss"}:
            backend = DEFAULT_VECTOR_BACKEND
        return backend

    def _load_vector_extension(self) -> None:
        if self._vector_extension_loaded:
            return
        extension_path = os.environ.get(VECTOR_EXTENSION_ENV)
        if not extension_path:
            raise RuntimeError(
                f"Vector extension not configured. Set {VECTOR_EXTENSION_ENV} to the extension path."
            )
        paths = [item.strip() for item in extension_path.split(";") if item.strip()]
        if not paths:
            raise RuntimeError(
                f"Vector extension not configured. Set {VECTOR_EXTENSION_ENV} to the extension path."
            )
        self.conn.enable_load_extension(True)
        try:
            for path in paths:
                self.conn.load_extension(path)
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"Failed to load vector extension: {exc}") from exc
        self._vector_extension_loaded = True

    def _vector_table_exists(self) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (EMBEDDING_VECTOR_TABLE,),
        ).fetchone()
        return row is not None

    def _get_embedding_metadata(self, key: str) -> Optional[str]:
        row = self.conn.execute(
            f"SELECT value FROM {EMBEDDING_METADATA_TABLE} WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return row["value"]

    def _set_embedding_metadata(self, key: str, value: str) -> None:
        self.conn.execute(
            f"""
            INSERT INTO {EMBEDDING_METADATA_TABLE} (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _ensure_vector_table(self, embedding_dim: int, embedding_model: Optional[str]) -> None:
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive.")

        backend = self._vector_backend()
        self._load_vector_extension()

        stored_backend = self._get_embedding_metadata("embedding_backend")
        stored_dim = self._get_embedding_metadata("embedding_dim")
        stored_model = self._get_embedding_metadata("embedding_model")

        if stored_backend and stored_backend != backend:
            raise RuntimeError(
                f"Embedding backend mismatch (db={stored_backend}, env={backend})."
            )
        if stored_dim and int(stored_dim) != embedding_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch (db={stored_dim}, env={embedding_dim})."
            )
        if stored_model and embedding_model and stored_model != embedding_model:
            raise RuntimeError(
                f"Embedding model mismatch (db={stored_model}, env={embedding_model})."
            )

        if not self._vector_table_exists():
            if backend == "vss":
                create_sql = (
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {EMBEDDING_VECTOR_TABLE} "
                    f"USING vss0(embedding({embedding_dim}))"
                )
            else:
                create_sql = (
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {EMBEDDING_VECTOR_TABLE} "
                    f"USING vec0(embedding float[{embedding_dim}])"
                )
            with self.conn:
                self.conn.execute(create_sql)
                self._set_embedding_metadata("embedding_backend", backend)
                self._set_embedding_metadata("embedding_dim", str(embedding_dim))
                if embedding_model:
                    self._set_embedding_metadata("embedding_model", embedding_model)
        else:
            with self.conn:
                self._set_embedding_metadata("embedding_backend", backend)
                self._set_embedding_metadata("embedding_dim", str(embedding_dim))
                if embedding_model:
                    self._set_embedding_metadata("embedding_model", embedding_model)

    def _delete_embeddings_for_conversation(self, conversation_id: str) -> None:
        if self._vector_extension_loaded and self._vector_table_exists():
            self.conn.execute(
                f"""
                DELETE FROM {EMBEDDING_VECTOR_TABLE}
                WHERE rowid IN (
                    SELECT embedding_id
                    FROM {EMBEDDING_MAP_TABLE}
                    WHERE conversation_id = ?
                )
                """,
                (conversation_id,),
            )
        self.conn.execute(
            f"DELETE FROM {EMBEDDING_MAP_TABLE} WHERE conversation_id = ?",
            (conversation_id,),
        )

    def _next_embedding_id(self) -> int:
        row = self.conn.execute(
            f"SELECT COALESCE(MAX(embedding_id), 0) + 1 AS next_id FROM {EMBEDDING_MAP_TABLE}"
        ).fetchone()
        next_id = int(row["next_id"]) if row is not None else 1
        if self._vector_extension_loaded and self._vector_table_exists():
            row = self.conn.execute(
                f"SELECT COALESCE(MAX(rowid), 0) + 1 AS next_id FROM {EMBEDDING_VECTOR_TABLE}"
            ).fetchone()
            if row is not None:
                next_id = max(next_id, int(row["next_id"]))
        return next_id

    # ------------------------------------------------------------------ import

    def import_conversations_json(
        self,
        json_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, int]:
        """
        Import ChatGPT conversations from an exported JSON file.

        Deduplicates on conversation_id; if a newer version of a conversation is
        imported, the previous messages are replaced.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as fp:
            conversations = json.load(fp)

        total = len(conversations)
        inserted = updated = skipped = 0

        for index, conversation in enumerate(conversations, start=1):
            action = self._upsert_conversation(conversation)
            if action == "inserted":
                inserted += 1
            elif action == "updated":
                updated += 1
            else:
                skipped += 1

            if progress_callback:
                progress_callback(index, total)

        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def import_projects_json(
        self,
        json_path: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, int]:
        """
        Import ChatGPT project metadata from projects.json exports.
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)

        if isinstance(payload, dict) and "projects" in payload:
            projects = payload["projects"]
        else:
            projects = payload

        if not isinstance(projects, list):
            raise ValueError("projects.json is expected to contain a list of project objects.")

        total = len(projects)
        inserted = updated = skipped = 0

        for index, project in enumerate(projects, start=1):
            action = self._upsert_project(project)
            if action == "inserted":
                inserted += 1
            elif action == "updated":
                updated += 1
            else:
                skipped += 1

            if progress_callback:
                progress_callback(index, total)

        return {"inserted": inserted, "updated": updated, "skipped": skipped}

    def _upsert_conversation(self, conversation: Dict[str, Any]) -> str:
        conversation_id = (
            conversation.get("conversation_id") or conversation.get("id")
        )
        if not conversation_id:
            return "skipped"

        update_time = _as_float(conversation.get("update_time"))
        project_id = _extract_project_id(conversation)

        with self._lock:
            row = self.conn.execute(
                "SELECT update_time FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()

            if row is not None:
                stored_update = _as_float(row["update_time"])
                # Skip older or identical snapshots.
                if stored_update is not None and update_time is not None:
                    if stored_update >= update_time:
                        return "skipped"

            metadata = {
                "conversation_origin": conversation.get("conversation_origin"),
                "voice": conversation.get("voice"),
                "owner": conversation.get("owner"),
                "is_read_only": conversation.get("is_read_only"),
                "is_do_not_remember": conversation.get("is_do_not_remember"),
                "memory_scope": conversation.get("memory_scope"),
                "async_status": conversation.get("async_status"),
                "gizmo_type": conversation.get("gizmo_type"),
                "plugin_ids": conversation.get("plugin_ids"),
                "disabled_tool_ids": conversation.get("disabled_tool_ids"),
                "project_snapshot": conversation.get("project"),
            }

            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO conversations (
                        conversation_id,
                        project_id,
                        title,
                        create_time,
                        update_time,
                        default_model_slug,
                        is_archived,
                        is_starred,
                        raw_metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        project_id = excluded.project_id,
                        title = excluded.title,
                        create_time = excluded.create_time,
                        update_time = excluded.update_time,
                        default_model_slug = excluded.default_model_slug,
                        is_archived = excluded.is_archived,
                        is_starred = excluded.is_starred,
                        raw_metadata = excluded.raw_metadata
                    """,
                    (
                        conversation_id,
                        project_id,
                        conversation.get("title"),
                        _as_float(conversation.get("create_time")),
                        update_time,
                        conversation.get("default_model_slug"),
                        _as_int_bool(conversation.get("is_archived")),
                        _as_int_bool(conversation.get("is_starred")),
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )

                # Replace messages for this conversation.
                self._delete_embeddings_for_conversation(conversation_id)
                self.conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                )

                message_rows = []
                for position, (message_id, parent_id, message) in enumerate(
                    _iterate_messages(conversation), start=1
                ):
                    extracted_text = _extract_message_text(message)
                    message_rows.append(
                        (
                            message_id,
                            conversation_id,
                            parent_id,
                            position,
                            message.get("author", {}).get("role"),
                            (message.get("content") or {}).get("content_type"),
                            extracted_text,
                            _as_float(message.get("create_time")),
                        )
                    )

                if message_rows:
                    self.conn.executemany(
                        """
                        INSERT OR REPLACE INTO messages (
                            message_id,
                            conversation_id,
                            parent_id,
                            position,
                            author_role,
                            content_type,
                            text_content,
                            create_time
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        message_rows,
                    )

        return "updated" if row is not None else "inserted"

    def _upsert_project(self, project: Dict[str, Any]) -> str:
        project_id = project.get("project_id") or project.get("id")
        if not project_id:
            return "skipped"

        update_time = (
            _as_float(project.get("update_time"))
            or _as_float(project.get("updated_at"))
            or _as_float(project.get("modified_time"))
        )

        with self._lock:
            row = self.conn.execute(
                "SELECT update_time FROM projects WHERE project_id = ?",
                (project_id,),
            ).fetchone()

            if row is not None:
                stored_update = _as_float(row["update_time"])
                if stored_update is not None and update_time is not None and stored_update >= update_time:
                    return "skipped"

            name = project.get("title") or project.get("name")
            description = project.get("description") or project.get("summary")
            create_time = (
                _as_float(project.get("create_time"))
                or _as_float(project.get("created_at"))
                or _as_float(project.get("creation_time"))
            )
            default_model = (
                project.get("default_model_slug")
                or project.get("default_model")
                or project.get("model")
            )

            metadata = {
                "team_id": project.get("team_id"),
                "visibility": project.get("visibility"),
                "archived": project.get("archived"),
                "owner_id": project.get("owner_id"),
                "raw": project,
            }

            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO projects (
                        project_id,
                        name,
                        description,
                        create_time,
                        update_time,
                        default_model_slug,
                        raw_metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id) DO UPDATE SET
                        name = excluded.name,
                        description = excluded.description,
                        create_time = COALESCE(excluded.create_time, projects.create_time),
                        update_time = excluded.update_time,
                        default_model_slug = excluded.default_model_slug,
                        raw_metadata = excluded.raw_metadata
                    """,
                    (
                        project_id,
                        name,
                        description,
                        create_time,
                        update_time,
                        default_model,
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )

        return "updated" if row is not None else "inserted"

    # --------------------------------------------------------------- embeddings

    def build_message_embeddings(
        self,
        embedder: Any,
        batch_size: int = 64,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT m.message_id, m.conversation_id, m.create_time, m.text_content
                FROM messages m
                LEFT JOIN {EMBEDDING_MAP_TABLE} map
                    ON map.message_id = m.message_id
                WHERE map.message_id IS NULL
                  AND m.text_content IS NOT NULL
                  AND m.text_content <> ''
                ORDER BY m.conversation_id, m.position
                """
            ).fetchall()

        total = len(rows)
        if total == 0:
            return {"embedded": 0, "total": 0, "skipped": 0}

        with self._lock:
            self._load_vector_extension()

        max_chars = _embedding_max_chars()
        embedded = 0
        embedding_dim: Optional[int] = None
        embedding_model = getattr(embedder, "model", None)
        next_id: Optional[int] = None

        for batch in _chunked(rows, batch_size):
            embeddings_by_message: Dict[str, List[float]] = {}
            short_rows: List[Any] = []
            short_texts: List[str] = []

            for row in batch:
                text = row["text_content"] or ""
                if not text.strip():
                    continue
                if max_chars and len(text) > max_chars:
                    embedding = _embed_text_with_chunking(embedder, text, max_chars, batch_size)
                    embeddings_by_message[row["message_id"]] = embedding
                else:
                    short_rows.append(row)
                    short_texts.append(text)

            if short_texts:
                embeddings = embedder.embed_texts(short_texts)
                if len(embeddings) != len(short_rows):
                    raise RuntimeError("Embedding response count mismatch.")
                for row, embedding in zip(short_rows, embeddings):
                    embeddings_by_message[row["message_id"]] = embedding

            if not embeddings_by_message:
                continue

            if embedding_dim is None:
                embedding_dim = len(next(iter(embeddings_by_message.values())))
                with self._lock:
                    self._ensure_vector_table(embedding_dim, embedding_model)
                    next_id = self._next_embedding_id()

            if embedding_dim is None or next_id is None:
                raise RuntimeError("Embedding index failed to initialize.")

            for embedding in embeddings_by_message.values():
                if len(embedding) != embedding_dim:
                    raise RuntimeError("Embedding dimension mismatch in batch.")

            with self._lock, self.conn:
                for row in batch:
                    embedding = embeddings_by_message.get(row["message_id"])
                    if embedding is None:
                        continue
                    vector_blob = _serialize_embedding(embedding)
                    self.conn.execute(
                        f"""
                        INSERT INTO {EMBEDDING_VECTOR_TABLE} (rowid, embedding)
                        VALUES (?, ?)
                        """,
                        (next_id, vector_blob),
                    )
                    self.conn.execute(
                        f"""
                        INSERT INTO {EMBEDDING_MAP_TABLE} (
                            embedding_id, message_id, conversation_id, create_time
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            next_id,
                            row["message_id"],
                            row["conversation_id"],
                            row["create_time"],
                        ),
                    )
                    next_id += 1

            embedded += len(embeddings_by_message)
            if progress_callback:
                progress_callback(embedded, total)

        return {"embedded": embedded, "total": total, "skipped": total - embedded}

    def semantic_search_conversations(
        self,
        query: str,
        embedder: Any,
        project_id: Optional[str] = None,
        limit: int = 200,
        message_limit: int = 200,
    ) -> List[sqlite3.Row]:
        if not query:
            return self.list_conversations(project_id=project_id)

        if message_limit <= 0:
            raise ValueError("message_limit must be positive.")
        if limit <= 0:
            raise ValueError("limit must be positive.")

        with self._lock:
            self._load_vector_extension()

        max_chars = _embedding_max_chars()
        if max_chars and len(query) > max_chars:
            embedding = _embed_text_with_chunking(embedder, query, max_chars, 32)
        else:
            embeddings = embedder.embed_texts([query])
            if not embeddings:
                return []
            embedding = embeddings[0]
        embedding_dim = len(embedding)
        embedding_model = getattr(embedder, "model", None)
        vector_blob = _serialize_embedding(embedding)
        backend = self._vector_backend()

        with self._lock:
            self._ensure_vector_table(embedding_dim, embedding_model)
            if not self._vector_table_exists():
                raise RuntimeError("Embedding index not initialized. Build embeddings first.")

            total_embeddings = self.conn.execute(
                f"SELECT COUNT(*) AS total FROM {EMBEDDING_MAP_TABLE}"
            ).fetchone()["total"]
            if total_embeddings == 0:
                raise RuntimeError("No embeddings found. Build embeddings first.")

            match_clause = "vss_search(v.embedding, ?)" if backend == "vss" else "v.embedding MATCH ?"
            sql = f"""
                WITH matches AS (
                    SELECT map.conversation_id, v.distance
                    FROM {EMBEDDING_VECTOR_TABLE} v
                    JOIN {EMBEDDING_MAP_TABLE} map
                        ON map.embedding_id = v.rowid
                    WHERE {match_clause}
                    ORDER BY v.distance
                    LIMIT ?
                )
                SELECT c.conversation_id,
                       c.project_id,
                       c.title,
                       c.create_time,
                       c.update_time,
                       MIN(matches.distance) AS best_distance
                FROM matches
                JOIN conversations c
                    ON c.conversation_id = matches.conversation_id
            """
            params: List[Any] = [vector_blob, message_limit]
            if project_id:
                sql += " WHERE c.project_id = ?"
                params.append(project_id)
            sql += """
                GROUP BY c.conversation_id
                ORDER BY best_distance ASC, COALESCE(c.update_time, c.create_time, 0) DESC
                LIMIT ?
            """
            params.append(limit)

            cursor = self.conn.execute(sql, tuple(params))
            return list(cursor.fetchall())

    # ------------------------------------------------------------------ queries

    def list_conversations(self, project_id: Optional[str] = None) -> List[sqlite3.Row]:
        with self._lock:
            sql = """
                SELECT conversation_id, project_id, title, create_time, update_time
                FROM conversations
            """
            params: Tuple[Any, ...] = ()
            if project_id:
                sql += " WHERE project_id = ?"
                params = (project_id,)
            sql += " ORDER BY COALESCE(update_time, create_time, 0) DESC"
            cursor = self.conn.execute(sql, params)
            return list(cursor.fetchall())

    def search_conversations(self, query: str, project_id: Optional[str] = None) -> List[sqlite3.Row]:
        pattern = f"%{query.lower()}%"
        with self._lock:
            sql = """
                SELECT conversation_id, project_id, title, create_time, update_time
                FROM conversations
                WHERE (
                    LOWER(COALESCE(title, '')) LIKE ?
                    OR EXISTS (
                        SELECT 1 FROM messages
                        WHERE messages.conversation_id = conversations.conversation_id
                          AND LOWER(COALESCE(messages.text_content, '')) LIKE ?
                    )
                )
            """
            params: List[Any] = [pattern, pattern]
            if project_id:
                sql += " AND project_id = ?"
                params.append(project_id)
            sql += " ORDER BY COALESCE(update_time, create_time, 0) DESC"
            cursor = self.conn.execute(sql, tuple(params))
            return list(cursor.fetchall())

    def list_projects(self) -> List[sqlite3.Row]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT project_id, name, description, create_time, update_time
                FROM projects
                ORDER BY COALESCE(update_time, create_time, 0) DESC
                """
            )
            return list(cursor.fetchall())

    def get_conversation_messages(self, conversation_id: str) -> List[sqlite3.Row]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT message_id,
                       parent_id,
                       position,
                       author_role,
                       content_type,
                       text_content,
                       create_time
                FROM messages
                WHERE conversation_id = ?
                ORDER BY position ASC
                """,
                (conversation_id,),
            )
            return list(cursor.fetchall())


# --------------------------------------------------------------------------- #
# helpers


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int_bool(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    return None


def _iterate_messages(conversation: Dict[str, Any]) -> Iterable[MessageTuple]:
    """
    Yield (message_id, parent_id, message_payload) in a stable pre-order walk.
    """
    mapping = conversation.get("mapping") or {}
    if not isinstance(mapping, dict):
        return

    roots = [
        node_id
        for node_id, node in mapping.items()
        if isinstance(node, dict) and node.get("parent") is None
    ]
    if not roots:
        roots = list(mapping.keys())

    visited: set[str] = set()

    def traverse(node_id: str) -> Iterable[MessageTuple]:
        if node_id in visited:
            return
        visited.add(node_id)
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            return
        message = node.get("message")
        if isinstance(message, dict) and message.get("id"):
            yield message["id"], node.get("parent"), message
        for child_id in node.get("children") or []:
            yield from traverse(child_id)

    for root in roots:
        yield from traverse(root)


def _extract_message_text(message: Dict[str, Any]) -> str:
    """
    Pull human-readable text from the message payload, handling a variety of
    content payloads (text, code, execution output, attachments, etc.).
    """
    content = message.get("content") or {}
    content_type = content.get("content_type")
    if isinstance(content_type, str) and content_type.lower() in NON_SEARCHABLE_CONTENT_TYPES:
        return ""
    segments: List[str] = []

    parts = content.get("parts")
    if isinstance(parts, list):
        for part in parts:
            rendered = _render_part(part)
            if rendered:
                segments.append(rendered)

    if isinstance(content.get("text"), str):
        segments.append(content["text"])

    if isinstance(content.get("title"), str):
        segments.append(content["title"])

    if isinstance(content.get("body"), str):
        segments.append(content["body"])

    if not segments and content:
        try:
            segments.append(json.dumps(content, ensure_ascii=False))
        except TypeError:
            segments.append(str(content))

    return "\n\n".join(segment.strip() for segment in segments if segment).strip()


def _render_part(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        if isinstance(part.get("text"), str):
            return part["text"]
        if "asset_pointer" in part:
            pointer = part.get("asset_pointer")
            return f"[Attachment: {pointer}]"
        if "code" in part and isinstance(part["code"], str):
            return part["code"]
        try:
            return json.dumps(part, ensure_ascii=False)
        except TypeError:
            return str(part)
    return str(part)


def _extract_project_id(conversation: Dict[str, Any]) -> Optional[str]:
    project_id = conversation.get("project_id") or conversation.get("projectId")
    if project_id:
        return str(project_id)

    project_data = conversation.get("project")
    if isinstance(project_data, dict):
        candidate = project_data.get("id") or project_data.get("project_id")
        if candidate:
            return str(candidate)

    metadata = conversation.get("metadata")
    if isinstance(metadata, dict):
        candidate = metadata.get("project_id") or metadata.get("projectId")
        if candidate:
            return str(candidate)

    return None


def _serialize_embedding(vector: List[float]) -> bytes:
    blob = array("f", vector)
    if sys.byteorder != "little":
        blob.byteswap()
    return blob.tobytes()


def _chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _embedding_max_chars() -> int:
    raw_value = os.environ.get(EMBEDDING_MAX_CHARS_ENV)
    if not raw_value:
        return DEFAULT_EMBEDDING_MAX_CHARS
    try:
        parsed = int(raw_value)
    except ValueError:
        return DEFAULT_EMBEDDING_MAX_CHARS
    return parsed if parsed > 0 else 0


def _chunk_text(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    paragraphs = [part for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        return [text[:max_chars]]

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            candidate = paragraph
        else:
            candidate = f"{current}\n\n{paragraph}"

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
            continue

        for start in range(0, len(paragraph), max_chars):
            chunk = paragraph[start : start + max_chars]
            if chunk.strip():
                chunks.append(chunk)

    if current:
        chunks.append(current)

    return chunks


def _embed_text_with_chunking(
    embedder: Any,
    text: str,
    max_chars: int,
    batch_size: int,
) -> List[float]:
    chunks = _chunk_text(text, max_chars)
    if not chunks:
        raise RuntimeError("No text chunks to embed.")

    if len(chunks) == 1:
        embeddings = embedder.embed_texts(chunks)
        if not embeddings:
            raise RuntimeError("Embedding response was empty.")
        return embeddings[0]

    embeddings: List[List[float]] = []
    for chunk_batch in _chunked(chunks, batch_size):
        embeddings.extend(embedder.embed_texts(chunk_batch))

    if not embeddings:
        raise RuntimeError("Embedding response was empty.")

    return _average_embeddings(embeddings)


def _average_embeddings(embeddings: List[List[float]]) -> List[float]:
    if not embeddings:
        raise RuntimeError("No embeddings to average.")

    dimension = len(embeddings[0])
    totals = [0.0] * dimension
    for embedding in embeddings:
        if len(embedding) != dimension:
            raise RuntimeError("Embedding dimension mismatch while averaging.")
        for index, value in enumerate(embedding):
            totals[index] += float(value)

    scale = 1.0 / len(embeddings)
    return [value * scale for value in totals]
