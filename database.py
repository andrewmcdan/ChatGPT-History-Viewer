import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


MessageTuple = Tuple[str, Optional[str], Dict[str, Any]]


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
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _ensure_schema(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT,
                    create_time REAL,
                    update_time REAL,
                    default_model_slug TEXT,
                    is_archived INTEGER,
                    is_starred INTEGER,
                    raw_metadata TEXT
                )
                """
            )
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
                CREATE INDEX IF NOT EXISTS idx_messages_text
                ON messages(text_content)
                """
            )
            self.conn.commit()

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

    def _upsert_conversation(self, conversation: Dict[str, Any]) -> str:
        conversation_id = (
            conversation.get("conversation_id") or conversation.get("id")
        )
        if not conversation_id:
            return "skipped"

        update_time = _as_float(conversation.get("update_time"))

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
            }

            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO conversations (
                        conversation_id,
                        title,
                        create_time,
                        update_time,
                        default_model_slug,
                        is_archived,
                        is_starred,
                        raw_metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
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

    # ------------------------------------------------------------------ queries

    def list_conversations(self) -> List[sqlite3.Row]:
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT conversation_id, title, create_time, update_time
                FROM conversations
                ORDER BY COALESCE(update_time, create_time, 0) DESC
                """
            )
            return list(cursor.fetchall())

    def search_conversations(self, query: str) -> List[sqlite3.Row]:
        pattern = f"%{query.lower()}%"
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT conversation_id, title, create_time, update_time
                FROM conversations
                WHERE LOWER(COALESCE(title, '')) LIKE ?
                   OR EXISTS (
                        SELECT 1 FROM messages
                        WHERE messages.conversation_id = conversations.conversation_id
                          AND LOWER(COALESCE(messages.text_content, '')) LIKE ?
                    )
                ORDER BY COALESCE(update_time, create_time, 0) DESC
                """,
                (pattern, pattern),
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
