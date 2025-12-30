# ChatGPT History Viewer

ChatGPT History Viewer is a desktop utility for browsing exported ChatGPT conversations entirely offline. It uses Tkinter for the UI and SQLite for storage so you can import one or more `conversations.json` exports, deduplicate by `conversation_id`, and keep a searchable archive on your machine.

## Features
- Load any ChatGPT `conversations.json` export via the File menu.
- Optional ChatGPT Projects support: import `projects.json` to populate project names and filter conversations by project.
- Automatic deduplication: newer snapshots replace older records on import.
- Persisted storage in SQLite (`chat_history.db`) for quick startup and search.
- Conversation list sorted by most recent activity with timestamps.
- Full message viewer with role-aware styling (user/assistant/system/tool).
- Global text search across titles and message bodies with inline highlighting.
- Optional semantic search powered by embeddings and a SQLite vector extension.
- Attachment and code payloads flattened into readable text for searching.

## Requirements
- Python 3.11 or later (Tkinter and SQLite ship with the standard interpreter on Windows/macOS/Linux).

## Getting Started
1. **Install dependencies**  
   No third-party packages are required; Tkinter and SQLite are part of the Python standard library.

2. **Run the viewer**
   ```bash
   python main.py
   ```

3. **Import conversations**
   - Choose *File → Load conversations.json…*.
   - Pick one of your exported ChatGPT history files.
   - The status bar shows import progress and the conversation list refreshes when done.
   - (Optional) Load *projects.json* to enrich the UI with project names and enable project-level filtering.

4. **Browse & search**
   - Click a title in the left pane to render the full thread.
   - Use the search box to locate phrases across all conversations; matching text is highlighted in the detail pane.

## Configuration (.env)
- The app reads a `.env` file from the project root on startup (existing environment variables take priority).
- Edit `.env` to configure:
  - `OPENAI_API_KEY`, `OPENAI_EMBEDDINGS_MODEL`, `OPENAI_BASE_URL`
  - `CHAT_HISTORY_VECTOR_EXTENSION`, `CHAT_HISTORY_VECTOR_BACKEND`
  - `CHAT_HISTORY_EMBEDDING_MAX_CHARS`

## Semantic Search (Optional)
1. Install a SQLite vector extension and set `CHAT_HISTORY_VECTOR_EXTENSION` to its file path.
   - For multiple extensions (for example `vector0` + `vss0`), separate paths with `;` and list `vector0` first.
   - Supported backends: `sqlite-vec` (default) or `sqlite-vss` via `CHAT_HISTORY_VECTOR_BACKEND`.
2. Set `OPENAI_API_KEY` (and optionally `OPENAI_EMBEDDINGS_MODEL`).
   - Long messages are chunked before embedding; override the chunk size with `CHAT_HISTORY_EMBEDDING_MAX_CHARS` (default 8000).
3. Use *Tools -> Build embeddings...* to generate vectors for messages.
4. Toggle **Semantic** in the search bar to use embedding search.

## Data Storage
- The application stores imported data in `chat_history.db` in the project root.
- Projects (when imported) live in the same database alongside conversations and messages.
- You can safely delete the database to start fresh (you will need to re-import exports).

## Development Notes
- The schema is defined in `database.py`; all read/write access routes through `ChatHistoryDatabase` (including the optional `projects` table).
- UI behaviour and interaction wiring live in `main.py`.
- `python -m compileall .` can be used for a quick syntax check without launching the UI.

## Roadmap Ideas
- Migrate search to SQLite FTS for faster fuzzy matching.
- Surface tool call payloads, browser snapshots, or attachments in a richer detail pane.
- Bundle with PyInstaller/Briefcase for simple distribution without a Python install requirement.

## License
Specify the license that applies to this project (e.g., MIT) once decided.
