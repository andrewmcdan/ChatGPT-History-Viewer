import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from database import ChatHistoryDatabase


APP_TITLE = "ChatGPT History Viewer"
DEFAULT_DB_PATH = Path("chat_history.db")


class ChatHistoryViewer(tk.Tk):
    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x720")

        self.db = ChatHistoryDatabase(db_path)
        self.conversation_rows: List[dict] = []

        self.search_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.conversation_title_var = tk.StringVar(value="Select a conversation to view its messages.")

        self._build_ui()
        self._bind_events()

        self.refresh_conversation_list()

    # ------------------------------------------------------------------ UI init

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_menu()
        self._build_search_bar()
        self._build_paned_layout()
        self._build_status_bar()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Load conversations.json...", command=self._select_json_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)

        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _build_search_bar(self) -> None:
        search_frame = ttk.Frame(self, padding=(10, 10, 10, 0))
        search_frame.grid(row=0, column=0, sticky="ew")
        search_frame.columnconfigure(1, weight=1)

        ttk.Label(search_frame, text="Search:").grid(row=0, column=0, padx=(0, 6))

        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky="ew")

        ttk.Button(search_frame, text="Search", command=self.on_search).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(search_frame, text="Clear", command=self.on_clear_search).grid(row=0, column=3, padx=(6, 0))

    def _build_paned_layout(self) -> None:
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Conversation list panel
        left_frame = ttk.Frame(paned)
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)

        self.conversation_listbox = tk.Listbox(
            left_frame,
            exportselection=False,
            activestyle="dotbox",
            height=20,
        )
        self.conversation_listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.conversation_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.conversation_listbox.config(yscrollcommand=scrollbar.set)

        paned.add(left_frame, weight=1)

        # Conversation detail panel
        right_frame = ttk.Frame(paned)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        title_label = ttk.Label(
            right_frame,
            textvariable=self.conversation_title_var,
            font=("Segoe UI", 12, "bold"),
        )
        title_label.grid(row=0, column=0, sticky="w", padx=(0, 0), pady=(0, 8))

        self.conversation_text = tk.Text(
            right_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 10),
            undo=False,
        )
        self.conversation_text.grid(row=1, column=0, sticky="nsew")

        text_scroll = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.conversation_text.yview)
        text_scroll.grid(row=1, column=1, sticky="ns")
        self.conversation_text.config(yscrollcommand=text_scroll.set)

        self._configure_text_tags()

        paned.add(right_frame, weight=3)

    def _build_status_bar(self) -> None:
        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(10, 6))
        status.grid(row=2, column=0, sticky="ew")

    def _configure_text_tags(self) -> None:
        self.conversation_text.tag_configure("role_user", foreground="#1c7ed6", font=("Consolas", 10, "bold"))
        self.conversation_text.tag_configure("role_assistant", foreground="#2f9e44", font=("Consolas", 10, "bold"))
        self.conversation_text.tag_configure("role_system", foreground="#f08c00", font=("Consolas", 10, "bold"))
        self.conversation_text.tag_configure("role_tool", foreground="#862e9c", font=("Consolas", 10, "bold"))
        self.conversation_text.tag_configure("role_other", foreground="#495057", font=("Consolas", 10, "bold"))

        self.conversation_text.tag_configure("body_text", foreground="#212529", font=("Consolas", 10))
        self.conversation_text.tag_configure("timestamp", foreground="#868e96", font=("Consolas", 9, "italic"))
        self.conversation_text.tag_configure(
            "search_match",
            background="#ffe066",
            foreground="#000000",
        )

    def _bind_events(self) -> None:
        self.conversation_listbox.bind("<<ListboxSelect>>", self.on_conversation_selected)
        self.search_entry.bind("<Return>", self.on_search)

    # ---------------------------------------------------------------- actions

    def refresh_conversation_list(self, query: Optional[str] = None) -> None:
        if query:
            rows = self.db.search_conversations(query)
            self.status_var.set(f"Search results: {len(rows)} conversation(s) matching '{query}'.")
        else:
            rows = self.db.list_conversations()
            self.status_var.set(f"Loaded {len(rows)} conversations.")

        self.conversation_rows = [
            {
                "conversation_id": row["conversation_id"],
                "title": row["title"],
                "create_time": row["create_time"],
                "update_time": row["update_time"],
            }
            for row in rows
        ]

        self.conversation_listbox.delete(0, tk.END)

        for row in self.conversation_rows:
            title = row["title"] or "(Untitled conversation)"
            timestamp = format_timestamp(row["update_time"]) or format_timestamp(row["create_time"]) or ""
            display_text = f"{title}"
            if timestamp:
                display_text += f"  —  {timestamp}"
            self.conversation_listbox.insert(tk.END, display_text)

        if self.conversation_rows:
            self.conversation_listbox.selection_clear(0, tk.END)
            self.conversation_listbox.selection_set(0)
            self.conversation_listbox.event_generate("<<ListboxSelect>>")
        else:
            self.clear_conversation_view()

    def on_search(self, event: Optional[tk.Event] = None) -> None:
        query = self.search_var.get().strip()
        if query:
            self.refresh_conversation_list(query)
        else:
            self.refresh_conversation_list()

    def on_clear_search(self) -> None:
        self.search_var.set("")
        self.refresh_conversation_list()

    def on_conversation_selected(self, event: Optional[tk.Event] = None) -> None:
        selection = self.conversation_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        if index >= len(self.conversation_rows):
            return

        row = self.conversation_rows[index]
        title = row["title"] or "(Untitled conversation)"
        timestamp = format_timestamp(row["update_time"]) or format_timestamp(row["create_time"])

        header = title if not timestamp else f"{title}  —  {timestamp}"
        self.conversation_title_var.set(header)

        messages = self.db.get_conversation_messages(row["conversation_id"])
        self._render_conversation_messages(messages)

    def clear_conversation_view(self) -> None:
        self.conversation_title_var.set("No conversations loaded yet.")
        self.conversation_text.config(state="normal")
        self.conversation_text.delete("1.0", tk.END)
        self.conversation_text.config(state="disabled")

    def _render_conversation_messages(self, messages: List) -> None:
        self.conversation_text.config(state="normal")
        self.conversation_text.delete("1.0", tk.END)

        if not messages:
            self.conversation_text.insert(tk.END, "No messages to display.", ("body_text",))
            self.conversation_text.config(state="disabled")
            return

        for message in messages:
            role = (message["author_role"] or "other").lower()
            role_tag = f"role_{role}" if role in {"user", "assistant", "system", "tool"} else "role_other"

            timestamp = format_timestamp(message["create_time"])
            header_line = role.capitalize()
            if timestamp:
                header_line += f"  —  {timestamp}"

            self.conversation_text.insert(tk.END, header_line + "\n", (role_tag,))

            body = (message["text_content"] or "").strip()
            if not body:
                body = "[no textual content]"
            self.conversation_text.insert(tk.END, body + "\n\n", ("body_text",))

        self._apply_search_highlight()
        self.conversation_text.config(state="disabled")

    def _apply_search_highlight(self) -> None:
        query = self.search_var.get().strip()
        self.conversation_text.tag_remove("search_match", "1.0", tk.END)
        if not query:
            return

        start = "1.0"
        while True:
            pos = self.conversation_text.search(query, start, tk.END, nocase=True)
            if not pos:
                break
            end = f"{pos}+{len(query)}c"
            self.conversation_text.tag_add("search_match", pos, end)
            start = end

    # ------------------------------------------------------------ file actions

    def _select_json_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select conversations.json",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        self._import_json_file(Path(file_path))

    def _import_json_file(self, path: Path) -> None:
        self.status_var.set(f"Importing {path.name}...")

        def worker() -> None:
            try:
                def progress(current: int, total: int) -> None:
                    self.after(
                        0,
                        lambda: self.status_var.set(
                            f"Importing {path.name}: {current}/{total} conversations..."
                        ),
                    )

                stats = self.db.import_conversations_json(path, progress_callback=progress)
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._handle_import_error(path, exc))
            else:
                self.after(0, lambda: self._handle_import_success(path, stats))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_import_success(self, path: Path, stats: dict) -> None:
        inserted = stats.get("inserted", 0)
        updated = stats.get("updated", 0)
        skipped = stats.get("skipped", 0)
        self.status_var.set(
            f"Import complete: {inserted} inserted, {updated} updated, {skipped} skipped from {path.name}."
        )
        self.refresh_conversation_list(self.search_var.get().strip() or None)

    def _handle_import_error(self, path: Path, error: Exception) -> None:
        messagebox.showerror("Import failed", f"Could not import {path}.\n\n{error}")
        self.status_var.set("Import failed.")

    # ---------------------------------------------------------------- cleanup

    def destroy(self) -> None:
        self.db.close()
        super().destroy()


def format_timestamp(timestamp: Optional[float]) -> str:
    if timestamp in (None, 0):
        return ""
    try:
        dt = datetime.fromtimestamp(float(timestamp))
    except (ValueError, OSError, TypeError):
        return ""
    return dt.strftime("%Y-%m-%d %H:%M")


def main() -> None:
    app = ChatHistoryViewer(DEFAULT_DB_PATH)
    app.mainloop()


if __name__ == "__main__":
    main()
