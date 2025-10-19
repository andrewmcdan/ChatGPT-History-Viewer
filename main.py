import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from database import ChatHistoryDatabase


META_CONTENT_TYPES = {
    "thoughts",
    "reasoning_recap",
    "user_editable_context",
    "tether_quote",
    "computer_output",
    "execution_output",
    "system_error",
    "tether_browsing_display",
}

INLINE_PATTERN = re.compile(r"(\*\*.+?\*\*|__.+?__|`.+?`|\*[^*]+\*|_[^_]+_)")


APP_TITLE = "ChatGPT History Viewer"
DEFAULT_DB_PATH = Path("chat_history.db")
ALL_PROJECTS_OPTION = "All projects"


class ChatHistoryViewer(tk.Tk):
    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x720")

        self.db = ChatHistoryDatabase(db_path)
        self.conversation_rows: List[dict] = []

        self.search_var = tk.StringVar()
        self.show_meta_var = tk.BooleanVar(value=False)
        self.project_filter_var = tk.StringVar(value=ALL_PROJECTS_OPTION)
        self.status_var = tk.StringVar(value="Ready")
        self.conversation_title_var = tk.StringVar(value="Select a conversation to view its messages.")
        self.project_filter_map: Dict[str, Optional[str]] = {ALL_PROJECTS_OPTION: None}
        self.project_lookup: Dict[str, str] = {}
        self.projects: List[dict] = []

        self._build_ui()
        self._bind_events()

        self.refresh_project_choices()
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
        file_menu.add_command(label="Load projects.json...", command=self._select_projects_file)
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
        ttk.Checkbutton(
            search_frame,
            text="Show system/tool entries",
            variable=self.show_meta_var,
            command=self.on_toggle_meta,
        ).grid(row=0, column=4, padx=(12, 0))
        ttk.Label(search_frame, text="Project:").grid(row=0, column=5, padx=(12, 6))
        self.project_filter = ttk.Combobox(
            search_frame,
            state="readonly",
            textvariable=self.project_filter_var,
            values=[ALL_PROJECTS_OPTION],
            width=28,
        )
        self.project_filter.grid(row=0, column=6, sticky="ew")
        self.project_filter.current(0)
        self.project_filter.bind("<<ComboboxSelected>>", self.on_project_filter_changed)

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
            font=("Segoe UI", 11),
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
        self.conversation_text.tag_configure("role_user", foreground="#1c7ed6", font=("Segoe UI", 11, "bold"))
        self.conversation_text.tag_configure("role_assistant", foreground="#2f9e44", font=("Segoe UI", 11, "bold"))
        self.conversation_text.tag_configure("role_system", foreground="#f08c00", font=("Segoe UI", 11, "bold"))
        self.conversation_text.tag_configure("role_tool", foreground="#862e9c", font=("Segoe UI", 11, "bold"))
        self.conversation_text.tag_configure("role_other", foreground="#495057", font=("Segoe UI", 11, "bold"))

        self.conversation_text.tag_configure("body_text", foreground="#212529", font=("Segoe UI", 10), spacing3=8)
        self.conversation_text.tag_configure("timestamp", foreground="#868e96", font=("Segoe UI", 9, "italic"))
        self.conversation_text.tag_configure("meta_header", foreground="#868e96")
        self.conversation_text.tag_configure("meta_body", foreground="#495057", font=("Segoe UI", 10, "italic"))
        self.conversation_text.tag_configure("meta_placeholder", foreground="#adb5bd", font=("Segoe UI", 10, "italic"))

        self.conversation_text.tag_configure("md_heading1", font=("Segoe UI", 16, "bold"), spacing1=6, spacing3=6)
        self.conversation_text.tag_configure("md_heading2", font=("Segoe UI", 14, "bold"), spacing1=6, spacing3=6)
        self.conversation_text.tag_configure("md_heading3", font=("Segoe UI", 12, "bold"), spacing1=4, spacing3=4)
        self.conversation_text.tag_configure("md_bold", font=("Segoe UI", 10, "bold"))
        self.conversation_text.tag_configure("md_italic", font=("Segoe UI", 10, "italic"))
        self.conversation_text.tag_configure("md_inline_code", font=("Consolas", 10), background="#e9ecef")
        self.conversation_text.tag_configure(
            "md_codeblock",
            font=("Consolas", 10),
            background="#f1f3f5",
            lmargin1=20,
            lmargin2=20,
            spacing1=4,
            spacing3=6,
        )
        self.conversation_text.tag_configure(
            "md_blockquote",
            font=("Segoe UI", 10, "italic"),
            foreground="#495057",
            lmargin1=20,
            lmargin2=30,
        )
        self.conversation_text.tag_configure(
            "md_list_item",
            lmargin1=25,
            lmargin2=45,
        )

        self.conversation_text.tag_configure(
            "search_match",
            background="#ffe066",
            foreground="#000000",
        )

    def _bind_events(self) -> None:
        self.conversation_listbox.bind("<<ListboxSelect>>", self.on_conversation_selected)
        self.search_entry.bind("<Return>", self.on_search)

    # ---------------------------------------------------------------- actions

    def refresh_project_choices(self) -> None:
        rows = self.db.list_projects()
        self.projects = [
            {
                "project_id": row["project_id"],
                "name": row["name"],
                "description": row["description"],
                "create_time": row["create_time"],
                "update_time": row["update_time"],
            }
            for row in rows
        ]

        mapping: Dict[str, Optional[str]] = {ALL_PROJECTS_OPTION: None}
        values = [ALL_PROJECTS_OPTION]
        lookup: Dict[str, str] = {}

        for row in self.projects:
            project_id = row["project_id"]
            if not project_id:
                continue
            base_name = (row["name"] or "").strip()
            if not base_name:
                base_name = f"Project {project_id[:8]}"
            lookup[project_id] = base_name

            display_name = base_name
            suffix = 1
            id_hint = project_id[:8]
            while display_name in mapping:
                extra = id_hint if suffix == 1 else f"{id_hint}-{suffix}"
                display_name = f"{base_name} ({extra})"
                suffix += 1

            mapping[display_name] = project_id
            values.append(display_name)

        previous_selection = self.project_filter_var.get()
        self.project_filter_map = mapping
        self.project_lookup = lookup
        self.project_filter["values"] = values

        if previous_selection not in mapping:
            previous_selection = ALL_PROJECTS_OPTION
        self.project_filter_var.set(previous_selection)
        self.project_filter.set(previous_selection)

    def _current_project_id(self) -> Optional[str]:
        return self.project_filter_map.get(self.project_filter_var.get())

    def _project_label(self, project_id: Optional[str]) -> Optional[str]:
        if not project_id:
            return None
        label = self.project_lookup.get(project_id)
        if label:
            return label
        if isinstance(project_id, str) and project_id:
            return project_id[:8]
        return None

    def refresh_conversation_list(self, query: Optional[str] = None) -> None:
        project_id = self._current_project_id()
        if query:
            rows = self.db.search_conversations(query, project_id=project_id)
            status_message = f"Search results: {len(rows)} conversation(s) matching '{query}'."
        else:
            rows = self.db.list_conversations(project_id=project_id)
            status_message = f"Loaded {len(rows)} conversations."

        if project_id:
            project_label = self._project_label(project_id) or project_id
            status_message += f" (Project: {project_label})"

        self.status_var.set(status_message)

        self.conversation_rows = [
            {
                "conversation_id": row["conversation_id"],
                "project_id": row["project_id"],
                "title": row["title"],
                "create_time": row["create_time"],
                "update_time": row["update_time"],
            }
            for row in rows
        ]

        self.conversation_listbox.delete(0, tk.END)

        filtering_by_project = project_id is not None

        for row in self.conversation_rows:
            title = row["title"] or "(Untitled conversation)"
            timestamp = format_timestamp(row["update_time"]) or format_timestamp(row["create_time"]) or ""
            display_text = title
            project_label = self._project_label(row.get("project_id"))
            if project_label and not filtering_by_project:
                display_text = f"[{project_label}] {display_text}"
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

    def on_project_filter_changed(self, event: Optional[tk.Event] = None) -> None:
        selection = self.project_filter_var.get()
        if selection not in self.project_filter_map:
            self.project_filter_var.set(ALL_PROJECTS_OPTION)
        query = self.search_var.get().strip()
        self.refresh_conversation_list(query if query else None)

    def on_toggle_meta(self) -> None:
        self.on_conversation_selected()

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
        project_label = self._project_label(row.get("project_id"))

        header_parts = []
        if project_label:
            header_parts.append(f"[{project_label}]")
        header_parts.append(title)
        if timestamp:
            header_parts.append(timestamp)

        self.conversation_title_var.set("  —  ".join(header_parts))

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

        show_meta = self.show_meta_var.get()
        render_items = []

        for message in messages:
            role = (message["author_role"] or "other").lower()
            content_type = (message["content_type"] or "").lower()
            text_content = (message["text_content"] or "").strip()
            is_tool = role == "tool"
            is_meta = is_tool or role == "system" or content_type in META_CONTENT_TYPES

            if not show_meta:
                if is_tool:
                    render_items.append(
                        {
                            "role": role,
                            "timestamp": message["create_time"],
                            "body": "[Tool entry omitted]",
                            "placeholder": True,
                            "meta": True,
                            "content_type": content_type,
                        }
                    )
                    continue
                if is_meta:
                    continue

            if not text_content:
                # Skip empty fillers entirely.
                continue

            body_text = text_content
            if content_type == "code" and body_text:
                body_text = f"```\n{body_text}\n```"

            render_items.append(
                {
                    "role": role,
                    "timestamp": message["create_time"],
                    "body": body_text,
                    "placeholder": False,
                    "meta": is_meta,
                    "content_type": content_type,
                }
            )

        if not render_items:
            message_text = "No messages to display with the current filters."
            self.conversation_text.insert(tk.END, message_text, ("body_text",))
            self.conversation_text.config(state="disabled")
            return

        for item in render_items:
            role = item["role"]
            role_tag = f"role_{role}" if role in {"user", "assistant", "system", "tool"} else "role_other"
            header_tags = (role_tag,)
            if item["meta"]:
                header_tags = self._combine_tags(header_tags, ("meta_header",))

            timestamp = format_timestamp(item["timestamp"])
            header_line = role.capitalize()
            if show_meta and item["meta"] and item["content_type"]:
                header_line += f" ({item['content_type']})"
            if timestamp:
                header_line += f"  —  {timestamp}"

            self.conversation_text.insert(tk.END, header_line + "\n", header_tags)

            if item["placeholder"]:
                self.conversation_text.insert(
                    tk.END,
                    item["body"] + "\n\n",
                    ("body_text", "meta_placeholder"),
                )
                continue

            base_tags: Tuple[str, ...] = ("body_text",)
            if show_meta and item["meta"]:
                base_tags = self._combine_tags(base_tags, ("meta_body",))

            self._insert_markdown(item["body"], base_tags)
            self.conversation_text.insert(tk.END, "\n")

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

    # --------------------------------------------------------- markdown render

    @staticmethod
    def _combine_tags(*tag_groups: Tuple[str, ...]) -> Tuple[str, ...]:
        combined: List[str] = []
        for group in tag_groups:
            for tag in group:
                if tag not in combined:
                    combined.append(tag)
        return tuple(combined)

    def _insert_markdown(self, text: str, base_tags: Tuple[str, ...]) -> None:
        lines = text.splitlines()
        if not lines:
            self.conversation_text.insert(tk.END, "\n", base_tags)
            return

        in_code_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code_block:
                    in_code_block = False
                    self.conversation_text.insert(tk.END, "\n", base_tags)
                else:
                    in_code_block = True
                continue

            if in_code_block:
                code_tags = self._combine_tags(base_tags, ("md_codeblock",))
                self.conversation_text.insert(tk.END, line + "\n", code_tags)
                continue

            if not stripped:
                self.conversation_text.insert(tk.END, "\n", base_tags)
                continue

            if stripped.startswith("# "):
                heading_tags = ("md_heading1",)
                self.conversation_text.insert(tk.END, stripped[2:].strip() + "\n", heading_tags)
                continue
            if stripped.startswith("## "):
                heading_tags = ("md_heading2",)
                self.conversation_text.insert(tk.END, stripped[3:].strip() + "\n", heading_tags)
                continue
            if stripped.startswith("### "):
                heading_tags = ("md_heading3",)
                self.conversation_text.insert(tk.END, stripped[4:].strip() + "\n", heading_tags)
                continue
            if stripped.startswith(("> ", ">>")):
                quote_tags = self._combine_tags(base_tags, ("md_blockquote",))
                quote_text = stripped.lstrip(">").strip()
                self._insert_inline_segments(quote_text, quote_tags)
                self.conversation_text.insert(tk.END, "\n", quote_tags)
                continue
            if stripped.startswith(("- ", "* ")):
                bullet_tags = self._combine_tags(base_tags, ("md_list_item",))
                bullet_text = stripped[2:].strip()
                self.conversation_text.insert(tk.END, "• ", bullet_tags)
                self._insert_inline_segments(bullet_text, bullet_tags)
                self.conversation_text.insert(tk.END, "\n", bullet_tags)
                continue

            paragraph_tags = base_tags
            self._insert_inline_segments(line, paragraph_tags)
            self.conversation_text.insert(tk.END, "\n", paragraph_tags)

    def _insert_inline_segments(self, text: str, base_tags: Tuple[str, ...]) -> None:
        for segment_text, segment_tags in self._split_inline_segments(text, base_tags):
            self.conversation_text.insert(tk.END, segment_text, segment_tags)

    def _split_inline_segments(
        self,
        text: str,
        base_tags: Tuple[str, ...],
    ) -> List[Tuple[str, Tuple[str, ...]]]:
        segments: List[Tuple[str, Tuple[str, ...]]] = []
        index = 0

        for match in INLINE_PATTERN.finditer(text):
            start, end = match.span()
            if start > index:
                segments.append((text[index:start], base_tags))

            token = match.group(0)
            if token.startswith("**") and token.endswith("**"):
                segments.append((token[2:-2], self._combine_tags(base_tags, ("md_bold",))))
            elif token.startswith("__") and token.endswith("__"):
                segments.append((token[2:-2], self._combine_tags(base_tags, ("md_bold",))))
            elif token.startswith("`") and token.endswith("`"):
                segments.append((token[1:-1], self._combine_tags(base_tags, ("md_inline_code",))))
            elif token.startswith("*") and token.endswith("*"):
                segments.append((token[1:-1], self._combine_tags(base_tags, ("md_italic",))))
            elif token.startswith("_") and token.endswith("_"):
                segments.append((token[1:-1], self._combine_tags(base_tags, ("md_italic",))))
            else:
                segments.append((token, base_tags))

            index = end

        if index < len(text):
            segments.append((text[index:], base_tags))

        return segments

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

    def _select_projects_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select projects.json",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        self._import_projects_file(Path(file_path))

    def _import_projects_file(self, path: Path) -> None:
        self.status_var.set(f"Importing {path.name}...")

        def worker() -> None:
            try:
                def progress(current: int, total: int) -> None:
                    self.after(
                        0,
                        lambda: self.status_var.set(
                            f"Importing {path.name}: {current}/{total} projects..."
                        ),
                    )

                stats = self.db.import_projects_json(path, progress_callback=progress)
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: self._handle_project_import_error(path, exc))
            else:
                self.after(0, lambda: self._handle_project_import_success(path, stats))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_import_success(self, path: Path, stats: dict) -> None:
        inserted = stats.get("inserted", 0)
        updated = stats.get("updated", 0)
        skipped = stats.get("skipped", 0)
        self.status_var.set(
            f"Import complete: {inserted} inserted, {updated} updated, {skipped} skipped from {path.name}."
        )
        self.refresh_project_choices()
        self.refresh_conversation_list(self.search_var.get().strip() or None)

    def _handle_import_error(self, path: Path, error: Exception) -> None:
        messagebox.showerror("Import failed", f"Could not import {path}.\n\n{error}")
        self.status_var.set("Import failed.")

    def _handle_project_import_success(self, path: Path, stats: dict) -> None:
        inserted = stats.get("inserted", 0)
        updated = stats.get("updated", 0)
        skipped = stats.get("skipped", 0)
        self.status_var.set(
            f"Project import complete: {inserted} inserted, {updated} updated, {skipped} skipped from {path.name}."
        )
        self.refresh_project_choices()
        self.refresh_conversation_list(self.search_var.get().strip() or None)

    def _handle_project_import_error(self, path: Path, error: Exception) -> None:
        messagebox.showerror("Project import failed", f"Could not import {path}.\n\n{error}")
        self.status_var.set("Project import failed.")

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
