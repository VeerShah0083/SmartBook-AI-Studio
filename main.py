"""
main.py — CLI Runner
======================
Drives the LangGraph multi-agent textbook system from the terminal.

Responsibilities:
  • Bootstrap the graph with a minimal initial state.
  • Stream graph events and pretty-print automated node completions.
  • Detect interrupt() pauses, display the prompt, collect user input,
    and resume with Command(resume=<value>).
  • Loop until the graph reaches END or the user presses Ctrl-C.

How LangGraph interrupts work (quick reference)
-------------------------------------------------
1. A node calls interrupt(payload) — the graph saves a checkpoint and
   the current stream ends normally (StopIteration).
2. Call graph.get_state(config) to inspect the suspended state.
   state.tasks[i].interrupts[0].value  →  the payload passed to interrupt().
3. Collect user input, then call:
      graph.stream(Command(resume=user_input), config, stream_mode="updates")
   The node re-runs from the top; previously resolved interrupt() calls
   return their saved values instantly; the next unresolved one pauses again.
4. Repeat until state.next == () (graph complete).
"""

from __future__ import annotations

import sys
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.columns import Columns
from rich.table import Table
from langgraph.types import Command

console = Console(width=72)

# ─── Node display metadata ────────────────────────────────────────────────────

_NODE_META: dict[str, tuple[str, str, str]] = {
    # node_name: (icon, label, rich_colour)
    "scope_architect":     ("🎯", "Scope Architect",      "cyan"),
    "structural_scaffold": ("📐", "Structural Scaffold",  "blue"),
    "writer_gate":         ("🚦", "Writer Gate",          "yellow"),
    "targeted_researcher": ("🔬", "Targeted Researcher",  "magenta"),
    "writer_draft":        ("✍️ ", "Content Writer",       "green"),
    "review_editor":       ("📋", "QA Review Editor",     "bright_yellow"),
    "doc_exporter":        ("📦", "Document Exporter",    "bright_green"),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _print_header() -> None:
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("Multi-Agent Textbook Creation System\n", "bold cyan"),
                ("LangGraph  ·  Claude claude-sonnet-4-6  ·  6-Node Architecture\n", "dim"),
                ("\nNodes: Scope → Scaffold → Gate → Research → Draft → Review → Export",
                 "dim italic"),
            ),
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print()


def _print_node_start(node_name: str) -> None:
    if node_name.startswith("__"):
        return
    icon, label, colour = _NODE_META.get(node_name, ("⚙️ ", node_name, "white"))
    console.print(f"\n[{colour}]{icon}  {label}[/{colour}]  [dim]running…[/dim]")


def _print_automated_result(node_name: str, updates: dict) -> None:
    """Pretty-print the outcome of a fully automated node."""
    icon, label, colour = _NODE_META.get(node_name, ("⚙️ ", node_name, "white"))

    if node_name == "structural_scaffold":
        toc = updates.get("table_of_contents", [])
        table = Table(show_header=True, header_style="bold blue",
                      box=None, padding=(0, 2))
        table.add_column("Ch", style="dim", width=4)
        table.add_column("Title", style="bold")
        table.add_column("Focus", style="dim italic")
        for ch in toc:
            table.add_row(
                str(ch["chapter_id"]),
                ch["title"],
                ch["focus"][:55] + ("…" if len(ch["focus"]) > 55 else ""),
            )
        console.print(
            Panel(table,
                  title=f"[{colour}]{icon} {label} — {len(toc)} chapters[/{colour}]",
                  border_style=colour)
        )

    elif node_name == "targeted_researcher":
        console.print(
            f"  [{colour}]{icon}  {label}[/{colour}]  "
            f"[dim]→ research cache updated[/dim]"
        )

    elif node_name == "writer_draft":
        draft = updates.get("active_chapter_draft", "")
        console.print(
            f"  [{colour}]{icon}  {label}[/{colour}]  "
            f"[dim]→ draft complete ({len(draft):,} chars)[/dim]"
        )

    elif node_name == "doc_exporter":
        path = updates.get("export_path", "unknown")
        console.print(
            Panel(
                f"[bold bright_green]✅  Textbook compiled successfully![/bold bright_green]\n\n"
                f"📄  Saved to:  [underline]{path}[/underline]",
                border_style="bright_green",
                title="Export Complete",
                padding=(1, 3),
            )
        )


def _handle_interrupt(interrupt_payload) -> str:
    """
    Display the interrupt payload and collect user input.
    Accepts either a dict (with a "prompt" key) or a plain string.
    Returns the user's response as a string.
    """
    if isinstance(interrupt_payload, dict):
        node   = interrupt_payload.get("node", "")
        step   = interrupt_payload.get("step", "")
        prompt = interrupt_payload.get("prompt", str(interrupt_payload))
    else:
        node, step = "", ""
        prompt = str(interrupt_payload)

    _, _, colour = _NODE_META.get(node, ("", node, "white"))

    console.print()
    console.print(f"[{colour}]{prompt}[/{colour}]")

    # Collect input — strip leading/trailing whitespace
    try:
        raw = console.input(f"[bold {colour}]  ▶  [/bold {colour}]").strip()
    except EOFError:
        raw = ""

    # Default fallbacks for empty responses
    if not raw:
        if step in ("questionnaire_intro", "export_format"):
            return "A"
        if step == "binary_gate":
            return "APPROVE"
        return "A"

    return raw


def _stream(graph, input_data, config: dict) -> None:
    """
    Stream graph events.  For each event:
      - If the key is "__interrupt__" → the stream will end after this; ignore here.
      - If the key is a known automated node → pretty-print the result.
    Interrupts are handled in the outer loop via get_state().
    """
    for event in graph.stream(input_data, config, stream_mode="updates"):
        for node_name, updates in event.items():
            if node_name == "__interrupt__":
                continue    # Handled by outer loop
            if not isinstance(updates, dict):
                continue
            _print_automated_result(node_name, updates)


# ─── Main entry point ─────────────────────────────────────────────────────────

def run() -> None:
    from graph import build_graph

    _print_header()

    graph      = build_graph()
    
    import os
    if os.path.exists(".current_session"):
        with open(".current_session", "r") as f:
            thread_id = f.read().strip()
    else:
        thread_id = str(uuid.uuid4())
        with open(".current_session", "w") as f:
            f.write(thread_id)
            
    config     = {"configurable": {"thread_id": thread_id}}

    console.print(f"[dim]Session: {thread_id}[/dim]\n")

    # ── Check if we have an existing session ──────────────────────────────
    current_state = graph.get_state(config)
    
    if current_state.values:
        # We are resuming an existing crashed/paused session!
        console.print(f"[green]Resuming existing session from checkpoint...[/green]\n")
        _stream(graph, None, config)
    else:
        # ── Minimal bootstrap state ───────────────────────────────────────────
        initial_state: dict = {
            "subject":               "",
            "grade_level":           "",
            "target_reading_age":    0,
            "pedagogical_style":     "",
            "compliance_standards":  [],
            "scope_profile":         "",
            "table_of_contents":     [],
            "total_chapters":        0,
            "current_chapter_index": 0,
            "current_step":          "questionnaire",
            "approved_chapters":     [],
            "research_cache":        {},
            "active_chapter_draft":  None,
            "user_feedback_buffer":  None,
            "selected_activity_type": None,
            "export_format":         "word",
            "export_path":           None,
        }
        _stream(graph, initial_state, config)

    # ── HITL resume loop ──────────────────────────────────────────────────
    try:
        while True:
            graph_state = graph.get_state(config)

            # Collect all pending interrupts across tasks
            pending: list = []
            for task in graph_state.tasks:
                for intr in task.interrupts:
                    pending.append(intr)

            # Graph finished?
            if not graph_state.next and not pending:
                console.print()
                console.print(Rule("[bold green]Session Complete[/bold green]", style="green"))
                final = graph_state.values
                if final.get("export_path"):
                    console.print(f"\n[bold]Export saved to:[/bold] {final['export_path']}\n")
                else:
                    console.print(
                        "\n[yellow]Note: No export path found in state. "
                        "Check that doc_exporter ran successfully.[/yellow]\n"
                    )
                break

            if not pending:
                # Graph suspended between nodes (no interrupt) — resume automatically
                _stream(graph, Command(resume=None), config)
                continue

            # ── Handle the first pending interrupt ────────────────────────────
            payload    = pending[0].value
            user_input = _handle_interrupt(payload)

            # Show chapter progress if this looks like a chapter-level event
            if isinstance(payload, dict):
                ch_id = payload.get("chapter_id")
                total = graph_state.values.get("total_chapters", 0)
                if ch_id and total:
                    pct = int((ch_id - 1) / total * 100)
                    console.print(
                        f"\n  [dim]Chapter progress: {ch_id}/{total}  "
                        f"({'█' * (pct // 10)}{'░' * (10 - pct // 10)})  {pct}%[/dim]"
                    )

            # Resume graph with user input
            _stream(graph, Command(resume=user_input), config)
    except Exception as e:
        console.print_exception(show_locals=False)
        console.print("\n[bold red]⚠️ Fatal Error encountered. Rescuing drafted content...[/bold red]")
        state = graph.get_state(config).values
        # If a chapter was drafting/reviewing when it crashed, rescue it!
        if state.get("active_chapter_draft") and state.get("active_chapter_index") is not None:
            ch_idx = state["active_chapter_index"]
            ch_key = f"chapter_{ch_idx}"
            if "chapters_drafted" not in state:
                state["chapters_drafted"] = {}
            if ch_key not in state["chapters_drafted"]:
                state["chapters_drafted"][ch_key] = state["active_chapter_draft"]
                
        if "chapters_drafted" in state and state["chapters_drafted"]:
            with open("partial_draft.md", "w", encoding="utf-8") as f:
                for k in sorted(state["chapters_drafted"].keys()):
                    f.write(state["chapters_drafted"][k] + "\n\n")
            console.print("[green]✔ Successfully saved progress to partial_draft.md![/green]")
            
            # Try to export DOCX as well
            try:
                from nodes import _export_docx
                toc = []
                chapters = []
                for ch in state.get("table_of_contents", []):
                    ch_key = f"chapter_{ch['chapter_id']}"
                    if ch_key in state["chapters_drafted"]:
                        toc.append(ch)
                        chapters.append(state["chapters_drafted"][ch_key])
                if chapters:
                    docx_path = _export_docx(state, chapters, toc, ".", "partial_draft")
                    console.print(f"[green]✔ Successfully exported partial DOCX to {docx_path}![/green]")
            except Exception as docx_e:
                console.print(f"[yellow]Note: Could not generate partial DOCX: {docx_e}[/yellow]")
        sys.exit(1)
# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️   Session interrupted by user (Ctrl-C).[/yellow]\n")
        sys.exit(0)
    except Exception as exc:
        console.print_exception(show_locals=False)
        sys.exit(1)
