"""
graph.py — LangGraph Graph Assembly
=====================================
Wires all six agent nodes into a single compiled StateGraph.

Flow:
  START
    │
    ▼
  scope_architect        (Node 1  — HITL questionnaire)
    │
    ▼
  structural_scaffold    (Node 3  — automated TOC)
    │
    ▼
  ┌─► writer_gate        (Node 4a — HITL micro-approval)
  │     │
  │     ▼
  │   targeted_researcher (Node 2 — automated RAG)
  │     │
  │     ▼
  │   writer_draft        (Node 4b — automated drafting)
  │     │
  │     ▼
  │   review_editor       (Node 5  — HITL binary gate)
  │     │
  │   ┌─┴──────────────────────────────┐
  │   │                                │
  │ REVISE                          APPROVE
  │   │                                │
  └───┘                     ┌──────────┴──────────┐
                         more chapters?        last chapter?
                             │                     │
                             └─► writer_gate    doc_exporter  (Node 6)
                                                   │
                                                  END

Key design decisions
---------------------
• Node 4 is split into writer_gate + writer_draft so the HITL intercept
  and the bulk-generation step are separate checkpointed graph nodes.
• The conditional router after review_editor inspects two state fields:
    - current_step == "revise"  →  route back to writer_gate (same chapter)
    - current_chapter_index >= total_chapters  →  route to doc_exporter
    - otherwise  →  route to writer_gate (next chapter)
• MemorySaver is used for in-process checkpointing.  Swap for
  SqliteSaver / PostgresSaver for persistence across process restarts.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import TextbookSystemState
from nodes import (
    scope_architect,
    structural_scaffold,
    writer_gate,
    targeted_researcher,
    writer_draft,
    review_editor,
    doc_exporter,
)


# ─── Conditional router ───────────────────────────────────────────────────────

def _route_after_review(state: TextbookSystemState) -> str:
    """
    Called after review_editor returns.

    Three possible outcomes:
      "revise"    → same chapter must be rewritten; go back to writer_gate
      "approved"  → chapter committed; check whether more chapters remain
                    → writer_gate (next chapter) OR doc_exporter (done)
    """
    step  = state.get("current_step", "")
    idx   = state.get("current_chapter_index", 0)
    total = state.get("total_chapters", 1)

    if step == "revise":
        return "writer_gate"        # loop back — index NOT incremented

    # Chapter was approved; index was already incremented by review_editor
    if idx >= total:
        return "doc_exporter"       # all chapters done
    return "writer_gate"            # start next chapter


# ─── Graph factory ────────────────────────────────────────────────────────────

def build_graph():
    """
    Assemble, compile, and return the LangGraph StateGraph.

    Returns a compiled graph ready for streaming.
    The MemorySaver checkpointer enables interrupt/resume and state replay.
    """
    builder = StateGraph(TextbookSystemState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("scope_architect",      scope_architect)
    builder.add_node("structural_scaffold",  structural_scaffold)
    builder.add_node("writer_gate",          writer_gate)
    builder.add_node("targeted_researcher",  targeted_researcher)
    builder.add_node("writer_draft",         writer_draft)
    builder.add_node("review_editor",        review_editor)
    builder.add_node("doc_exporter",         doc_exporter)

    # ── Linear backbone ───────────────────────────────────────────────────
    builder.add_edge(START,                  "scope_architect")
    builder.add_edge("scope_architect",      "structural_scaffold")
    builder.add_edge("structural_scaffold",  "writer_gate")
    builder.add_edge("writer_gate",          "targeted_researcher")
    builder.add_edge("targeted_researcher",  "writer_draft")
    builder.add_edge("writer_draft",         "review_editor")

    # ── Conditional branch after review ───────────────────────────────────
    builder.add_conditional_edges(
        "review_editor",
        _route_after_review,
        {
            "writer_gate":   "writer_gate",
            "doc_exporter":  "doc_exporter",
        },
    )

    builder.add_edge("doc_exporter", END)

    # ── Compile with persistent checkpointer ──────────────────────────────
    import sqlite3
    conn = sqlite3.connect("state.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return builder.compile(checkpointer=checkpointer)
