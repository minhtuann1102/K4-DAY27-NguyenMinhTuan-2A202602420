"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

Goals:

1. Use AsyncSqliteSaver so the graph can resume after a crash.
2. Define and emit an `AuditEntry` (common/schemas.py) for every meaningful step,
   so the full session can be replayed.
3. Verify with `uv run python -m audit.replay --thread <id>`.

Approach:
    - Read `node_fetch_pr` below — it is the one fully-worked example. Pay attention
      to *why* each AuditEntry field has the value it does at that step.
    - For every other node, you decide what to log. Field reference:
      `common/schemas.py:AuditEntry`. Helper: `risk_level_for(confidence)`.
    - Implement the one-line body of `audit()`. Everything else (graph wiring,
      checkpointer setup, interrupt/resume loop) is already done for you.

TODOs to complete (10 total): the audit() body, ONE AuditEntry per node for
analyze/route/commit/auto_approve/synthesize, TWO each for human_approval and
escalate (before + after the interrupt).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

import aiosqlite
from dotenv import load_dotenv
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_conn, db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.prompts import REVIEW_SYSTEM_PROMPT, review_user_message
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


async def audit(state, entry: AuditEntry, *, once: bool = False) -> None:
    """Write one structured AuditEntry row to the `audit_events` table.

    A node that calls interrupt() is re-run from the top when the graph
    resumes, so any row written *before* the interrupt would be logged twice.
    Pass once=True for those rows: it skips the write when this thread has
    already logged the same action.
    """
    if once:
        async with db_conn() as conn:
            async with conn.execute(
                "SELECT 1 FROM audit_events WHERE thread_id = ? AND action = ? LIMIT 1",
                (state["thread_id"], entry.action),
            ) as cur:
                if await cur.fetchone() is not None:
                    return
    await write_audit_event(
        thread_id=state["thread_id"],
        pr_url=state["pr_url"],
        entry=entry,
    )


# ─── Reference example — read this carefully ───────────────────────────────
async def node_fetch_pr(state):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    # We've only fetched the diff, not analyzed it. So:
    #   - confidence is unknown → 0.0
    #   - risk_level can't be derived from confidence yet → "med" as neutral default
    #   - decision is "pending" — nothing has been decided
    #   - reviewer_id is None — no human is involved at this stage
    #   - reason is a short human-readable summary of what happened
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }
# ───────────────────────────────────────────────────────────────────────────


async def node_analyze(state):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm(temperature=0).with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": review_user_message(
                state["pr_title"], state["pr_files"], state["pr_diff"],
            )},
        ])
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning[:200] if a.confidence_reasoning else None,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": a}


async def node_route(state):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision=decision,
        reason=f"confidence={c:.2f} → {decision}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"decision": decision}


async def node_human_approval(state):
    t0 = time.monotonic()
    a = state["analysis"]

    # Audit BEFORE the interrupt — reviewer hasn't responded yet.
    # once=True: this node re-runs from the top on resume (see audit()).
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="Waiting for human reviewer decision",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ), once=True)

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
        "draft_body": _render_comment_body(state),   # what "approve" would post
    })

    # Audit AFTER resume — we now know the reviewer's choice
    t1 = time.monotonic()
    choice = resp.get("choice", "reject")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval_response",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision=choice,
        reason=resp.get("feedback") or f"Reviewer chose: {choice}",
        execution_time_ms=int((time.monotonic() - t1) * 1000),
    ))
    return {
        "human_choice": resp.get("choice"),
        "human_feedback": resp.get("feedback"),
        "edited_body": resp.get("edited_body") if choice == "edit" else None,
    }


async def node_confirm_refined(state):
    """Escalation path, UI only: show the refined review and wait for a
    final confirm before anything is posted (README: Exercise 5, <58% row)."""
    t0 = time.monotonic()
    a = state["analysis"]

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="confirm_refined",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="Waiting for reviewer to confirm the refined review",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ), once=True)

    resp = interrupt({
        "kind": "refined_confirm",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "qa": state.get("escalation_answers") or {},
        "diff_preview": state["pr_diff"][:2000],
        "draft_body": _render_comment_body(state),
    })

    t1 = time.monotonic()
    choice = resp.get("choice", "reject")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="confirm_refined_response",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision=choice,
        reason=resp.get("feedback") or f"Reviewer chose: {choice}",
        execution_time_ms=int((time.monotonic() - t1) * 1000),
    ))
    return {
        "refined_choice": choice,
        "human_feedback": resp.get("feedback"),
        "edited_body": resp.get("edited_body") if choice == "edit" else None,
    }


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _comment_body(state) -> str:
    """The reviewer's edited comment if there is one, else the rendered review."""
    return (state.get("edited_body") or "").strip() or _render_comment_body(state)


def _post(state) -> tuple[str, str | None]:
    """Post the comment. Returns (final_action, comment_url)."""
    try:
        url = post_review_comment(state["pr_url"], _comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return "committed", url
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed", None


def _should_post(state) -> bool:
    """Decide whether the commit node posts a comment.

    Paths converging on commit:
      1. human_approval → commit   post on approve, or on edit with a rewritten body
      2. escalate → synthesize → commit
           - confirm gate on (UI): post only if the reviewer confirmed
             (`refined_choice` is set only by that gate)
           - confirm gate off (CLI): always post the refined review
    """
    edited = bool((state.get("edited_body") or "").strip())
    refined = state.get("refined_choice")
    if refined is not None:
        return refined == "approve" or (refined == "edit" and edited)
    if state.get("escalation_answers"):
        return True
    choice = state.get("human_choice")
    return choice == "approve" or (choice == "edit" and edited)


async def node_commit(state):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    update: dict = {}
    if _should_post(state):
        action, url = _post(state)
        if url:
            update["posted_comment_url"] = url
        if action == "committed":
            update["posted_comment_body"] = _comment_body(state)
    else:
        verdict = state.get("refined_choice") or state.get("human_choice")
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={verdict})")
        action = "rejected"
    human_involved = bool(
        state.get("human_choice") or state.get("refined_choice") or state.get("escalation_answers")
    )
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=state["analysis"].confidence,
        risk_level=risk_level_for(state["analysis"].confidence),
        reviewer_id=os.environ.get("GITHUB_USER") if human_involved else None,
        decision=action,
        reason=state.get("human_feedback") or ("escalation path" if state.get("escalation_answers") else None),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    update["final_action"] = action
    return update


async def node_auto_approve(state):
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action, url = _post(state)
    reason = f"High confidence ({a.confidence:.0%}), no human review needed"
    if action != "committed":
        reason += " — posting the comment failed"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=None,
        decision="auto",
        reason=reason,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    update = {"final_action": f"auto_{action}"}
    if url:
        update["posted_comment_url"] = url
        update["posted_comment_body"] = _comment_body(state)
    return update


async def node_escalate(state):
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?"]

    # Audit BEFORE the interrupt — reviewer hasn't answered yet.
    # once=True: this node re-runs from the top on resume (see audit()).
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="escalate",
        reason=f"Low confidence ({a.confidence:.0%}), asking {len(questions)} question(s)",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ), once=True)

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })

    # Audit AFTER resume — we now have answers
    t1 = time.monotonic()
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate_response",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="pending",
        reason=f"Reviewer answered {len(answers)} question(s)",
        execution_time_ms=int((time.monotonic() - t1) * 1000),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    orig = state["analysis"]
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": "Refine review with reviewer answers."},
            {"role": "user", "content": (
                f"Diff:\n{state['pr_diff']}\n\n"
                f"Original analysis:\n{orig.summary}\n\n"
                f"Risk factors:\n{chr(10).join(orig.risk_factors)}\n\n"
                f"Q&A:\n{qa}"
            )},
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        reviewer_id=os.environ.get("GITHUB_USER"),
        decision="pending",
        reason=f"Refined from {orig.confidence:.0%} → {refined.confidence:.0%} after Q&A",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": refined}


@asynccontextmanager
async def open_checkpointer():
    """AsyncSqliteSaver on the shared SQLite file (tables created if missing).

    Equivalent to ``AsyncSqliteSaver.from_conn_string(db_path())`` except that
    our graph-state type is allow-listed for deserialization: LangGraph warns
    about unregistered types today and will block them in a future release,
    which would break resuming a session that holds a ``PRAnalysis``.
    """
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("common.schemas", "PRAnalysis")])
    async with aiosqlite.connect(db_path()) as conn:
        cp = AsyncSqliteSaver(conn, serde=serde)
        await cp.setup()
        yield cp


def build_graph(checkpointer, confirm_refined: bool = False):
    """Build the review graph.

    confirm_refined=False (default, the CLI / README flow diagram):
        escalate → synthesize → commit        (refined review is posted)
    confirm_refined=True (the Streamlit app):
        escalate → synthesize → confirm_refined → commit
        the reviewer sees the refined review and must confirm before it posts.
    """
    g = StateGraph(ReviewState)
    nodes = [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit), ("escalate", node_escalate), ("synthesize", node_synthesize),
    ]
    if confirm_refined:
        nodes.append(("confirm_refined", node_confirm_refined))
    for name, fn in nodes:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    if confirm_refined:
        g.add_edge("synthesize", "confirm_refined")
        g.add_edge("confirm_refined", "commit")
    else:
        g.add_edge("synthesize", "commit")
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload):
    kind = payload["kind"]
    if kind in ("approval_request", "refined_confirm"):
        console.print(Panel.fit(
            payload["summary"], title=f"conf={payload['confidence']:.0%}", border_style="green",
        ))
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        resp = {"choice": choice, "feedback": console.input("Feedback: ").strip()}
        if choice == "edit":
            console.print("[dim]Draft comment:[/dim]\n" + payload.get("draft_body", ""))
            resp["edited_body"] = console.input("Edited comment (single line; blank = don't post): ").strip()
        return resp
    return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with open_checkpointer() as cp:
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume an existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
