"""Exercise 1 — Confidence scoring + routing.

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of
three terminal nodes by confidence. Goal: see the three branches print
different messages on different PRs.

"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import fetch_pr
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title, "pr_diff": pr.diff,
        "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    prompt = f"""You are a senior security-focused software engineer reviewing a pull request.

PR Title: {state['pr_title']}
Files changed: {', '.join(state['pr_files'])}

Diff:
{state['pr_diff']}

Analyze this PR thoroughly. Pay special attention to:
- Security vulnerabilities (weak hashing, SQL injection, hardcoded secrets, plaintext credentials)
- Authentication and authorization issues
- Data validation and sanitization problems
- Missing tests for new code

Confidence scoring guidelines:
- Score > 0.73: Simple, low-risk change with no security concerns (typos, docs, trivial refactors)
- Score 0.58–0.72: Moderate risk, needs human approval
- Score < 0.58: HIGH RISK — security issues, auth changes, SQL injection, weak crypto (MD5/SHA1 passwords), hardcoded tokens, no tests. You MUST escalate these.

If you detect security issues like MD5/SHA1 password hashing, SQL injection, plaintext token storage, or hardcoded credentials, your confidence MUST be below 0.58.

Populate escalation_questions with specific technical questions for the reviewer if confidence < 0.58."""

    with console.status("[dim]LLM thinking...[/dim]"):
        analysis = llm.invoke([{"role": "user", "content": prompt}])

    console.print(f"  [green]✓[/green] confidence = {analysis.confidence:.0%}")
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    confidence = state["analysis"].confidence
    if confidence >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif confidence < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision = {decision}")
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]✓ AUTO APPROVE[/green] — high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]✓ HUMAN APPROVAL[/yellow] — placeholder, exercise 2 will pause here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]✓ ESCALATE[/red] — placeholder, exercise 3 will ask the reviewer questions")
    return {"final_action": "pending_escalation"}


def build_graph():
    g = StateGraph(ReviewState)

    # Add nodes
    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("auto_approve", node_auto_approve)
    g.add_node("human_approval", node_human_approval)
    g.add_node("escalate", node_escalate)

    # Edges: START → fetch_pr → analyze → route
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")

    # Conditional routing from "route" node
    g.add_conditional_edges(
        "route",
        lambda state: state["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )

    # Terminal nodes → END
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", END)
    g.add_edge("escalate", END)

    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 — confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    final = app.invoke({"pr_url": args.pr})

    console.rule("Final")
    console.print(f"confidence = {final['analysis'].confidence:.0%}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()
