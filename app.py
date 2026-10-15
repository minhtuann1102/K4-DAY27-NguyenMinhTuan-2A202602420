"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer,
                                  then the refined review for a final confirm

Sessions survive a page reload: the active thread id is mirrored into the
``?thread=<id>`` query parameter, and its state (pending question or final
result) is rebuilt from the LangGraph checkpointer in ``hitl_audit.db``.
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from common.db import db_conn, replay_events
from common.github import parse_pr_url
from exercises.exercise_4_audit import build_graph, open_checkpointer


load_dotenv()

st.set_page_config(page_title="HITL PR Review", layout="wide")


# ─── Session state ─────────────────────────────────────────────────────────
for _key, _default in {
    "thread_id": None,          # active LangGraph thread
    "pr_url": "",
    "interrupt_payload": None,  # payload of the pending interrupt(), if any
    "final": None,              # final graph state, once the run has finished
    "needs_continue": False,    # thread stopped without a pending question
    "restore_tried": False,     # ?thread= restored once per browser session
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ─── Graph access ──────────────────────────────────────────────────────────
def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def run_graph(
    pr_url: str,
    thread_id: str,
    resume_value=None,
    *,
    continue_existing: bool = False,
):
    """Invoke the graph once. Returns the final state or {'__interrupt__': ...}.

    - default:             start a new run for ``pr_url``
    - resume_value given:  answer the pending interrupt()
    - continue_existing:   re-enter a thread that stopped without a question
    """
    async with open_checkpointer() as cp:
        # confirm_refined: after an escalation the reviewer confirms the
        # refined review before anything is posted.
        app = build_graph(cp, confirm_refined=True)
        cfg = _cfg(thread_id)

        if continue_existing:
            return await app.ainvoke(None, cfg)
        if resume_value is not None:
            return await app.ainvoke(Command(resume=resume_value), cfg)
        return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)


async def load_thread(thread_id: str) -> dict:
    """Rebuild the UI state of a thread from the LangGraph checkpointer."""
    async with open_checkpointer() as cp:
        app = build_graph(cp, confirm_refined=True)
        snap = await app.aget_state(_cfg(thread_id))

    payload = None
    for task in snap.tasks:
        if task.interrupts:
            payload = task.interrupts[0].value
            break
    return {"values": dict(snap.values or {}), "payload": payload, "pending": bool(snap.next)}


def drive(coro_factory, spinner_text: str):
    """Run a graph call with a spinner; show an error instead of a traceback."""
    try:
        with st.spinner(spinner_text):
            return asyncio.run(coro_factory())
    except Exception as e:  # network, auth, LLM or DB errors
        st.error(f"{type(e).__name__}: {e}")
        return None


def apply_result(result: dict | None) -> None:
    """Store a graph result: either the next question or the final state."""
    if result is None:
        return
    st.session_state.needs_continue = False
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


def activate_thread(thread_id: str) -> bool:
    """Make ``thread_id`` the active session, restoring its state."""
    try:
        loaded = asyncio.run(load_thread(thread_id))
    except Exception as e:
        st.warning(f"Could not load session {thread_id}: {type(e).__name__}: {e}")
        return False

    values = loaded["values"]
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = values.get("pr_url", "")
    st.session_state.interrupt_payload = loaded["payload"]
    st.session_state.final = None
    st.session_state.needs_continue = False
    if loaded["payload"] is None:
        if loaded["pending"]:
            st.session_state.needs_continue = True   # stopped mid-run
        elif values:
            st.session_state.final = values          # finished
    st.query_params["thread"] = thread_id
    return True


# Restore the session after a full page reload (session_state starts empty).
if (
    st.session_state.thread_id is None
    and not st.session_state.restore_tried
    and st.query_params.get("thread")
):
    st.session_state.restore_tried = True
    activate_thread(st.query_params["thread"])


# ─── Page ──────────────────────────────────────────────────────────────────
st.title("🤖 HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
async def _list_threads():
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id, pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event,
                   -- MAX() on the text would order alphabetically (high < low < med)
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                        WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 10
            """
        ) as cur:
            return await cur.fetchall()


with st.sidebar:
    st.header("Recent sessions")
    try:
        rows = asyncio.run(_list_threads())
    except Exception as e:
        rows = []
        st.caption(f"Could not read the audit DB: {type(e).__name__}: {e}")
    else:
        if not rows:
            st.caption("No sessions yet.")

    for r in rows:
        tid = r["thread_id"]
        pr_short = r["pr_url"].split("/pull/")
        pr_label = f"PR#{pr_short[1]}" if len(pr_short) > 1 else r["pr_url"][:30]
        risk_emoji = {"low": "🟢", "med": "🟡", "high": "🔴"}.get(r["worst_risk"], "⚪")
        active = "▶ " if tid == st.session_state.thread_id else ""
        if st.button(f"{active}{risk_emoji} {pr_label} — {r['events']} events", key=f"sess_{tid}"):
            if activate_thread(tid):
                st.rerun()


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("🚀 Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
SEVERITY_EMOJI = {"blocker": "🔴", "issue": "🟠", "suggestion": "🟡", "nit": "⚪"}


def render_actions(payload: dict, key: str, *, approve_label: str, reject_label: str) -> dict | None:
    """Approve / Reject / Edit buttons. Returns the resume dict, or None.

    "Edit" opens an editor pre-filled with the comment that would be posted;
    the reviewer's rewritten text is what ends up on the PR.
    """
    feedback = st.text_input("Feedback (optional)", key=f"{key}_feedback")
    editing_key = f"{key}_editing"

    if st.session_state.get(editing_key):
        edited = st.text_area(
            "Edit the comment before posting",
            value=payload.get("draft_body", ""),
            height=260,
            key=f"{key}_edited",
        )
        col1, col2 = st.columns(2)
        if col1.button("📨 Post edited comment", type="primary", key=f"{key}_post_edit"):
            if edited.strip():
                return {"choice": "edit", "feedback": feedback, "edited_body": edited}
            st.warning("The edited comment is empty.")
        if col2.button("↩️ Cancel edit", key=f"{key}_cancel_edit"):
            st.session_state[editing_key] = False
            st.rerun()
        return None

    col1, col2, col3 = st.columns(3)
    if col1.button(approve_label, type="primary", key=f"{key}_approve"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button(reject_label, key=f"{key}_reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit", key=f"{key}_edit"):
        st.session_state[editing_key] = True
        st.rerun()
    return None


def render_comments_and_diff(payload: dict, *, expand_draft: bool) -> None:
    comments = payload.get("comments", [])
    if comments:
        st.markdown("**Review comments:**")
        for c in comments:
            emoji = SEVERITY_EMOJI.get(c["severity"], "•")
            st.markdown(f"- {emoji} **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    if payload.get("diff_preview"):
        with st.expander("📄 Diff preview"):
            st.code(payload["diff_preview"], language="diff")
    if payload.get("draft_body"):
        with st.expander("💬 Comment that will be posted", expanded=expand_draft):
            st.markdown(payload["draft_body"])


def render_approval_card(payload: dict, thread_id: str) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    st.subheader(f"👤 Approval requested — confidence {payload['confidence']:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))
    st.markdown(f"**Summary:** {payload['summary']}")
    render_comments_and_diff(payload, expand_draft=False)
    return render_actions(
        payload, f"{thread_id}_approval",
        approve_label="✅ Approve", reject_label="❌ Reject",
    )


def render_escalation_card(payload: dict, thread_id: str) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    st.subheader(f"🚨 Strong escalation — confidence {payload['confidence']:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))

    risk_factors = payload.get("risk_factors", [])
    if risk_factors:
        st.error("⚠️ **Risk factors:** " + " · ".join(risk_factors))

    st.markdown(f"**Summary:** {payload['summary']}")
    st.markdown("---")
    st.markdown("**Please answer the following questions to help refine the review:**")

    questions = payload.get("questions", [])
    with st.form(f"escalation_{thread_id}"):
        answers: dict[str, str] = {}
        for i, q in enumerate(questions):
            answers[q] = st.text_area(f"Q{i + 1}: {q}", key=f"{thread_id}_q_{i}", height=80)
        if st.form_submit_button("📤 Submit answers", type="primary"):
            if all(v.strip() for v in answers.values()):
                return answers
            st.warning("Please answer all questions before submitting.")
    return None


def render_refined_card(payload: dict, thread_id: str) -> dict | None:
    """After escalation: show the refined review and ask for a final confirm."""
    st.subheader(f"🧾 Confirm the refined review — confidence {payload['confidence']:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))
    st.markdown(f"**Summary:** {payload['summary']}")

    qa = payload.get("qa") or {}
    if qa:
        with st.expander("🗣️ Your answers to the escalation questions"):
            for q, a in qa.items():
                st.markdown(f"**{q}**\n\n> {a}")

    render_comments_and_diff(payload, expand_draft=True)
    st.info("Nothing has been posted yet. Confirm to publish this review on the PR.")
    return render_actions(
        payload, f"{thread_id}_refined",
        approve_label="✅ Confirm & post", reject_label="🗑️ Discard",
    )


def render_final(final: dict, pr_url: str) -> None:
    action = final.get("final_action") or "?"
    analysis = final.get("analysis")
    comment_url = final.get("posted_comment_url") or pr_url

    if action in ("committed", "auto_committed"):
        st.success(f"✅ **{action}** — review comment posted to the PR.")
        if analysis:
            st.metric("Final confidence", f"{analysis.confidence:.0%}")
            st.markdown(f"**Summary:** {analysis.summary}")
        if comment_url:
            st.link_button("View comment on GitHub", comment_url)
    elif action in ("commit_failed", "auto_commit_failed"):
        st.error(
            f"❌ **{action}** — the review was produced but posting the comment failed. "
            "Check GITHUB_TOKEN (scope `public_repo`) and the PR URL."
        )
        if analysis:
            st.markdown(f"**Summary:** {analysis.summary}")
    elif action == "rejected":
        st.warning("⛔ Rejected — no comment posted.")
    else:
        st.info(f"final_action = {action}")


def render_timeline(thread_id: str) -> None:
    try:
        events = asyncio.run(replay_events(thread_id))
    except Exception:
        return
    if not events:
        return
    with st.expander(f"🧾 Audit timeline — {len(events)} events"):
        st.dataframe(
            [
                {
                    "time": e["timestamp"],
                    "action": e["action"],
                    "confidence": round(e["confidence"], 2),
                    "risk": e["risk_level"],
                    "decision": e["decision"],
                    "reviewer": e["reviewer_id"] or "-",
                    "ms": e["execution_time_ms"],
                    "reason": e["reason"] or "",
                }
                for e in events
            ],
            hide_index=True,
        )


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted:
    candidate = pr_url.strip()
    try:
        parse_pr_url(candidate)
    except ValueError:
        st.error("That doesn't look like a GitHub PR URL — expected .../owner/repo/pull/<number>.")
    else:
        new_thread = str(uuid.uuid4())
        st.session_state.pr_url = candidate
        st.session_state.thread_id = new_thread
        st.session_state.interrupt_payload = None
        st.session_state.final = None
        st.query_params["thread"] = new_thread

        result = drive(lambda: run_graph(candidate, new_thread), "🔍 Fetching PR + asking the LLM...")
        if result is None:
            # Keep the error on screen (no rerun) and offer to continue.
            st.session_state.needs_continue = True
        else:
            apply_result(result)
            st.rerun()

# A thread that stopped without a pending question (e.g. the LLM call failed).
if st.session_state.needs_continue and st.session_state.thread_id:
    st.warning("This session stopped before it finished.")
    if st.button("▶️ Continue from the last checkpoint", key="continue_btn"):
        result = drive(
            lambda: run_graph(
                st.session_state.pr_url, st.session_state.thread_id, continue_existing=True
            ),
            "⏳ Continuing...",
        )
        if result is not None:
            apply_result(result)
            st.rerun()

# Render the pending question, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    thread_id = st.session_state.thread_id
    answer = None
    kind = payload.get("kind")
    if kind == "approval_request":
        answer = render_approval_card(payload, thread_id)
    elif kind == "escalation":
        answer = render_escalation_card(payload, thread_id)
    elif kind == "refined_confirm":
        answer = render_refined_card(payload, thread_id)
    else:
        st.error(f"Unknown interrupt kind: {kind!r}")

    if answer is not None:
        result = drive(
            lambda: run_graph(st.session_state.pr_url, thread_id, resume_value=answer),
            "⏳ Resuming...",
        )
        if result is not None:   # on error the card stays so it can be resubmitted
            apply_result(result)
            st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    render_final(st.session_state.final, st.session_state.pr_url)

if st.session_state.thread_id:
    st.caption(
        f"thread_id = `{st.session_state.thread_id}`  ·  "
        f"replay: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
    render_timeline(st.session_state.thread_id)
