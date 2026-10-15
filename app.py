"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("🤖 HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    try:
        async def _list_threads():
            from common.db import db_conn
            async with db_conn() as conn:
                async with conn.execute(
                    """
                    SELECT thread_id, pr_url,
                           MIN(timestamp) AS started,
                           MAX(timestamp) AS last_event,
                           MAX(risk_level) AS worst_risk,
                           COUNT(*) AS events
                      FROM audit_events
                     GROUP BY thread_id, pr_url
                     ORDER BY MAX(timestamp) DESC
                     LIMIT 10
                    """
                ) as cur:
                    return await cur.fetchall()

        rows = asyncio.run(_list_threads())
        if rows:
            for r in rows:
                tid = r["thread_id"]
                pr_short = r["pr_url"].split("/pull/")
                pr_label = f"PR#{pr_short[1]}" if len(pr_short) > 1 else r["pr_url"][:30]
                risk_emoji = {"low": "🟢", "med": "🟡", "high": "🔴"}.get(r["worst_risk"], "⚪")
                if st.sidebar.button(
                    f"{risk_emoji} {pr_label} — {r['events']} events",
                    key=f"sess_{tid}",
                    use_container_width=True,
                ):
                    st.session_state.thread_id = tid
                    st.session_state.pr_url = r["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()
        else:
            st.caption("No sessions yet.")
    except Exception:
        st.caption("(No audit DB yet — run a review first)")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("🚀 Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"👤 Approval requested — confidence {conf:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))
    st.markdown(f"**Summary:** {payload['summary']}")

    comments = payload.get("comments", [])
    if comments:
        st.markdown("**Review comments:**")
        for c in comments:
            severity_emoji = {"blocker": "🔴", "issue": "🟠", "suggestion": "🟡", "nit": "⚪"}.get(c["severity"], "•")
            st.markdown(f"- {severity_emoji} **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("📄 Diff preview"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"🚨 Strong escalation — confidence {conf:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))

    risk_factors = payload.get("risk_factors", [])
    if risk_factors:
        st.error("⚠️ **Risk factors:** " + " · ".join(risk_factors))

    st.markdown(f"**Summary:** {payload['summary']}")
    st.markdown("---")
    st.markdown("**Please answer the following questions to help refine the review:**")

    questions = payload.get("questions", [])
    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, q in enumerate(questions):
            answers[q] = st.text_area(f"Q{i+1}: {q}", key=f"q_{i}", height=80)
        submitted_esc = st.form_submit_button("📤 Submit answers", type="primary")
        if submitted_esc:
            # validate all answered
            if all(v.strip() for v in answers.values()):
                return answers
            else:
                st.warning("Please answer all questions before submitting.")
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg
            )
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("🔍 Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result
    st.rerun()

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind")
    answer = None
    if kind == "approval_request":
        answer = render_approval_card(payload)
    elif kind == "escalation":
        answer = render_escalation_card(payload)

    if answer is not None:
        with st.spinner("⏳ Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url,
                st.session_state.thread_id,
                resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    analysis = final.get("analysis")

    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✅ **{action}** — comment posted to [{st.session_state.pr_url}]({st.session_state.pr_url})")
        if analysis:
            st.metric("Final confidence", f"{analysis.confidence:.0%}")
            st.markdown(f"**Summary:** {analysis.summary}")
    elif action == "rejected":
        st.warning("⛔ Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")

    st.caption(
        f"thread_id = `{st.session_state.thread_id}`  ·  "
        f"replay: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
