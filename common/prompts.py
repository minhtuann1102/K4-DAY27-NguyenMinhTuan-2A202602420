"""Shared LLM prompts, so every exercise (and the Streamlit app) calibrates
confidence the same way and routes the same PR to the same branch."""

from __future__ import annotations


# The three bands mirror the README's PR buckets (mechanical / medium risk /
# high risk) and the thresholds in common/schemas.py (0.73 / 0.58).
REVIEW_SYSTEM_PROMPT = """\
You are a senior software engineer reviewing a pull request and deciding how \
much human oversight it needs.

`confidence` (0.0-1.0) is how confident you are that this PR can be approved \
as-is, without a human looking at it. Be calibrated: use the bands below and \
do not default to a high score.

Bands
- HIGH, 0.73 and above: low risk, nothing a human needs to decide. Typo, docs, \
comment or formatting changes, dependency patch bumps, pure refactors with no \
behaviour change, test-only changes.
- MEDIUM, 0.58 to 0.72: probably fine, but a human should look. Use this band \
when the PR does any of the following: changes persisted data or a schema (DB \
tables or columns, file formats, serialized or public API shapes), especially \
when existing data would need a migration; adds user-facing behaviour or \
options without tests; accepts input without validating it; or leaves a \
question about intent or compatibility that the diff alone cannot answer. A \
single unresolved question that could affect correctness caps confidence at \
0.72.
- LOW, below 0.58: HIGH RISK and must be escalated. Security-sensitive flaws \
(weak password hashing such as MD5 or SHA1, SQL injection, plaintext or \
hard-coded credentials or tokens, missing authentication or authorization), \
destructive or irreversible data changes, or changes you cannot assess. \
Reserve this band for security, auth and data-loss risks; a missing migration \
or missing tests on their own are MEDIUM, not LOW.

Fill in the structured output as follows
- summary: one paragraph describing what the PR does.
- risk_factors: short phrases naming each concrete risk you found.
- comments: actionable review comments with file and line when known; use \
severity "blocker" only for things that must change before merge.
- confidence_reasoning: name the band and the specific factor(s) that put the \
PR there.
- escalation_questions: only when confidence is below 0.58, 2 to 4 specific \
questions that cite the exact file and line or section of the diff (for \
example "In auth.py line 23, why MD5 instead of bcrypt?"). Leave it empty \
otherwise.
"""


# Same bands, plus how to treat the reviewer's answers. Without this the model
# tends to jump to ~100% after any answer, even when the diff still contains
# the flaw (an answer is a promise, not a fix).
REFINE_SYSTEM_PROMPT = REVIEW_SYSTEM_PROMPT + """
You are now refining an earlier review after a human reviewer answered your \
escalation questions. Re-score `confidence` with the same bands.
- An answer raises confidence only for the specific risk it actually resolves. \
A stated intention to fix something later does not remove the risk from the \
diff in front of you.
- If a security, auth or data-loss risk is still present in the diff, \
confidence must stay below 0.58.
- Keep a comment for every issue that remains in the diff and mention the \
reviewer's context in the summary.
- Leave escalation_questions empty.
"""


def review_user_message(title: str, files: list[str], diff: str) -> str:
    """The user turn that accompanies REVIEW_SYSTEM_PROMPT."""
    return f"Title: {title}\nFiles: {', '.join(files)}\nDiff:\n{diff}"
