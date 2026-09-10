"""Regression guard for ``scripts/_verify_categories.py``'s answer classifier.

``testpaths = ["tests"]`` makes ``scripts/`` structurally uncoverable, so the
sweep harness's one load-bearing assertion — is this answer real, a decline, or
a provider-outage error — would otherwise ship unverified. Its verdict decides
whether a category PASSes, so a bug there silently invents or hides failures.

It had exactly that bug: matching a decline marker *anywhere* in the text graded
four correct, fully-grounded answers as refusals, because rule 8 of
``SYSTEM_PROMPT`` asks the model to close with a scope caveat ("...that part is
not covered; contact the Help Desk"). The marker is evidence of compliance, not
refusal. It then had a second, quieter one: the model writes U+2019, so every
marker containing an apostrophe could never match live output at all.

Every ANSWER_* fixture below is a verbatim excerpt of a real response captured
from the live endpoint, typographic punctuation included, so these pin observed
behaviour rather than a guess about it.
"""

from __future__ import annotations

import pytest

from scripts._verify_categories import classify

# Correct answers that *contain* decline vocabulary — the false-positive class.
ANSWER_WITH_TRAILING_CAVEAT = """To turn on 2-step verification (multi-factor authentication) for your
Microsoft 365 account, follow the steps outlined in article 10:

1. **Sign in** to Microsoft 365 with your usual username and password.
2. Choose your preferred verification method: the Microsoft Authenticator app or SMS.

These steps enable MFA for your sign-in. If you need exact button labels
beyond what is shown here, refer to the full article."""

ANSWER_WITH_MIDTEXT_CAVEAT = """The knowledge base provides a documented procedure for resetting the
**student portal** password. If the SIS has a different reset flow, that part is
not covered in the supplied context.

1. Go to the student portal at https://students.amref.ac.ke/#.
2. Click "Forgot your password?"."""

# A real decline: refusal up front. Verbatim, including the U+2019 the model
# actually emits — a straight-quote fixture would pass while production text
# silently never matched.
ANSWER_DECLINE = (
    "I\u2019m sorry, but the capital of France is not covered in the available "
    "knowledge base. Please contact the AMREF Help Desk for assistance."
)

# The apostrophe trap on its own: every marker containing "'" was unreachable
# against real output, leaving the whole check resting on the few without one.
ANSWER_DECLINE_SMART_APOSTROPHE = (
    "I don\u2019t have information on where to check grades in the available "
    "knowledge base. Please contact the AMREF Help Desk."
)

# Rule 8 tells a decline to enumerate in-scope topics, so a refusal legitimately
# carries a numbered list. Any "has steps therefore answered" shortcut grades
# this OK and flips the offtopic class to a false PASS.
ANSWER_DECLINE_WITH_LIST = """That isn't covered in the knowledge base. I can help with:

1. LMS and Moodle login
2. Microsoft Authenticator and MFA
3. SMOWL proctoring"""

# What llm.py returns *as the answer* on provider failure instead of raising —
# the nine-day outage this harness exists to detect.
ANSWER_LLM_ERROR = (
    "I could not generate an answer just now. This is not a gap in the "
    "knowledge base — please check the server logs."
)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (ANSWER_WITH_TRAILING_CAVEAT, "OK"),
        (ANSWER_WITH_MIDTEXT_CAVEAT, "OK"),
        (ANSWER_DECLINE, "DECLINE"),
        (ANSWER_DECLINE_SMART_APOSTROPHE, "DECLINE"),
        (ANSWER_DECLINE_WITH_LIST, "DECLINE"),
        (ANSWER_LLM_ERROR, "LLM_ERR"),
    ],
    ids=[
        "trailing-caveat",
        "midtext-caveat",
        "real-decline",
        "smart-apostrophe-decline",
        "enumerated-decline",
        "provider-outage",
    ],
)
def test_classify(answer: str, expected: str) -> None:
    assert classify(answer) == expected


def test_outage_string_outranks_everything() -> None:
    """LLM_ERR must win over any other shape, or an outage reads as success.

    The error string is returned in the answer field, so it can arrive wrapped
    in whatever the template produces.
    """
    assert classify(f"1. Step one\n{ANSWER_LLM_ERROR}") == "LLM_ERR"
