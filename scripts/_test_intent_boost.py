"""Quick verification that intent-aware boosting works end-to-end."""

import sys
from pathlib import Path

# Repo root on the path — modules import as `backend.app.*`, so the root, not
# `backend/`, is what has to be importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.rag.query_processing import process_query
from backend.app.rag.retriever import _boost_score


def test_boost_score():
    """Verify the boost scoring function."""
    print("Testing _boost_score()...")

    # Empty cases
    assert _boost_score("", ("password", "reset")) == 0.0
    assert _boost_score("some text", ()) == 0.0

    # Full match
    text = "To reset your password, go to the login portal"
    terms = ("password", "reset", "login", "portal")
    score = _boost_score(text, terms)
    assert score == 1.0, f"Expected 1.0, got {score}"

    # Partial match (case insensitive)
    text = "Your PASSWORD has expired"
    terms = ("password", "reset", "credentials")
    score = _boost_score(text, terms)
    assert abs(score - 1/3) < 0.01, f"Expected 0.33, got {score}"

    # Phrase match
    text = "Contact the help desk for Microsoft 365 support"
    terms = ("help desk", "Microsoft 365", "email")
    score = _boost_score(text, terms)
    assert abs(score - 2/3) < 0.01, f"Expected 0.67, got {score}"

    print("✓ _boost_score() works correctly")


def test_intent_detection():
    """Verify intent detection populates boost_terms."""
    print("\nTesting intent detection...")

    # Password reset intent
    processed = process_query(
        "I forgot my portal password",
        enable_normalization=True,
        enable_synonyms=True,
        enable_multi_query=True,
        max_variants=4
    )

    print(f"Query: {processed.original!r}")
    print(f"Normalized: {processed.normalized!r}")
    print(f"Intent names: {processed.intent_names}")
    print(f"Boost terms: {processed.boost_terms}")
    print(f"Procedural: {processed.procedural}")

    assert "password_reset" in processed.intent_names, \
        f"Expected password_reset intent, got {processed.intent_names}"
    assert "password" in processed.boost_terms, \
        f"Expected 'password' in boost_terms, got {processed.boost_terms}"
    assert processed.procedural is True, \
        "Expected procedural=True for password_reset"

    print("✓ Intent detection works")

    # Camera issue intent
    processed = process_query(
        "webcam not working during exam",
        enable_normalization=True,
        enable_synonyms=True,
        enable_multi_query=True,
        max_variants=4
    )

    print(f"\nQuery: {processed.original!r}")
    print(f"Intent names: {processed.intent_names}")
    print(f"Boost terms: {processed.boost_terms}")

    assert "camera_issue" in processed.intent_names
    assert any(term in processed.boost_terms for term in ["camera", "webcam", "SMOWL"])

    print("✓ Camera issue intent detected")


if __name__ == "__main__":
    test_boost_score()
    test_intent_detection()
    print("\n✅ All intent boost tests passed")
