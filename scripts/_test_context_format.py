"""Verify article-centric context assembly renders as intended."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.rag.retriever import HybridRetriever, RetrievedChunk


def _chunk(article_id, idx, text, title, score=0.8):
    return RetrievedChunk(
        chunk_id=f"{article_id}_{idx}",
        article_id=article_id,
        chunk_index=idx,
        text=text,
        title=title,
        url=f"https://kb.example/{article_id}",
        category="Accounts",
        summary=f"Overview of {title}." if idx == 0 else None,
        score=score,
    )


def main():
    r = HybridRetriever()

    chunks = [
        _chunk("1", 0, "[Password Reset]\n1. Open the portal.\n2. Click Forgot.", "Password Reset"),
        _chunk("1", 1, "[Password Reset]\n3. Check your email.\n4. Set a new password.", "Password Reset", 0.75),
        # Non-adjacent — should get a gap marker
        _chunk("1", 5, "[Password Reset]\nIf the link expires, request a new one.", "Password Reset", 0.7),
        _chunk("2", 0, "[Contact Help Desk]\nEmail helpdesk@amref.ac.ke", "Contact Help Desk", 0.6),
    ]

    out = r.format_context(chunks)
    print(out)

    assert out.count("Article 1: Password Reset") == 1, "title should appear once per article"
    assert out.count("Article 2: Contact Help Desk") == 1
    assert "[…]" in out, "gap marker missing between non-adjacent chunks"
    assert out.count("https://kb.example/1") == 1, "URL should appear once per article"
    assert "Overview of Password Reset." in out
    # Chunker header stripped from body (article header carries it now)
    assert "[Password Reset]\n1. Open" not in out

    print("\n" + "=" * 60)
    print("✅ article-centric context assembly verified")

    # Empty case
    assert "No relevant articles" in r.format_context([])
    print("✅ empty-context fallback intact")


if __name__ == "__main__":
    main()
