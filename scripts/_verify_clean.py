"""Throwaway check: does clean_html still split words / keep boilerplate?"""
import re
import sys
import httpx

sys.path.insert(0, ".")
from backend.app.utils.text import clean_html, extract_title  # noqa: E402

BASE = "https://helpdesk.amref.ac.ke/knowledgebase.php"

# Single-letter-then-space runs, e.g. "T ype", "windo w", "r edi r ec t ed".
SPLIT = re.compile(r"\b([A-Za-z]) ([a-z]{1,3})\b")


def main() -> None:
    with httpx.Client(verify=False, timeout=30.0, follow_redirects=True) as client:
        for article in (1, 9, 11, 14):
            resp = client.get(BASE, params={"article": article})
            html = resp.text
            text = clean_html(html)
            splits = SPLIT.findall(text)
            print(f"--- article {article}  title={extract_title(html)!r}")
            print(f"    chars={len(text)}  letter-split-runs={len(splits)}  {splits[:8]}")
            lowered = text.lower()
            leaked = [
                marker
                for marker in (
                    "skip to main content",
                    "suggested knowledgebase articles",
                    "was this article helpful",
                    "article id:",
                    "go back",
                )
                if marker in lowered
            ]
            print(f"    boilerplate-leaked={leaked}")
            print(f"    head: {text[:220]!r}")
            print()


if __name__ == "__main__":
    main()
