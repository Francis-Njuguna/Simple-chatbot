"""Unit tests for crawler link discovery."""

import re

from backend.app.ingest.crawler import ARTICLE_PATTERN, CATEGORY_PATTERN


def test_article_pattern() -> None:
    url = "https://helpdesk.amref.ac.ke/knowledgebase.php?article=11"
    match = ARTICLE_PATTERN.search(url)
    assert match is not None
    assert match.group(1) == "11"


def test_category_pattern() -> None:
    url = "https://helpdesk.amref.ac.ke/knowledgebase.php?category=2"
    match = CATEGORY_PATTERN.search(url)
    assert match is not None
    assert match.group(1) == "2"


def test_discover_from_html() -> None:
    html = """
    <a href="knowledgebase.php?article=9">Reset password</a>
    <a href="knowledgebase.php?category=3">Student Portal</a>
  """
    articles = set(re.findall(r"article=(\d+)", html))
    categories = set(re.findall(r"category=(\d+)", html))
    assert "9" in articles
    assert "3" in categories
