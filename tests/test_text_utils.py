"""Unit tests for text utilities."""

from backend.app.utils.text import clean_html, extract_title


def test_clean_html_removes_scripts() -> None:
    html = "<html><script>alert('x')</script><body><p>Hello World</p></body></html>"
    text = clean_html(html)
    assert "Hello World" in text
    assert "alert" not in text


def test_clean_html_removes_nav() -> None:
    html = "<nav>Menu</nav><main><p>Article content here</p></main>"
    text = clean_html(html)
    assert "Article content" in text
    assert "Menu" not in text


def test_extract_title() -> None:
    html = "<html><head><title>Reset Password - Help Desk</title></head></html>"
    title = extract_title(html)
    assert title == "Reset Password"
