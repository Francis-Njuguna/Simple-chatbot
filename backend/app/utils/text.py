"""HTML and text utilities."""

import re
from typing import Optional

from bs4 import BeautifulSoup


def clean_html(html: str) -> str:
    """Remove HTML tags, scripts, nav elements, and normalize whitespace."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()

    for selector in [
        ".navbar",
        ".nav",
        ".sidebar",
        ".menu",
        "#menu",
        ".breadcrumb",
        ".footer",
    ]:
        for el in soup.select(selector):
            el.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_title(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        title = re.sub(r"\s*-\s*Help Desk.*$", "", title, flags=re.IGNORECASE)
        return title.strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else None


def extract_images_from_html(html: str, base_url: str) -> list[dict[str, str]]:
    """Extract image URLs and alt text from HTML content."""
    soup = BeautifulSoup(html, "lxml")
    images: list[dict[str, str]] = []
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if not src or src.startswith("data:"):
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = base_url.rstrip("/") + src
        elif not src.startswith("http"):
            src = base_url.rstrip("/") + "/" + src.lstrip("/")
        alt = img.get("alt", "") or img.get("title", "") or ""
        images.append({"url": src, "alt_text": alt.strip()})
    return images
