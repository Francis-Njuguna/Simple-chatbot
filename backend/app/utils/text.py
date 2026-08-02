"""HTML and text utilities."""

import re
from typing import Any, Optional

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


# Chrome/boilerplate headings that are page furniture, not article titles.
# The KB template renders "Article Details" as an <h2> on every single page.
GENERIC_TITLES = {
    "article details",
    "knowledgebase",
    "knowledge base",
    "help desk",
    "amiu",
    "home",
    "suggested knowledgebase articles",
    "skip to main content",
}


def _is_generic_title(title: str) -> bool:
    return title.strip().lower() in GENERIC_TITLES


def extract_title(html: str) -> Optional[str]:
    """Best real article heading, or None.

    Order matters: <h1> is checked before <title> because the KB template
    sometimes fills <title> with the boilerplate "Article Details" while the
    <h1> still carries the true heading.
    """
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(strip=True)
        if text and not _is_generic_title(text):
            return text

    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        title = re.sub(r"\s*-\s*Help Desk.*$", "", title, flags=re.IGNORECASE)
        title = title.strip()
        if title and not _is_generic_title(title):
            return title

    # Last resort: the first non-generic heading of any level.
    for heading in soup.find_all(["h2", "h3", "h4"]):
        text = heading.get_text(strip=True)
        if text and len(text) > 3 and not _is_generic_title(text):
            return text

    return None


# Site chrome that appears on every article and carries no article meaning.
BOILERPLATE_IMAGE_NAMES = {"amiuhelp.png", "logo.png", "spacer.gif"}

# Template comment sitting just before the header logo in the DOM; it would
# otherwise be picked up as that image's "nearby text".
_TEMPLATE_NOISE = re.compile(r"custom code to be included", re.IGNORECASE)


_CAPTION_BLOCKS = ["p", "li", "td", "h1", "h2", "h3", "h4"]


def _nearby_caption(img: Any, max_chars: int = 160) -> str:
    """Derive a caption from the step text an image illustrates.

    The KB pages ship almost no alt text — instructions read "click X, as shown
    below" followed by the screenshot, so the surrounding prose is the best
    available description. Text is gathered at *block* level: the source HTML
    splits sentences across many inline tags, so individual text nodes are
    meaningless fragments (", Opera Mini) or search").
    """
    block = img.find_parent(["p", "li", "td", "div"])
    if block is None:
        return ""

    # The image's own block, when it holds a caption rather than just the <img>.
    own = " ".join(block.get_text(separator=" ").split())
    if 20 <= len(own) <= 400 and not _TEMPLATE_NOISE.search(own):
        return own[:max_chars].strip()

    for sibling in block.find_all_previous(_CAPTION_BLOCKS):
        text = " ".join(sibling.get_text(separator=" ").split())
        if len(text) >= 20 and not _TEMPLATE_NOISE.search(text):
            return text[:max_chars].strip()
    return ""


def _nearby_context(img: Any, max_chars: int = 400) -> str:
    """Collect the prose immediately around an image, for its embedding.

    Distinct from :func:`_nearby_caption`, which picks ONE short block to show
    the user. This gathers several blocks — before *and* after the image — to
    give the vector enough of the procedure to be semantically distinguishable.
    Screenshots on these pages are nearly captionless, so the surrounding steps
    are the only signal that separates "Authenticator setup" from "LMS login".

    Never rendered to the user; it only ever feeds ``embed_text``.
    """
    block = img.find_parent(["p", "li", "td", "div"])
    if block is None:
        return ""

    parts: list[str] = []
    seen: set[str] = set()

    def _take(node: Any) -> bool:
        """Append node text; return True once the budget is spent."""
        text = " ".join(node.get_text(separator=" ").split())
        if len(text) < 20 or _TEMPLATE_NOISE.search(text) or text in seen:
            return False
        seen.add(text)
        parts.append(text)
        return sum(len(p) for p in parts) >= max_chars

    # Two blocks of lead-in (the instruction the screenshot illustrates), then
    # the image's own block, then one block of follow-on.
    preceding = []
    for sibling in block.find_all_previous(_CAPTION_BLOCKS):
        preceding.append(sibling)
        if len(preceding) >= 2:
            break
    for sibling in reversed(preceding):
        if _take(sibling):
            break
    else:
        if not _take(block):
            for sibling in block.find_all_next(_CAPTION_BLOCKS):
                if _take(sibling):
                    break

    joined = " ".join(parts)
    return joined[:max_chars].strip()


def extract_images_from_html(html: str, base_url: str) -> list[dict[str, str]]:
    """Extract image URLs, alt text, derived captions and nearby context."""
    soup = BeautifulSoup(html, "lxml")
    images: list[dict[str, str]] = []
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if not src:
            continue

        # Inline base64 screenshots (article 11 serves all 13 of its images this
        # way). They are real 45-70KB PNGs, so keep the data URI intact and let
        # the image processor decode it rather than issue an HTTP request.
        is_data_uri = src.startswith("data:")
        if is_data_uri:
            if not src.startswith("data:image/"):
                continue
        elif src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = base_url.rstrip("/") + src
        elif not src.startswith("http"):
            src = base_url.rstrip("/") + "/" + src.lstrip("/")

        if not is_data_uri and src.rsplit("/", 1)[-1].lower() in BOILERPLATE_IMAGE_NAMES:
            continue

        alt = (img.get("alt", "") or img.get("title", "") or "").strip()

        # Prefer an explicit <figcaption>, then alt text, then nearby prose.
        caption = ""
        figure = img.find_parent("figure")
        if figure and figure.find("figcaption"):
            caption = figure.find("figcaption").get_text(strip=True)
        caption = caption or alt or _nearby_caption(img)

        images.append(
            {
                "url": src,
                "alt_text": alt,
                "caption": caption,
                "context": _nearby_context(img),
            }
        )
    return images
