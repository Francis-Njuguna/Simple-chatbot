"""Knowledge base web crawler."""

import os
import re
import ssl
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union
from urllib.parse import urljoin

import certifi
import httpx
from bs4 import BeautifulSoup

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger
from backend.app.utils.text import clean_html, extract_images_from_html, extract_title
from backend.app.utils.tls import build_kb_ssl_context

logger = get_logger(__name__)

ARTICLE_PATTERN = re.compile(r"knowledgebase\.php\?article=(\d+)", re.IGNORECASE)
CATEGORY_PATTERN = re.compile(r"knowledgebase\.php\?category=(\d+)", re.IGNORECASE)

# Ensure certifi is the process-wide default for any library that reads these.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# ⚠️  DEBUG WORKAROUND — HARD OVERRIDE ⚠️
# The single httpx client factory below (`_new_client`) is the ONE code path
# every category- and article-crawl request goes through. Previous attempts to
# disable verification via KB_VERIFY_SSL only took effect if that env var was
# set, and the diagnostics showed it was NOT — so the client was still built
# with an SSLContext and kept throwing CERTIFICATE_VERIFY_FAILED.
#
# This flag forces verify=False on the real crawl constructor unconditionally,
# independent of env resolution, so the crawl can complete against
# helpdesk.amref.ac.ke's incomplete certificate chain.
#
# INSECURE (MITM-exploitable). This is a local debugging switch only — flip it
# back to False and rely on KB_VERIFY_SSL / KB_CA_BUNDLE before deploying.
_FORCE_DISABLE_TLS_VERIFY = True


def _resolve_verify() -> Union[ssl.SSLContext, bool]:
    """Resolve the httpx ``verify`` value for the crawler.

    ⚠️  SECURITY / DEBUG WORKAROUND ⚠️
    ---------------------------------
    ``helpdesk.amref.ac.ke`` serves an incomplete TLS chain (missing
    intermediate CA), which no CA bundle can validate. To let ingestion run
    against it during debugging, TLS verification can be turned OFF by setting
    ``KB_VERIFY_SSL=false`` in the environment, or by the hard override
    ``_FORCE_DISABLE_TLS_VERIFY`` above. When disabled, ``verify=False`` is
    passed to httpx (equivalent to an ``ssl`` context with ``CERT_NONE``), so
    the crawler accepts ANY certificate.

    This is INSECURE (vulnerable to man-in-the-middle) and must NOT be enabled
    in a production/deployed build. As a guardrail we REFUSE to disable
    verification when ``APP_ENV`` looks like production, falling back to the
    secure context instead.
    """
    settings = get_settings()

    disable_requested = _FORCE_DISABLE_TLS_VERIFY or (not settings.kb_verify_ssl)

    if disable_requested:
        env = (settings.app_env or "").strip().lower()
        if env in {"production", "prod", "staging"}:
            logger.error(
                "TLS verify disable requested (KB_VERIFY_SSL=%s, "
                "_FORCE_DISABLE_TLS_VERIFY=%s) but APP_ENV=%s — REFUSING to "
                "disable TLS verification in a production-like environment. "
                "Falling back to secure verification.",
                settings.kb_verify_ssl,
                _FORCE_DISABLE_TLS_VERIFY,
                settings.app_env,
            )
            return build_kb_ssl_context()

        logger.warning(
            "!!! TLS VERIFICATION DISABLED for the KB crawler "
            "(KB_VERIFY_SSL=%s, _FORCE_DISABLE_TLS_VERIFY=%s). This is a DEBUG "
            "workaround for helpdesk.amref.ac.ke's incomplete certificate chain "
            "and is INSECURE (MITM-exploitable). DO NOT ship this to production. !!!",
            settings.kb_verify_ssl,
            _FORCE_DISABLE_TLS_VERIFY,
        )
        # httpx accepts verify=False to disable cert + hostname checks entirely.
        return False

    # Secure path: certifi roots + auto-recovered intermediate.
    return build_kb_ssl_context()


def _log_tls_diagnostics(verify: object) -> None:
    """Print exactly which code + trust material is live, to prove the path."""
    disabled = verify is False
    logger.warning("=== CRAWLER TLS DIAGNOSTICS ===")
    logger.warning("crawler module file : %s", __file__)
    logger.warning("certifi.where()     : %s", certifi.where())
    logger.warning("SSL_CERT_FILE env   : %s", os.environ.get("SSL_CERT_FILE"))
    logger.warning("_FORCE_DISABLE_TLS  : %s", _FORCE_DISABLE_TLS_VERIFY)
    logger.warning(
        "verify value        : %s",
        "False (VERIFICATION OFF)" if disabled else type(verify).__name__,
    )
    logger.warning("===============================")


@dataclass
class ArticleData:
    article_id: str
    title: str
    category: Optional[str]
    url: str
    html: str
    text: str
    images: list[dict[str, str]] = field(default_factory=list)


class KnowledgeBaseCrawler:
    """Crawls the Amref Help Desk knowledge base.

    Uses the explicit category list from KB_CATEGORY_IDS in config/env:
        1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

    For each category page every article link is collected, then each
    article is fetched and parsed.
    """

    # Explicit category IDs to crawl (loaded from settings)
    CATEGORY_IDS: list[str] = []

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.kb_base_url
        self.index_url = self.settings.kb_index_url
        self.CATEGORY_IDS = self.settings.kb_category_id_list
        self._client_timeout = httpx.Timeout(30.0, connect=10.0)
        # SSLContext (verification on, with the server's intermediate recovered)
        # or False when disabled (DEBUG-only insecure mode).
        self._verify = _resolve_verify()
        _log_tls_diagnostics(self._verify)
        logger.info(
            "Crawler initialised with %d explicit categories: %s",
            len(self.CATEGORY_IDS),
            ", ".join(self.CATEGORY_IDS),
        )

    def _new_client(self) -> httpx.AsyncClient:
        """Create the AsyncClient used for EVERY category/article crawl request.

        This is the single, real constructor path. When verification is
        disabled we pass ``verify=False`` explicitly and silence the resulting
        InsecureRequestWarning noise so the log stays readable — the loud
        security warning in ``_resolve_verify`` already covers the implication.
        """
        if self._verify is False:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Explicit verify=False on the actual crawl client.
                return httpx.AsyncClient(timeout=self._client_timeout, verify=False)
        return httpx.AsyncClient(timeout=self._client_timeout, verify=self._verify)

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> str:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.text

    def _discover_links(self, html: str) -> tuple[set[str], set[str]]:
        """Extract article and category IDs from a page's HTML."""
        soup = BeautifulSoup(html, "lxml")
        article_ids: set[str] = set()
        category_ids: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(self.base_url, href)
            if "knowledgebase.php" not in full_url:
                continue
            article_match = ARTICLE_PATTERN.search(full_url)
            if article_match:
                article_ids.add(article_match.group(1))
            category_match = CATEGORY_PATTERN.search(full_url)
            if category_match:
                category_ids.add(category_match.group(1))

        for match in ARTICLE_PATTERN.finditer(html):
            article_ids.add(match.group(1))
        for match in CATEGORY_PATTERN.finditer(html):
            category_ids.add(match.group(1))

        return article_ids, category_ids

    def _extract_category(self, html: str, soup: Optional[BeautifulSoup] = None) -> Optional[str]:
        soup = soup or BeautifulSoup(html, "lxml")
        for el in soup.find_all(["span", "div", "p", "a"]):
            text = el.get_text(strip=True)
            if text.lower().startswith("category:"):
                return text.split(":", 1)[1].strip()
        breadcrumb = soup.select_one(".breadcrumb, .breadcrumbs")
        if breadcrumb:
            parts = [p.strip() for p in breadcrumb.get_text("|").split("|") if p.strip()]
            if len(parts) >= 2:
                return parts[-2]
        return None

    def _parse_article(self, article_id: str, html: str, url: str) -> ArticleData:
        soup = BeautifulSoup(html, "lxml")
        title = extract_title(html) or f"Article {article_id}"
        category = self._extract_category(html, soup)
        text = clean_html(html)
        images = extract_images_from_html(html, self.base_url)

        for heading in soup.find_all(["h1", "h2", "h3"]):
            heading_text = heading.get_text(strip=True)
            if heading_text and len(heading_text) > 3 and heading_text not in title:
                if len(heading_text) < len(title):
                    title = heading_text

        return ArticleData(
            article_id=article_id,
            title=title,
            category=category,
            url=url,
            html=html,
            text=text,
            images=images,
        )

    async def discover_all_article_ids(self) -> set[str]:
        """Collect article IDs by fetching each explicit category page.

        Only the categories defined in KB_CATEGORY_IDS are visited:
            1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15

        No dynamic link-following — every category is fetched directly.
        """
        discovered_articles: set[str] = set()

        async with self._new_client() as client:
            for cat_id in self.CATEGORY_IDS:
                cat_url = f"{self.base_url}/knowledgebase.php?category={cat_id}"
                try:
                    logger.info("Crawling category %s → %s", cat_id, cat_url)
                    cat_html = await self._fetch(client, cat_url)
                    article_ids, _ = self._discover_links(cat_html)
                    logger.info(
                        "Category %s — found %d article(s): %s",
                        cat_id,
                        len(article_ids),
                        ", ".join(sorted(article_ids, key=int)) if article_ids else "none",
                    )
                    discovered_articles.update(article_ids)
                except httpx.ConnectError as exc:
                    logger.error("CONNECT/TLS error crawling category %s: %r", cat_id, exc)
                except httpx.HTTPError as exc:
                    logger.warning("Failed to crawl category %s: %s", cat_id, exc)

        logger.info(
            "Discovered %d unique articles across %d categories",
            len(discovered_articles),
            len(self.CATEGORY_IDS),
        )
        return discovered_articles

    async def crawl_article(self, article_id: str) -> Optional[ArticleData]:
        url = f"{self.base_url}/knowledgebase.php?article={article_id}"
        async with self._new_client() as client:
            try:
                html = await self._fetch(client, url)
                return self._parse_article(article_id, html, url)
            except httpx.ConnectError as exc:
                logger.error("CONNECT/TLS error crawling article %s: %r", article_id, exc)
                return None
            except httpx.HTTPError as exc:
                logger.error("Failed to crawl article %s: %s", article_id, exc)
                return None

    async def crawl_all(self) -> list[ArticleData]:
        article_ids = await self.discover_all_article_ids()
        articles: list[ArticleData] = []
        for article_id in sorted(article_ids, key=int):
            article = await self.crawl_article(article_id)
            if article and article.text:
                articles.append(article)
                logger.info("Crawled article %s: %s", article_id, article.title)
        return articles
