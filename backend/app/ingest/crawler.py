"""Knowledge base web crawler."""

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger
from backend.app.utils.text import clean_html, extract_images_from_html, extract_title

logger = get_logger(__name__)

ARTICLE_PATTERN = re.compile(r"knowledgebase\.php\?article=(\d+)", re.IGNORECASE)
CATEGORY_PATTERN = re.compile(r"knowledgebase\.php\?category=(\d+)", re.IGNORECASE)


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
    """Crawls the Amref Help Desk knowledge base."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.kb_base_url
        self.index_url = self.settings.kb_index_url
        self._client_timeout = httpx.Timeout(30.0, connect=10.0)

    async def _fetch(self, client: httpx.AsyncClient, url: str) -> str:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.text

    def _discover_links(self, html: str) -> tuple[set[str], set[str]]:
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
        """Discover every article ID by crawling index and category pages."""
        discovered_articles: set[str] = set()
        discovered_categories: set[str] = set()
        to_visit_categories: set[str] = set()

        async with httpx.AsyncClient(timeout=self._client_timeout) as client:
            index_html = await self._fetch(client, self.index_url)
            articles, categories = self._discover_links(index_html)
            discovered_articles.update(articles)
            to_visit_categories.update(categories)

            while to_visit_categories:
                cat_id = to_visit_categories.pop()
                if cat_id in discovered_categories:
                    continue
                discovered_categories.add(cat_id)
                cat_url = f"{self.base_url}/knowledgebase.php?category={cat_id}"
                try:
                    cat_html = await self._fetch(client, cat_url)
                    cat_articles, cat_categories = self._discover_links(cat_html)
                    discovered_articles.update(cat_articles)
                    to_visit_categories.update(cat_categories - discovered_categories)
                except httpx.HTTPError as exc:
                    logger.warning("Failed to crawl category %s: %s", cat_id, exc)

        logger.info(
            "Discovered %d articles across %d categories",
            len(discovered_articles),
            len(discovered_categories),
        )
        return discovered_articles

    async def crawl_article(self, article_id: str) -> Optional[ArticleData]:
        url = f"{self.base_url}/knowledgebase.php?article={article_id}"
        async with httpx.AsyncClient(timeout=self._client_timeout) as client:
            try:
                html = await self._fetch(client, url)
                return self._parse_article(article_id, html, url)
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
