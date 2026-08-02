"""Image download and metadata extraction."""

import base64
import binascii
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import httpx

from backend.app.config import get_settings
from backend.app.utils.logging import get_logger
from backend.app.utils.tls import build_kb_ssl_context

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


class ImageProcessor:
    """Downloads and catalogs images from articles and local folders."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.images_dir = Path(self.settings.images_dir)
        self.static_dir = Path(self.settings.static_images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        # Same TLS handling as the crawler — images live on the same host
        # that serves the incomplete certificate chain.
        self._verify = build_kb_ssl_context()

    def _generate_image_id(self, source: str) -> str:
        return hashlib.sha256(source.encode()).hexdigest()[:16]

    def _build_record(
        self,
        *,
        image_id: str,
        filename: str,
        filepath: Path,
        static_path: str,
        caption: str,
        alt_text: str,
        article_id: str,
        category: Optional[str],
        article_title: str,
        source_url: str,
        context: str = "",
    ) -> dict[str, Any]:
        # The vector for an image is built from words alone, so the embed text
        # has to carry the article's subject. Keying only on "article {id}" made
        # every image of an article embed almost identically, which is why their
        # retrieval scores came back tied. `context` (the surrounding step text)
        # is what finally separates two screenshots from the *same* article.
        # Duplicates are dropped so a caption derived from alt text or from the
        # nearby prose does not get counted twice.
        seen: set[str] = set()
        ordered: list[str] = []
        for part in (article_title, caption, alt_text, context, category or "", "screenshot"):
            part = (part or "").strip()
            if part and part.lower() not in seen:
                seen.add(part.lower())
                ordered.append(part)
        embed_text = " | ".join(ordered)
        return {
            "image_id": image_id,
            "filename": filename,
            "filepath": str(filepath),
            "static_path": static_path,
            "caption": caption or article_title or f"Image from article {article_id}",
            "alt_text": alt_text,
            "article_id": article_id,
            "category": category,
            "keywords": ",".join(
                p for p in (article_id, category or "", article_title, alt_text) if p
            ),
            "source_url": source_url,
            "embed_text": embed_text,
        }

    def _save_data_uri(
        self,
        data_uri: str,
        article_id: str,
        alt_text: str,
        category: Optional[str],
        caption: str,
        article_title: str,
        context: str = "",
    ) -> Optional[dict[str, Any]]:
        """Decode and store an inline base64 image."""
        header, _, payload = data_uri.partition(",")
        if not payload:
            return None

        mime = header.split(";")[0].removeprefix("data:")
        ext = MIME_EXTENSIONS.get(mime, ".png")

        try:
            content = base64.b64decode(payload)
        except (binascii.Error, ValueError) as exc:
            logger.warning("Failed to decode inline image on article %s: %s", article_id, exc)
            return None

        # Hash the bytes: inline images have no URL to key on, and this also
        # collapses the same screenshot repeated across articles.
        image_id = self._generate_image_id(hashlib.sha256(content).hexdigest())
        filename = f"{article_id}_{image_id}{ext}"
        filepath = self.static_dir / filename
        filepath.write_bytes(content)

        return self._build_record(
            image_id=image_id,
            filename=filename,
            filepath=filepath,
            static_path=f"/static/images/{filename}",
            caption=(caption or alt_text or "").strip(),
            alt_text=alt_text,
            article_id=article_id,
            category=category,
            article_title=article_title,
            source_url=f"inline:{image_id}",
            context=context,
        )

    async def download_image(
        self,
        url: str,
        article_id: str,
        alt_text: str = "",
        category: Optional[str] = None,
        caption: Optional[str] = None,
        article_title: str = "",
        context: str = "",
    ) -> Optional[dict[str, Any]]:
        if url.startswith("data:"):
            return self._save_data_uri(
                url, article_id, alt_text, category, caption or "", article_title, context
            )

        try:
            async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=True, verify=self._verify
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "image" not in content_type and not any(
                    ext in url.lower() for ext in IMAGE_EXTENSIONS
                ):
                    return None

                ext = ".png"
                for image_ext in IMAGE_EXTENSIONS:
                    if image_ext in url.lower():
                        ext = image_ext
                        break

                image_id = self._generate_image_id(url)
                filename = f"{article_id}_{image_id}{ext}"
                filepath = self.static_dir / filename

                filepath.write_bytes(response.content)

                return self._build_record(
                    image_id=image_id,
                    filename=filename,
                    filepath=filepath,
                    static_path=f"/static/images/{filename}",
                    caption=(caption or alt_text or "").strip(),
                    alt_text=alt_text,
                    article_id=article_id,
                    category=category,
                    article_title=article_title,
                    source_url=url,
                    context=context,
                )
        except httpx.HTTPError as exc:
            logger.warning("Failed to download image %s: %s", url, exc)
            return None

    def scan_local_images(self) -> list[dict[str, Any]]:
        """Scan data/images folder for pre-existing images with metadata JSON."""
        results: list[dict[str, Any]] = []
        metadata_file = self.images_dir / "metadata.json"

        metadata_map: dict[str, dict[str, Any]] = {}
        if metadata_file.exists():
            metadata_map = json.loads(metadata_file.read_text(encoding="utf-8"))

        for path in self.images_dir.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rel_name = path.name
            meta = metadata_map.get(rel_name, {})
            image_id = meta.get("image_id") or self._generate_image_id(str(path))

            dest = self.static_dir / rel_name
            if not dest.exists():
                dest.write_bytes(path.read_bytes())

            results.append(
                {
                    "image_id": image_id,
                    "filename": rel_name,
                    "filepath": str(dest),
                    "static_path": f"/static/images/{rel_name}",
                    "caption": meta.get("caption", rel_name),
                    "alt_text": meta.get("alt_text", ""),
                    "article_id": meta.get("article_id"),
                    "category": meta.get("category"),
                    "keywords": meta.get("keywords", ""),
                    "source_url": meta.get("source_url"),
                    "embed_text": (
                        f"{meta.get('caption', rel_name)} | "
                        f"{meta.get('alt_text', '')} | "
                        f"{meta.get('category', '')} | "
                        f"article {meta.get('article_id', '')}"
                    ),
                }
            )
        return results
