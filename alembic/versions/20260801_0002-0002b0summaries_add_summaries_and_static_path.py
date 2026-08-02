"""Add article summaries + image static_path.

Revision ID: 0002b0summaries
Revises: 0001a0baseline
Create Date: 2026-08-01

Why
---
* ``summary`` — an EXTRACTIVE summary written during ingestion (no LLM call, so
  ingestion stays fast and offline-capable). Fed to the RAG prompt as
  article-level framing so the model can synthesise instead of pointing.
* ``llm_summary`` / ``llm_summary_at`` — OPTIONAL enrichment produced by a
  background pass. Never required for ingestion to succeed.
* ``static_path`` on images — Chroma previously held the only copy of the
  browser-servable ``/static/images/...`` path. With Postgres as the source of
  truth for metadata it belongs here; it is backfilled from ``filename``.

All columns are nullable, so this migration is safe on the populated database
and needs no downtime.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002b0summaries"
down_revision: Union[str, None] = "0001a0baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("document_metadata", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column("document_metadata", sa.Column("llm_summary", sa.Text(), nullable=True))
    op.add_column(
        "document_metadata",
        sa.Column("llm_summary_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.add_column("image_metadata", sa.Column("static_path", sa.String(length=1000), nullable=True))

    # Backfill: the served path is deterministic from the stored filename.
    op.execute(
        """
        UPDATE image_metadata
           SET static_path = '/static/images/' || filename
         WHERE static_path IS NULL
           AND filename IS NOT NULL
        """
    )


def downgrade() -> None:
    op.drop_column("image_metadata", "static_path")
    op.drop_column("document_metadata", "llm_summary_at")
    op.drop_column("document_metadata", "llm_summary")
    op.drop_column("document_metadata", "summary")
