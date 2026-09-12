"""Timestamped speech persistence."""

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None


def upgrade():
    op.create_table(
        "transcripts",
        sa.Column("video_id", sa.String(36), primary_key=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("language", sa.String(20)),
        sa.Column("error", sa.Text()),
    )
    op.create_table(
        "segments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("video_id", sa.String(36), sa.ForeignKey("transcripts.video_id"), nullable=False),
        sa.Column("start", sa.Float(), nullable=False),
        sa.Column("end", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
    )
    op.create_index("ix_segments_video_id", "segments", ["video_id"])


def downgrade():
    op.drop_table("segments")
    op.drop_table("transcripts")
