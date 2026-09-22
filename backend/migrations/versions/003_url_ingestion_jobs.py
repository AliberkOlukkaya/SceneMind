"""Classify durable job failures for bounded retries."""

import sqlalchemy as sa
from alembic import op

revision = "003"
down_revision = "002"


def upgrade():
    op.add_column("jobs", sa.Column("retryable", sa.Boolean(), nullable=True))


def downgrade():
    op.drop_column("jobs", "retryable")
