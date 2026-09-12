"""Run against a dedicated empty validation database, never a production database."""

import os
from uuid import uuid4

from sqlalchemy import delete, select

from app.config import settings
from app.database import engine, migrate
from app.jobs import claim, enqueue, finish, jobs


def main():
    url = os.environ["SCENEMIND_VALIDATION_DATABASE_URL"]
    if not url.startswith("postgresql"):
        raise ValueError("A dedicated PostgreSQL validation URL is required")
    settings.database_url = url
    migrate()
    with engine().connect() as conn:
        if conn.execute(select(jobs.c.id).limit(1)).first():
            raise ValueError("Validation database must have an empty jobs table")
    identifier = enqueue(str(uuid4()), "ingest")
    try:
        job = claim()
        assert job["id"] == identifier
        finish(job)
        with engine().connect() as conn:
            assert (
                conn.execute(select(jobs.c.status).where(jobs.c.id == identifier)).scalar_one()
                == "ready"
            )
    finally:
        with engine().begin() as conn:
            conn.execute(delete(jobs).where(jobs.c.id == identifier))
    print("PostgreSQL migrations, enqueue, claim and completion passed")


if __name__ == "__main__":
    main()
