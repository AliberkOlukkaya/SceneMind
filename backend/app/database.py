from functools import lru_cache
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Float, ForeignKey, String, Text, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    pass


class Transcript(Base):
    __tablename__ = "transcripts"
    video_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(200))
    language: Mapped[str | None] = mapped_column(String(20))
    error: Mapped[str | None] = mapped_column(Text)


class Segment(Base):
    __tablename__ = "segments"
    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("transcripts.video_id"), index=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)


@lru_cache
def engine_for(url: str):
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database != ":memory:":
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        url,
        connect_args={"check_same_thread": False} if parsed.get_backend_name() == "sqlite" else {},
    )


def engine():
    return engine_for(settings.database_url)


def migrate():
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parents[1] / "migrations"))
    with engine().begin() as connection:
        if connection.dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        elif connection.dialect.name == "postgresql":
            connection.exec_driver_sql("SELECT pg_advisory_xact_lock(734220)")
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
