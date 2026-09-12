from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCENEMIND_", env_file=".env", extra="ignore")
    cors_origins: list[str] = ["http://localhost:3000"]
    data_dir: Path = Path("data/videos")
    max_upload_bytes: int = Field(default=250 * 1024 * 1024, gt=0)
    sampling_interval: float = Field(default=5, ge=1, le=60)
    max_duration: float = Field(default=1800, gt=0)
    processing_timeout: int = Field(default=300, gt=0)


settings = Settings()
