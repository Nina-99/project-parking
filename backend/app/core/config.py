"""
Application settings via pydantic-settings.

Todas las variables de entorno se cargan acá.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # .env also feeds docker-compose (POSTGRES_*, SECRET_KEY, ...). Those are
        # not application settings, and forbidding them made every import fail.
        extra="ignore",
    )

    # App
    PROJECT_NAME: str = "Parking Intelligence System"
    VERSION: str = "0.2.0"
    DEBUG: bool = True

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ]

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://parking:parking@localhost:5432/parking"

    # Redis / Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # YOLO
    YOLO_MODEL_PATH: str = "yolo26n.pt"
    YOLO_CONFIDENCE: float = 0.5
    DETECTION_INTERVAL_MS: int = 500  # Sample frame every N ms

    # RTSP — default cameras (overridable via .env)
    # NoDecode: the value is a comma-separated list, not JSON. Without it
    # pydantic-settings tries to json.loads() the raw string first and fails.
    RTSP_CAMERAS: Annotated[list[str], NoDecode] = []

    @field_validator("RTSP_CAMERAS", mode="before")
    @classmethod
    def _parse_cameras(cls, value: object) -> object:
        """Accept the documented comma-separated form; an empty value means none.

        pydantic would otherwise try to JSON-decode the raw string, so both an
        empty value and ``gate-1=rtsp://...,gate-2=...`` failed to load -- the
        documented format could never have worked.
        """
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    # OCR
    OCR_LANG: str = "en"  # PaddleOCR language


settings = Settings()
