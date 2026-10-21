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
    # The worker is synchronous, so it needs the non-async driver. It was only in
    # .env, never declared here, and extra="ignore" hid the gap until the worker
    # tried to use it.
    DATABASE_URL_SYNC: str = "postgresql+psycopg://parking:parking@localhost:5432/parking"

    # Redis / Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # YOLO
    YOLO_MODEL_PATH: str = "yolo26n.pt"
    YOLO_CONFIDENCE: float = 0.5

    # Vehicle attribute classifiers (tools/train_*_classifier.py output).
    # The engine refuses to run when these are missing: a body type or colour
    # guessed without a model would be written to the database as fact.
    TYPE_MODEL_PATH: str = "/mnt/nina/SmartParking/runs/resnet18_final_type_v2_best.pt"
    COLOR_MODEL_PATH: str = "/mnt/nina/SmartParking/checkpoints/resnet18_color_v3_best.pt"

    # Where entry crops are written for the incident search thumbnails
    ENTRY_IMAGE_DIR: str = "/var/lib/smart-parking/entries"
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
