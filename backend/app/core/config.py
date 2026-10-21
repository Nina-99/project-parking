"""
Application settings via pydantic-settings.

Todas las variables de entorno se cargan acá.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
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
    RTSP_CAMERAS: list[str] = []

    # OCR
    OCR_LANG: str = "en"  # PaddleOCR language


settings = Settings()
