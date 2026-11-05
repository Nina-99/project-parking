# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""
Application settings via pydantic-settings.

Todas las variables de entorno se cargan acá.
"""
from __future__ import annotations

from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import field_validator, model_validator
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

    # Auth. The read API exposes plates, so it is not public.
    SECRET_KEY: str = "dev-secret-change-me"
    AUTH_COOKIE_NAME: str = "parking_session"
    AUTH_COOKIE_SECURE: bool = False  # true behind TLS
    AUTH_TOKEN_TTL_SECONDS: int = 8 * 3600
    OPERATOR_USER: str = "operator"
    OPERATOR_PASSWORD: str = "operator"
    LOGIN_RATE_LIMIT: int = 10  # attempts per window, per client IP
    LOGIN_RATE_WINDOW_SECONDS: int = 300

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
    # Channel the worker publishes gate events on and the API relays to browsers.
    EVENTS_CHANNEL: str = "parking:events"

    # YOLO
    YOLO_MODEL_PATH: str = "yolo26n.pt"
    YOLO_CONFIDENCE: float = 0.5

    # Vehicle attribute classifiers (tools/train_*_classifier.py output).    # The engine refuses to run when these are missing: a body type or colour
    # guessed without a model would be written to the database as fact.
    #
    # Point these at the *_best.pt files. The class order is read from the
    # *_metrics.json sitting beside the checkpoint, so a new taxonomy (furgon
    # merged away: 5 types) needs no code change here -- only a matching pair.
    TYPE_MODEL_PATH: str = (
        "/home/nina/FNI2025/ProjectParking/datasets/runs/resnet18_final_type_v3_best.pt"
    )
    COLOR_MODEL_PATH: str = (
        "/home/nina/FNI2025/ProjectParking/datasets/runs/resnet18_final_color_v5_best.pt"
    )

    # Where entry crops are written for the incident search thumbnails
    ENTRY_IMAGE_DIR: str = "/var/lib/smart-parking/entries"
    DETECTION_INTERVAL_MS: int = 500  # Sample frame every N ms

    # An active session older than this is shown to the operator as orphaned.
    # It is a display policy, not a state transition: no job writes it.
    STALE_SESSION_HOURS: float = 12.0

    # Reports group by the site's local day and hour, not by UTC. A vehicle that
    # enters at 21:00 in La Paz belongs to that day, not to the next UTC one.
    REPORT_TIMEZONE: str = "America/La_Paz"

    # Total occupancy has a denominator, but no table: it is a site constant.
    PARKING_CAPACITY: int = 0

    # Access geometry. A vehicle only becomes an event when its ground-contact
    # point reaches the camera's inside_zone, the same rule the offline extractor
    # uses. CAMERA_ZONES maps the capture camera id to its key in the file.
    ZONES_CONFIG_PATH: str = "zones_config.json"
    CAMERA_ZONES: str = "gate-1=channel16,gate-2=channel6"

    # A camera that has not sent a heartbeat in this window is shown as offline.
    CAMERA_OFFLINE_AFTER_SECONDS: float = 20.0

    # Rows older than this are purged by the maintenance task; crops follow.
    RETENTION_DAYS: int = 90

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

    @field_validator("REPORT_TIMEZONE")
    @classmethod
    def _valid_report_timezone(cls, value: str) -> str:
        """Fail at startup, not on the first report request.

        PostgreSQL accepts an invalid zone in a parameter and raises at query
        time; the settings object is the only place where the mistake is still
        cheap to see.
        """
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(
                f"REPORT_TIMEZONE is not a valid IANA zone: {value}"
            ) from error
        return value

    @model_validator(mode="after")
    def _require_a_real_secret_outside_debug(self) -> Self:
        """A demo default must not survive into a non-debug deployment."""
        if self.DEBUG:
            return self
        if self.SECRET_KEY in {
            "dev-secret-change-me",
            "change-me-to-a-random-secret-key",
        }:
            raise ValueError("SECRET_KEY must be set when DEBUG=false")
        if len(self.SECRET_KEY) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        if self.OPERATOR_PASSWORD == "operator":
            raise ValueError("OPERATOR_PASSWORD must be changed when DEBUG=false")
        return self

    # OCR. Off until a reader is measured on the frozen test set: an unread
    # plate must not close the wrong session.
    #
    # Measured 2026-11-01 on the curated plate-disjoint test (159 crops, 116
    # plates, no plate shared with training): 85.4% character accuracy, 53.5%
    # exact match (95% CI 45.7-61.2), after 2-epoch stage 1 + 100-epoch stage 2
    # with crop jitter. Both framings improve together: served 54.4%, tight
    # 51.8%. Still not enough to close a session automatically -- the equality
    # check is the guard, not per-read confidence.
    #
    # NOT comparable with the older 83.1% / 42.6%: that was measured on 101
    # Paddle-cut crops whose recording sessions also appeared in training, which
    # leaked plate identity. This test shares no plate with training, so the
    # task is strictly harder.
    OCR_ENABLED: bool = True
    OCR_READER: Literal["paddle", "crnn"] = "crnn"
    # Confidence gates a *read*, not a match: matching still requires the exit
    # and the entry to produce the same plate. Measured on 88 vehicle crops the
    # CRNN's confidence separates weakly (median 0.981 on correct reads, 0.835 on
    # wrong ones) and raising this to 0.9 only lifts per-read exact match from
    # 26% to 51% while dropping coverage to 44%. The threshold is kept permissive
    # because the equality check is the real guard.
    OCR_MIN_CONFIDENCE: float = 0.60
    OCR_CRNN_CHECKPOINT: str = (
        "/home/nina/FNI2025/ProjectParking/runs/ocr_v3_plus/crnn_stage2_best.pt"
    )
    # Single-class plate detector (tools/train_plate_detector.py). Empty means
    # the CRNN is handed the vehicle crop directly, which it cannot read.
    # Measured 2026-10-24 on 58 unseen recording blocks: recall 0.971,
    # precision 0.986, mAP@50 0.990. That is the detection half of H1.4.
    PLATE_DETECTOR_PATH: str = (
        "/home/nina/FNI2025/ProjectParking/plates_det/runs/plate_det/weights/best.pt"
    )
    OCR_USE_GPU: bool = False
    OCR_LANG: str = "latin"  # PaddleOCR language


settings = Settings()
