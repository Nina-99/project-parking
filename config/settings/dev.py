"""
Django dev settings for ProjectParking.

Extends base.py with debug mode and SQLite fallback.
"""
from __future__ import annotations

from .base import *  # noqa: F403, F401

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# Allow all hosts in dev
ALLOWED_HOSTS = ["*"]

# SQLite fallback for local dev without PostgreSQL
# Override with DATABASE_URL in .env for PostgreSQL
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",  # noqa: F405
    }
}

# Django Debug Toolbar (optional)
INTERNAL_IPS = [
    "127.0.0.1",
]
