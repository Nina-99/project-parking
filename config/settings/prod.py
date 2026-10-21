"""
Django production settings for ProjectParking.

Extends base.py with strict production defaults.
"""
from __future__ import annotations

from .base import *  # noqa: F403, F401

# SECURITY WARNING: keep the secret key used in production secret!
DEBUG = False

# PostgreSQL is required in production
# DATABASE_URL must be set in environment
# If DATABASE_URL is not set, this will raise ImproperlyConfigured
DATABASES = {
    "default": env.db_url("DATABASE_URL"),  # noqa: F405
}

# Security settings
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"

# Static files
STATIC_ROOT = BASE_DIR / "staticfiles"  # noqa: F405
