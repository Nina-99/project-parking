"""
WSGI config for ProjectParking.

Exposes the WSGI callable as a module-level variable named ``application``.
"""
from __future__ import annotations

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.prod")

application = get_wsgi_application()
