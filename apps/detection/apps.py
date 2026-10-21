from __future__ import annotations

from django.apps import AppConfig


class DetectionConfig(AppConfig):
    """App configuration for the detection app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.detection"
    verbose_name = "Detection"
