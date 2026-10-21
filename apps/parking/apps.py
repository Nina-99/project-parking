from __future__ import annotations

from django.apps import AppConfig


class ParkingConfig(AppConfig):
    """App configuration for the parking app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.parking"
    verbose_name = "Parking"
