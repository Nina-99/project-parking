from __future__ import annotations

from django.apps import AppConfig


class VehiclesConfig(AppConfig):
    """App configuration for the vehicles app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.vehicles"
    verbose_name = "Vehicles"
