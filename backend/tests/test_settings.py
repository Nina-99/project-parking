# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Settings must load from the .env the project actually ships.

Both failures this guards against were real: the camera list is documented as a
comma-separated string but pydantic-settings tried to JSON-decode it, and .env
also feeds docker-compose, whose keys are not application settings.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings, settings


def test_the_shipped_env_file_loads() -> None:
    """Importing settings at all is the assertion; extras must not raise."""
    assert isinstance(settings.RTSP_CAMERAS, list)


def test_camera_list_accepts_the_documented_comma_format() -> None:
    configured = Settings(RTSP_CAMERAS="gate-1=rtsp://a/1,gate-2=rtsp://b/2")
    assert configured.RTSP_CAMERAS == ["gate-1=rtsp://a/1", "gate-2=rtsp://b/2"]


def test_an_empty_camera_list_means_no_cameras() -> None:
    assert Settings(RTSP_CAMERAS="").RTSP_CAMERAS == []


def test_a_proper_list_is_left_alone() -> None:
    assert Settings(RTSP_CAMERAS=["rtsp://a/1"]).RTSP_CAMERAS == ["rtsp://a/1"]


def test_production_refuses_the_demo_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(DEBUG=False, SECRET_KEY="dev-secret-change-me",
                 OPERATOR_PASSWORD="s3cret")


def test_production_refuses_the_env_placeholder_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(DEBUG=False, SECRET_KEY="change-me-to-a-random-secret-key",
                 OPERATOR_PASSWORD="s3cret")


def test_production_refuses_a_short_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(DEBUG=False, SECRET_KEY="short", OPERATOR_PASSWORD="s3cret")


def test_production_refuses_the_default_operator_password() -> None:
    with pytest.raises(ValidationError):
        Settings(DEBUG=False, SECRET_KEY="x" * 32, OPERATOR_PASSWORD="operator")


def test_production_accepts_a_long_secret() -> None:
    configured = Settings(DEBUG=False, SECRET_KEY="x" * 32,
                          OPERATOR_PASSWORD="s3cret")

    assert configured.DEBUG is False
