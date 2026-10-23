"""Translate domain failures into HTTP once.

A route should raise the domain error and know nothing about status codes. If a
new transport appears (a CLI, a queue consumer), the same error reaches it with
the same meaning.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.domain.errors import (
    SessionNotActiveError,
    SessionNotFoundError,
    VehicleNotFoundError,
)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(SessionNotFoundError)
    async def _session_not_found(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse({"detail": "session not found"}, status_code=404)

    @app.exception_handler(SessionNotActiveError)
    async def _session_not_active(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse({"detail": "session is not active"}, status_code=409)

    @app.exception_handler(VehicleNotFoundError)
    async def _vehicle_not_found(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse({"detail": "vehicle not found"}, status_code=404)
