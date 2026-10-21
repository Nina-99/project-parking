# ============================================================
# Stage 1: Builder — install Python deps via uv
# ============================================================
FROM python:3.12-slim AS builder

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN pip install uv --quiet

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY backend/ ./backend/
COPY rtsp-capture/ ./rtsp-capture/

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
