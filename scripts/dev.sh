#!/usr/bin/env bash
#
# Local development entry point.
#
#   ./scripts/dev.sh up        datastores in Docker (5433/6379), migrate, API,
#                              worker and Vite, all in the background
#   ./scripts/dev.sh down      stop the processes and the dev containers
#   ./scripts/dev.sh down --clean   also remove containers and the database volume
#   ./scripts/dev.sh status    show processes, containers and readiness
#   ./scripts/dev.sh compose   whole stack with Docker Compose (5433/8080)
#   ./scripts/dev.sh compose-down
#
# The host stack uses 5433/6379 because 5432 is often taken by another project.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEV="$ROOT/.dev"
LOGS="$DEV/logs"
PIDS="$DEV/pids"
ENTRIES="$DEV/entries"
mkdir -p "$LOGS" "$PIDS" "$ENTRIES"

PG_NAME="parking-dev-pg"
REDIS_NAME="parking-dev-redis"
PG_VOLUME="parking_dev_pgdata"

PG_PORT="${PG_PORT:-5433}"
REDIS_PORT="${REDIS_PORT:-6379}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
COMPOSE_WEB_PORT="${COMPOSE_WEB_PORT:-8080}"

export DATABASE_URL="postgresql+asyncpg://parking:parking@127.0.0.1:${PG_PORT}/parking"
export DATABASE_URL_SYNC="postgresql+psycopg://parking:parking@127.0.0.1:${PG_PORT}/parking"
export CELERY_BROKER_URL="redis://127.0.0.1:${REDIS_PORT}/0"
export CELERY_RESULT_BACKEND="$CELERY_BROKER_URL"
export ENTRY_IMAGE_DIR="$ENTRIES"

usage() {
  sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Falta '$1' en el PATH." >&2
    exit 1
  }
}

port_busy() {
  ss -ltn 2>/dev/null | grep -qE "[:.]$1[[:space:]]"
}

need_free_port() {
  if port_busy "$1"; then
    echo "El puerto $1 está ocupado. Liberalo o cambiá PG_PORT/REDIS_PORT/API_PORT/WEB_PORT." >&2
    exit 1
  fi
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "$1"
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$1"
}

ensure_postgres() {
  if container_running "$PG_NAME"; then
    echo "PostgreSQL ya está corriendo en $PG_PORT"
    return
  fi
  if container_exists "$PG_NAME"; then
    need_free_port "$PG_PORT"
    docker start "$PG_NAME" >/dev/null
  else
    need_free_port "$PG_PORT"
    docker run -d --name "$PG_NAME" \
      -e POSTGRES_DB=parking \
      -e POSTGRES_USER=parking \
      -e POSTGRES_PASSWORD=parking \
      -p "${PG_PORT}:5432" \
      -v "${PG_VOLUME}:/var/lib/postgresql/data" \
      postgres:16 >/dev/null
  fi
  for _ in $(seq 1 30); do
    if docker exec -e PGPASSWORD=parking "$PG_NAME" \
      psql -h 127.0.0.1 -U parking -d parking -c 'select 1' >/dev/null 2>&1; then
      echo "PostgreSQL listo en $PG_PORT"
      sleep 1
      return
    fi
    sleep 1
  done
  echo "PostgreSQL no arrancó. Mirá: docker logs $PG_NAME" >&2
  exit 1
}

ensure_redis() {
  if container_running "$REDIS_NAME"; then
    echo "Redis ya está corriendo en $REDIS_PORT"
    return
  fi
  if container_exists "$REDIS_NAME"; then
    need_free_port "$REDIS_PORT"
    docker start "$REDIS_NAME" >/dev/null
  else
    need_free_port "$REDIS_PORT"
    docker run -d --name "$REDIS_NAME" -p "${REDIS_PORT}:6379" redis:7 >/dev/null
  fi
  for _ in $(seq 1 20); do
    if docker exec "$REDIS_NAME" redis-cli ping 2>/dev/null | grep -q PONG; then
      echo "Redis listo en $REDIS_PORT"
      return
    fi
    sleep 1
  done
  echo "Redis no arrancó. Mirá: docker logs $REDIS_NAME" >&2
  exit 1
}

migrate() {
  echo "Aplicando migraciones..."
  uv run alembic -c backend/alembic.ini upgrade head
}

start_process() {
  local name="$1"
  local command="$2"
  local pidfile="$PIDS/$name.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "$name ya estaba corriendo (pid $(cat "$pidfile"))"
    return
  fi
  nohup bash -c "$command" >"$LOGS/$name.log" 2>&1 &
  echo $! >"$pidfile"
  echo "$name iniciado (pid $(cat "$pidfile"), log .dev/logs/$name.log)"
}

stop_process() {
  local name="$1"
  local pidfile="$PIDS/$name.pid"
  [[ -f "$pidfile" ]] || return 0
  local pid
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
    echo "$name detenido"
  fi
  rm -f "$pidfile"
}

wait_http() {
  local url="$1"
  local attempts="$2"
  local label="$3"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS -o /dev/null "$url" 2>/dev/null; then
      echo "$label listo"
      return
    fi
    sleep 1
  done
  echo "$label no respondió. Revisá .dev/logs/" >&2
  exit 1
}

print_urls() {
  cat <<EOF

  API       http://localhost:${API_PORT}
  Docs      http://localhost:${API_PORT}/docs
  Frontend  http://localhost:${WEB_PORT}
  Login     operator / operator (defaults; cambialos en .env)
  Logs      .dev/logs/
  Parar     ./scripts/dev.sh down

EOF
}

up() {
  require docker
  require uv
  require bun
  require curl
  ensure_postgres
  ensure_redis
  migrate
  start_process api "cd '$ROOT' && exec uv run uvicorn backend.app.main:app --host 0.0.0.0 --port ${API_PORT}"
  start_process worker "cd '$ROOT' && exec uv run celery -A backend.worker.celery_app worker --loglevel=info --concurrency=1"
  start_process frontend "cd '$ROOT/frontend' && exec bun run dev -- --host 0.0.0.0 --port ${WEB_PORT}"
  wait_http "http://127.0.0.1:${API_PORT}/health/ready" 40 "API"
  wait_http "http://127.0.0.1:${WEB_PORT}/" 30 "Frontend"
  print_urls
}

down() {
  stop_process frontend
  stop_process worker
  stop_process api
  docker stop "$PG_NAME" "$REDIS_NAME" >/dev/null 2>&1 || true
  if [[ "${1:-}" == "--clean" ]]; then
    docker rm -f "$PG_NAME" "$REDIS_NAME" >/dev/null 2>&1 || true
    docker volume rm "$PG_VOLUME" >/dev/null 2>&1 || true
    echo "Contenedores y volumen de desarrollo eliminados."
  fi
}

status() {
  echo "== contenedores =="
  docker ps --filter "name=parking-dev-" --format '  {{.Names}}  {{.Status}}' 2>/dev/null || true
  echo "== procesos =="
  for name in api worker frontend; do
    local pidfile="$PIDS/$name.pid"
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      echo "  $name corriendo (pid $(cat "$pidfile"))"
    else
      echo "  $name detenido"
    fi
  done
  echo "== salud =="
  curl -s "http://127.0.0.1:${API_PORT}/health/ready" 2>/dev/null || echo "  API no responde"
  echo
  curl -s -o /dev/null -w "  frontend: HTTP %{http_code}\n" "http://127.0.0.1:${WEB_PORT}/" 2>/dev/null || true
}

compose_up() {
  require docker
  POSTGRES_PORT="$PG_PORT" FRONTEND_PORT="$COMPOSE_WEB_PORT" \
    docker compose up -d --build
  cat <<EOF

  API       http://localhost:${API_PORT}
  Frontend  http://localhost:${COMPOSE_WEB_PORT}
  Login     operator / operator (defaults; cambialos en .env)
  Parar     ./scripts/dev.sh compose-down

EOF
}

compose_down() {
  POSTGRES_PORT="$PG_PORT" FRONTEND_PORT="$COMPOSE_WEB_PORT" \
    docker compose down
}

case "${1:-}" in
  up) up ;;
  down) shift; down "${1:-}" ;;
  status) status ;;
  compose) compose_up ;;
  compose-down) compose_down ;;
  *) usage; exit 1 ;;
esac
