# Arquitectura del software

El sistema es un monolito modular: un dominio puro decide, la aplicación
orquesta, los adaptadores hablan con el mundo y FastAPI solo traduce HTTP. Hay
**un único camino de escritura** (captura → Celery → caso de uso) y **un único
camino de eventos** (Postgres → Redis → WebSocket → cache del navegador).

## Camino rápido

```bash
# 1. Dependencias y esquema
docker compose up -d postgres redis
uv run alembic -c backend/alembic.ini upgrade head

# 2. Procesos
uv run celery -A backend.worker.celery_app worker --loglevel=info --concurrency=1
uv run uvicorn backend.app.main:app --reload

# 3. Verificación
curl http://127.0.0.1:8000/health/ready
uv run pytest -q
```

La UI pide usuario y contraseña: el token vive en una cookie httpOnly y la API
de lectura no responde sin él.

## Capas

| Capa | Ruta | Puede importar | Responsabilidad |
|---|---|---|---|
| Dominio | `backend/domain/` | nada del proyecto | Matching, colores, ranking, agregación por track, puertos |
| Aplicación | `backend/application/` | dominio | Casos de uso: registrar una observación |
| Adaptadores | `backend/adapters/` | dominio + aplicación | SQLAlchemy, Redis, reloj, WebSocket, Celery |
| Entrada HTTP | `backend/app/` | adaptadores + aplicación | Routers, schemas, configuración, lifespan |
| Dominio web | `frontend/src/domain/` | nada | Tipos y funciones puras |
| Aplicación web | `frontend/src/application/` | dominio + puertos | Hooks de consulta, sesión, eventos en vivo |
| Infraestructura web | `frontend/src/infrastructure/` | puertos | `fetch`, WebSocket |
| Presentación | `frontend/src/presentation/` | aplicación + dominio | Páginas y componentes |

Regla de dependencia: **las flechas apuntan hacia el dominio**. `presentation`
nunca importa `infrastructure`; el único lugar que cablea adaptadores es
`frontend/src/composition.tsx`.

## Flujo de un vehículo

```
capture.py ──frame──▶ Celery ──▶ engine (YOLO + clasificadores)
                                      │
                                      ▼
                            TrackAggregator (dominio)
                                      │ un evento por track
                                      ▼
                        register_observation (aplicación)
                          │            │
                    UnitOfWork     EventPublisher
                          │            │
                     PostgreSQL      Redis
                                       │
                                  relay (API)
                                       │
                                  WebSocket
                                       │
                          react-query invalida la cache
```

Invariante central: **un track es un evento, un evento es un vehículo y a lo
sumo una sesión**. Un vehículo aparece en varios frames; persistir cada frame
infla la ocupación y todos los estadísticos derivados.

## Decisiones que no se negocian

| Decisión | Motivo |
|---|---|
| El evento se emite tras confirmar el track (3 frames) y lleva la caja más grande | Un frame no es un vehículo; el crop grande es el que sirve al operador |
| El commit pertenece al caso de uso | Una respuesta HTTP no puede construirse sobre una transacción que todavía puede fallar |
| Se publica después del commit | Un evento en el dashboard debe describir una fila que existe |
| No hay endpoint HTTP de ingesta | Dos caminos de escritura con semánticas distintas fue exactamente el problema anterior |
| El esquema lo aplica Alembic | `create_all` no tiene historia ni downgrade |
| Las rutas de archivos no salen del servidor | El cliente pide `/api/v1/vehicles/{id}/image`; el filesystem no se filtra |
| La API de lectura exige sesión | Expone patentes, que son datos personales |
| El worker corre con `--concurrency=1` | El estado de tracking es por proceso; escalar exige moverlo a un store compartido |

## Mapa mínimo

| Archivo | Qué resuelve |
|---|---|
| `backend/domain/events.py` | Agrupa frames en tracks y emite un evento por vehículo |
| `backend/domain/matching.py` | Cierra una sesión solo con una patente exacta y confiable |
| `backend/application/register_observation.py` | Único camino de escritura: vehículo, sesión, publicación |
| `backend/adapters/outbound/db/unit_of_work.py` | Una transacción por caso de uso |
| `backend/adapters/inbound/http/realtime.py` | Fan-out de Redis a los navegadores |
| `frontend/src/application/queries.ts` | Toda lectura del UI pasa por el puerto `ApiPort` |
| `frontend/src/presentation/pages/Incidents.tsx` | Búsqueda por ventana y color, con el motivo de cada candidato |

## Verificación

```bash
uv run pytest -q                      # dominio, casos de uso, rutas y auth
uv run ruff check backend
cd frontend && npx tsc --noEmit && bun run build
uv run alembic -c backend/alembic.ini upgrade head --sql   # migración sin DB
```

## Deuda conocida

- El OCR todavía no está cableado al camino de salida. El caso de uso ya acepta
  `Plate`; falta extraerlo del frame y pasarlo.
- Sin ReID: se midió que el backbone no separa identidades (19,9 % de pares
  distintos por encima de 0,95), así que un exit sin patente queda sin match.
- Los tests son unitarios. Falta una prueba de integración con Postgres real
  para el ciclo entrada → evento → sesión → salida.
- El estado de tracking se pierde al reiniciar el worker. Es visible como una
  sesión que queda abierta; la política de sesión huérfana todavía no existe.
