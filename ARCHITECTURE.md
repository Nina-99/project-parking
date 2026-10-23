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
| El cierre manual es el camino operativo | Mientras el OCR no tenga exact match medido, una suposición vaciaría el lote en silencio |
| El OCR vive detrás de un puerto con dos adapters | PaddleOCR es el baseline operativo; el CRNN es el modelo de la tesis. `OCR_ENABLED=false` hasta medirlos sobre el mismo test |

## Mapa mínimo

| Archivo | Qué resuelve |
|---|---|
| `backend/domain/events.py` | Agrupa frames en tracks y emite un evento por vehículo |
| `backend/domain/matching.py` | Cierra una sesión solo con una patente exacta y confiable |
| `backend/application/register_observation.py` | Único camino de escritura: vehículo, sesión, publicación |
| `backend/application/close_session.py` | Cierre manual auditado, cuando el modelo no puede probar la salida |
| `backend/adapters/outbound/ml/plate_reader.py` | Elige PaddleOCR o el CRNN según configuración |
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

- El OCR está cableado pero apagado (`OCR_ENABLED=false`). Los dos lectores
  comparten el puerto; falta medirlos sobre el mismo test de patentes reales y
  elegir default con evidencia. El checkpoint CRNN disponible tiene 0 % de exact
  match, así que hoy no podría cerrar sesiones ni con el flag encendido.
- PaddleOCR ya localiza y lee la patente dentro del crop del vehículo; el CRNN
  necesita que otro detector le entregue el recorte de la patente.
- Sin ReID: se midió que el backbone no separa identidades (19,9 % de pares
  distintos por encima de 0,95), así que un exit sin patente queda sin match y se
  cierra a mano.
- Los tests son unitarios. Falta una prueba de integración con Postgres real
  para el ciclo entrada → evento → sesión → salida/manual close.
- El estado de tracking se pierde al reiniciar el worker. Es visible como una
  sesión que queda abierta; la UI la muestra como huérfana pasadas 12 horas.
