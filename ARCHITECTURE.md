# Arquitectura del software

El sistema es un monolito modular: un dominio puro decide, la aplicación
orquesta, los adaptadores hablan con el mundo y FastAPI solo traduce HTTP. Hay
**un único camino de escritura** (captura → Celery → caso de uso) y **un único
camino de eventos** (Postgres → Redis → WebSocket → cache del navegador).

## Camino rápido

```bash
# Todo el stack local: PostgreSQL 5433, Redis 6379, API 8000, Vite 5173.
# Aplica las migraciones antes de arrancar la API.
./scripts/dev.sh up

./scripts/dev.sh status
./scripts/dev.sh down            # conserva la base
./scripts/dev.sh down --clean    # borra contenedores y volumen

# Alternativa: todo en Docker (PostgreSQL 5433, frontend 8080).
./scripts/dev.sh compose
./scripts/dev.sh compose-down
```

El stack del host usa 5433/6379 a propósito: 5432 suele estar ocupado por otra
base. La UI pide usuario y contraseña (default `operator` / `operator`; cambialo
en `.env`): el token vive en una cookie httpOnly y la API de lectura no responde
sin él. Verificación puntual: `curl http://127.0.0.1:8000/health/ready`.

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
| La telemetría se escribe con el vehículo | `/stats/model` describe la inferencia que produjo las filas reales, no un benchmark aparte |
| La API no arranca con el secreto de demo fuera de `DEBUG` | Un default cómodo no puede llegar a un despliegue real |
| El rate limit del login falla abierto | Un Redis caído no debe dejar al operador afuera del sistema que monitorea |
| La telemetría y los crops expiran a los 90 días | El disco no crece sin límite; el vehículo y la sesión se conservan |

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

# Integración: crea y borra <base>_test, aplica las migraciones reales.
# Se salta sola si no hay PostgreSQL alcanzable.
DATABASE_URL_SYNC=postgresql+psycopg://parking:parking@localhost:5432/parking \
  uv run pytest -m integration
```

## Interfaz

La UI usa tokens semánticos (`canvas`, `surface`, `ink`, `muted`, `accent`,
`live`, `warn`, `danger`) definidos dos veces en `frontend/src/index.css`. El
tema se elige en **Configuración** (claro / oscuro / sistema), se guarda en
`localStorage` y se aplica antes de montar React para evitar el flash. El claro
es el default porque es el que se lee en un proyector.

`scripts/seed_demo.py` siembra tráfico de demostración (vehículos, sesiones,
telemetría y crops) para ensayar la defensa o llenar la UI cuando las cámaras no
están conectadas.

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
