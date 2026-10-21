# Simulación end-to-end con los videos grabados

Valida la cadena completa **sin el portón físico**: video grabado → RTSP → captura
→ Celery → detección → PostgreSQL → API.

```
videos .mp4 ──ffmpeg──▶ mediamtx ──RTSP──▶ capture.py ──Celery──▶ engine
                                                                  │
                          API ◀── PostgreSQL ◀── Vehicle + sesión ─┘
```

## Qué valida y qué no

**Sí**: descubrimiento y emparejamiento de horas entre cámaras, publicación RTSP de
las dos, decodificación a tiempo real, filtro de movimiento, despacho por Celery,
inferencia con los clasificadores entrenados, persistencia, y la consulta de
incidente devolviendo resultados reales.

**No**: la red (localhost no pierde paquetes ni reordena), ni la negociación
específica con la NVR real, ni el matcheo entrada/salida (necesita patente o ReID).

## Prerrequisitos

```bash
# mediamtx: binario único, no se versiona
curl -sL -o /tmp/m.tar.gz https://github.com/bluenviron/mediamtx/releases/download/v1.21.2/mediamtx_v1.21.2_linux_amd64.tar.gz
tar xzf /tmp/m.tar.gz -C /tmp mediamtx && install -m755 /tmp/mediamtx ~/.local/bin/mediamtx
```

Los videos van en `/mnt/nina/PARKING` (697 horas de `C.OCCIDENTE_ch16`, 429 de
`XVR_ch6`, **233 grabadas por las dos**).

## 1. Datastores

```bash
cd software
docker compose up -d postgres redis
```

Si el puerto 5432 ya está ocupado por otra base, no toques el compose: levantá el
postgres del simulador en otro puerto.

```bash
docker run -d --name parking-sim-pg \
    -e POSTGRES_DB=parking -e POSTGRES_USER=parking -e POSTGRES_PASSWORD=parking \
    -p 5433:5432 postgres:16

# tablas (el worker no las crea; el lifespan de la API sí)
DATABASE_URL=postgresql+asyncpg://parking:parking@localhost:5433/parking \
uv run python -c "
import asyncio, backend.app.models
from backend.app.core.database import Base, engine
async def main():
    async with engine.begin() as c: await c.run_sync(Base.metadata.create_all)
    await engine.dispose()
asyncio.run(main())"
```

## 2. Worker y API en el host

`.env` apunta a los nombres de servicio de compose (`redis`, `postgres`), que solo
resuelven dentro de la red de compose. Desde el host hay que overridear, o falla
la resolución de nombres y parece un problema de la simulación.

```bash
export CELERY_BROKER_URL=redis://localhost:6379/0
export CELERY_RESULT_BACKEND=redis://localhost:6379/0
export DATABASE_URL_SYNC=postgresql+psycopg://parking:parking@localhost:5433/parking
export ENTRY_IMAGE_DIR=/tmp/opencode/entries

uv run celery -A backend.worker.celery_app worker --loglevel=info --concurrency=1 &
DATABASE_URL=postgresql+asyncpg://parking:parking@localhost:5433/parking \
uv run uvicorn backend.app.main:app --port 8001 &
```

## 3. Replay de las cámaras

```bash
uv run python tools/simulate_cameras.py --hour 2026033010 --list-hours   # qué hay
uv run python tools/simulate_cameras.py --hour 2026033010               # servir
```

Imprime la línea de `RTSP_CAMERAS` y los comandos de captura. Las dos cámaras
reciben **la misma hora grabada**: si no, las entradas no tienen salidas que les
correspondan.

## 4. Captura

```bash
uv run python rtsp-capture/capture.py \
    --camera-id gate-1 --rtsp-url rtsp://127.0.0.1:8554/gate-1 \
    --interval 0.5 --gate entry
```

`--gate exit` en la cámara de salidas: solo una entrada abre sesión.

## 5. Verificar

```bash
docker exec parking-sim-pg psql -U parking -d parking -c \
  "select vehicle_type, color, count(*) from vehicles group by 1,2 order by 3 desc;"

curl "http://127.0.0.1:8001/api/v1/vehicles/search?from=2026-10-21T00:00:00Z&to=2026-10-22T23:59:00Z&color=plata"
curl "http://127.0.0.1:8001/api/v1/stats/colors?days=1"
```

Resultado de la primera corrida real: **26 vehículos** persistidos con tipos y
colores plausibles, y la búsqueda de `plata` devolvió primero los 3 aciertos
exactos y después los vecinos `blanco` y `gris` — que es exactamente lo que la
expansión por matriz de confusión tenía que hacer.

## Limitaciones conocidas

- **El reloj.** La app sella los eventos con la hora de pared, no con la del
  nombre del archivo. Una búsqueda por ventana durante un replay es relativa a
  "ahora"; para que signifique algo hay que offsetear el reloj del contenedor
  (`libfaketime`) o buscar por ventana relativa.
- **Warning de HEVC.** Al iniciar el stream aparecen frames sin referencia
  (`Could not find ref with POC`). `capture.py` reconecta ante **cualquier**
  lectura fallida, así que un solo hipo puede disparar un bucle de reconexión.
  Conviene contar fallos consecutivos antes de reconectar.
- **Localhost no es la red.** Para latencia, jitter y pérdida reales:
  `sudo tc qdisc add dev lo root netem delay 50ms loss 1%`.
- **El worker no cierra sesiones al salir.** Nadie matchea todavía, así que la
  ocupación solo crece. Es deliberado: inventar el match corrompería el número.

## Limpieza

```bash
# Ctrl+C al harness; después
docker rm -f parking-sim-pg && docker stop parking-redis
pkill -f backend.worker.celery_app
```

## Por qué el matcheo no usa similitud

Antes de implementar el matcheo se midió si el backbone del clasificador de tipo
sirve como *embedding* de identidad, que era la vía barata. Sobre 400 crops del
dataset congelado, 79.800 pares:

| | |
|---|---|
| pares del **mismo** vehículo | similitud media **0,860** |
| pares de vehículos **distintos** | p50 0,786 · p90 **0,973** · p99 0,991 |
| P(dos vehículos distintos > 0,95) | **19,9%** |
| P(dos vehículos distintos > 0,99) | 1,26% |

**Uno de cada cinco pares de vehículos sin relación se parece más que el par
promedio del mismo vehículo.** No existe umbral que separe: en 0,99 todavía
aceptás 1,26% de falsos positivos y rechazás vehículos reales (0,86). El backbone
está entrenado para silueta, no para identidad.

Por eso `backend/app/core/matching.py` matchea **solo por identificador exacto**
(la patente), con piso de confianza, y devuelve `None` si no hay prueba. Un exit
sin match deja la sesión abierta y publica `vehicle_exit_unmatched`: el operador
ve el estacionamiento llenarse, no vaciarse solo.

Hoy el engine **no corre OCR**, así que la patente siempre es `None` y ningún exit
cierra sesión. Está cableado igual: cuando exista un modelo de OCR, lo único que
cambia es de dónde sale la patente. Alternativas descartadas y por qué:

- **Similitud/reID con este backbone**: medido arriba, no sirve.
- **Modelo de ReID dedicado**: no hay datos pareados (mismo vehículo en las dos
  cámaras) para validarlo, así que no se puede afirmar que funcione.
- **OCR de patentes**: es la vía correcta y hay entrenador
  (`tools/train_ocr_crnn.py`) y generador sintético, pero **no hay checkpoint
  entrenado** y solo hay 285 patentes reales.

## Anomalía observada, sin atribuir

En algunas corridas el capturador iteró el bucle solo 2 veces en 20 s, cuando
debería muestrear 40. Aislado, `_has_motion` dispara correctamente contra el
stream en vivo (7 de 43 muestras) y el delta medido en vivo supera el umbral en
~20% de las muestras. La latencia por `read()` no se pudo medir porque el servidor
ya estaba caído.

Hipótesis a verificar en la máquina real: **HEVC con GOP largo y
`CAP_PROP_BUFFERSIZE = 1`** puede hacer que `read()` bloquee hasta el próximo frame
decodificable. Si se reproduce, probar sin fijar el buffer, o publicar con un GOP
más corto, o reencodear en vez de `-c copy`.
