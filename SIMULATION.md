<!--
SPDX-FileCopyrightText: 2026 Nina
SPDX-License-Identifier: LicenseRef-Proprietary
-->

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

## 1. Levantar la cadena

```bash
cd software
./scripts/dev.sh up        # postgres 5433, redis 6379, API 8000, worker, frontend 5173
./scripts/dev.sh status    # confirma los cinco procesos y la salud
./scripts/dev.sh down      # conserva la base
```

`dev.sh up` aplica las migraciones de Alembic antes de arrancar la API; ni la API
ni el worker crean el esquema por su cuenta.

## 2. Reproducir una hora grabada

```bash
uv run python tools/simulate_cameras.py --hour 2026041511 --list-hours   # qué horas hay
RTSP_CAMERAS="gate-1=rtsp://127.0.0.1:8554/gate-1,gate-2=rtsp://127.0.0.1:8554/gate-2" \
timeout 3600 uv run python tools/simulate_cameras.py \
    --hour 2026041511 --with-capture --transcode \
    --start-file /tmp/replay.start
```

Antes de servir nada, el arnés comprueba el broker y la base: si alguno no
responde, se detiene en el primer segundo en vez de transmitir una hora hacia
ninguna parte. Ese chequeo existe porque ya pasó lo contrario.

- `--with-capture` arranca `capture.py` para las dos cámaras. `gate-1` abre sesión
  y `gate-2` la cierra; se puede cambiar con `--gate gate-2=entry`.
- `--transcode` recodifica a H.264. Copiar el HEVC del grabador (`-c copy`) pierde
  fotogramas de referencia y el detector recibe imágenes dañadas.
- `--start-file` guarda el instante en que arrancó la reproducción. Es el
  `--replay-start` que necesita cualquier comparación contra una verdad de
  referencia.

Las dos cámaras reciben **la misma hora grabada**: si no, las entradas no tienen
salidas que les correspondan.

Para servir sin capturar —y correr `capture.py` a mano en otra terminal— omití
`--with-capture`; el arnés imprime entonces el comando exacto de cada cámara.

Al cortar, esperá unos segundos antes de leer la base: la cola del worker puede
tener eventos pendientes. Un `timeout` en seco se come los últimos.

## 3. Verificar

La API de lectura no es pública: primero hay que abrir sesión. El token viaja en
una cookie httpOnly.

```bash
curl -c /tmp/parking.cookies -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"operator","password":"operator"}'

# lo que el panel muestra en vivo
curl -b /tmp/parking.cookies http://127.0.0.1:8000/api/v1/stats/occupancy
curl -b /tmp/parking.cookies http://127.0.0.1:8000/api/v1/stats/active

# el evento que decidió el sistema, con su dirección
docker exec parking-dev-pg psql -U parking -d parking -c \
  "select camera_id, to_char(frame_timestamp,'HH24:MI:SS'), direction
     from detection_events order by frame_timestamp desc limit 10;"
```

Un evento por vehículo que **cruzó** el acceso, con `direction` igual a `entry` o
`exit`. Si la tabla se llena con vehículos que solo pasaron por la calle, la
geometría de esa cámara está admitiendo calle dentro del polígono: revisala con
`tools/calibrate_zones.py`.

En el navegador, `http://localhost:5173` con `operator` / `operator` (valores por
defecto de `.env`): el panel se actualiza por WebSocket sin recargar.

## Limitaciones conocidas

- **El reloj.** La app sella los eventos con la hora de pared, no con la del
  nombre del archivo. Por eso el arnés escribe `--start-file`: ese instante es el
  origen contra el que se compara cualquier verdad de referencia tomada sobre el
  video.
- **HEVC sin transcodificar.** Con `-c copy` el decodificador pierde fotogramas de
  referencia (`Could not find ref with POC`) y el detector ve imágenes dañadas.
  `--transcode` lo resuelve a cambio de CPU; sin él, cualquier medición de ese
  tramo mide el ruido del transporte, no el sistema.
- **El cierre por salida depende de la patente.** Una salida sin lectura
  confiable no cierra sesión y queda registrada como `vehicle_exit_unmatched`.
  Es deliberado: un match inventado vaciaría el lote en silencio.
- **Localhost no es la red.** Para latencia, jitter y pérdida reales:
  `sudo tc qdisc add dev lo root netem delay 50ms loss 1%`.

## Limpieza

```bash
# Ctrl+C al harness; después
./scripts/dev.sh down          # API, worker, frontend y contenedores
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
- **OCR de patentes**: es la vía correcta. Hay entrenador
  (`tools/train_ocr_crnn.py`), generador sintético y un checkpoint servido
  (`runs/ocr_v3_plus/crnn_stage2_best.pt`, $0{,}882$ de exactitud por carácter y
  $0{,}535$ de coincidencia exacta en su conjunto de prueba).

## El broker caído congelaba la cámara (resuelto)

Síntoma: en algunas corridas el capturador iteraba 2 veces en 20 s en vez de ~40.
La causa no era el video ni el decodificador — se descartaron con datos:

- Lecturas uniformes: media 63 ms, máximo 68 ms (= los 15 fps del stream), con y
  sin `CAP_PROP_BUFFERSIZE=1`.
- GOP de **1 segundo**, no largo.
- El gate de movimiento dispara bien aislado contra el stream vivo (7 de 43
  muestras) y el delta en vivo supera el umbral en ~20% de las muestras.

La causa era **`process_frame.delay()` con el broker caído: bloqueó 107 segundos**
antes de rendirse. Configurar `task_publish_retry = False` y timeouts de socket
**no alcanzó** — siguió bloqueando más de 55 s.

Arreglo (`rtsp-capture/capture.py`): el publish corre en su propio hilo y el bucle
nunca lo espera. **Un solo vuelo a la vez**: mientras el broker está enfermo el
hilo queda pegado, y encolar uno por frame crearía cientos. Los frames son
transitorios, así que se descartan (se cuentan en `dropped_frames`).

Verificado con el broker caído: **35 iteraciones en 20 s** (antes 2), 7 con
movimiento, 6 frames descartados. Y al salir con un hilo bloqueado, Python abortaba
con `FATAL: exception not rethrown` y core dump; el handler de señal ahora hace
`os._exit(0)` tras vaciar los buffers, con salida limpia.

Lección para el resto del sistema: una caída de Redis no puede costar dos minutos
de footage del portón. Cualquier productor sobre el camino crítico necesita lo
mismo.
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
