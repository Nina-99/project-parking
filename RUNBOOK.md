<!--
SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
SPDX-License-Identifier: LicenseRef-Proprietary
-->

# Runbook: reproducir todo desde cero

Todos los comandos, en orden, para dejar los datasets, los modelos y la ablación
listos. Cada bloque indica qué produce y cuánto tarda.

Trabajar siempre desde `software/`.

---

## 0. Estado de partida

Todo vive en el NVMe. El HDD externo (`/mnt/nina`) **ya no es fuente de verdad**:
empezó a devolver `EIO` en ~1.3% de los crops del pool y truncó archivos de
patentes sintéticas durante la transferencia. Los 33 se regeneraron de forma
determinística con `tools/repair_synth_plates.py`, verificando la etiqueta contra
el original, así que el conjunto conserva 360.000 válidos
de 360.000 muestras; las ocho restantes se descartaron. El material se rescató a
`/home/nina/FNI2025/ProjectParking`.

| artefacto | dónde | estado |
|---|---|---|
| Pool congelado | `datasets/pool_20261021` | 4.006 crops, `nc=5` (furgón fusionado) |
| Dataset de tipo | `datasets/type dataset v3` | 2.783 / 590 / 633, `nc=5` |
| Dataset de color | `datasets/color dataset v5` | 2.482 / 539 / 551, `nc=11` |
| Patentes reales | `plates/` | 496 (395 train / 101 test) |
| Patentes sintéticas | `plates/images/synth/` | 360.000 |
| Ablación OCR | `ablation/` | seed 7, `protocol.json` |
| Clasificadores | `datasets/runs/` | checkpoints + `*_metrics.json` |
| OCR | `runs/ocr_v2/` | stage 1 + stage 2 + `final_metrics.json` |

`train_all.sh` asume `SMARTPARKING=/home/nina/FNI2025/ProjectParking/datasets`
y `PLATES_ROOT=/home/nina/FNI2025/ProjectParking`. Sobrescribilos por entorno si
movés el material.

---

## 1. Los datasets

```bash
cd /home/nina/FNI2025/ProjectParking/software

# reconstruye tipo (nc=5) y color (nc=11) desde el pool
FORCE=1 ./train_all.sh datasets
```

`FORCE=1` es obligatorio si el directorio existe a medias: el script hace `skip`
por existencia, no por integridad. Un dataset truncado por una copia fallida se
saltea sin avisar.

Verificación antes de entrenar:

```bash
for d in "type dataset v3" "color dataset v5"; do
  echo "== $d"
  grep '^nc:' "/home/nina/FNI2025/ProjectParking/datasets/$d/data.yaml"
  for s in train valid test; do
    echo "  $s: $(ls "/home/nina/FNI2025/ProjectParking/datasets/$d/$s/images" | wc -l)"
  done
done
```

Esperado: tipo `nc=5` con 2783/590/633, color `nc=11` con 2482/539/551.

**Nunca** entrenar desde `car type`: sigue recibiendo etiquetas y el split dejaría
de corresponder al dataset publicado.

---

## 2. Clasificadores de tipo y color

```bash
./train_all.sh type      # ~6 min
./train_all.sh color     # ~6 min
```

Deja `resnet18_final_type_v3_best.pt` y `resnet18_final_color_v5_best.pt` en
`datasets/runs/`, cada uno con su `*_metrics.json` al lado.

**El `_metrics.json` es parte del checkpoint.** El engine lee de ahí el orden de
las clases (`class_names()` en `backend/worker/engine.py`); un `.pt` sin su JSON
no se puede servir, y el motor se niega a arrancar.

### 2.1 Augmentation

Sale de `build_transforms()` en `tools/_train_common.py` y difiere por tarea:

- **tipo**: `RandomResizedCrop(224, 0.7-1.0)` + flip + `ColorJitter(0.3, 0.3, 0.3)`
- **color**: `RandomResizedCrop(224, 0.7-1.0)` + flip + `ColorJitter(0.3, 0.3)`
  — **sin saturación ni hue**, porque el tono es la etiqueta

Validación y test: solo `Resize(224)` + normalización. Rotación y motion blur se
midieron y se descartaron: costaban más de lo que aportaban.

### 2.2 Selección del checkpoint

La época se elige con `macro-F1` sobre las clases de validación con **al menos 20
muestras** (`--select-min-support`, default 20). Antes se usaba el macro sobre
todas las presentes, y en color eso lo decidía una clase con 3 imágenes: un solo
acierto movía el macro 1/11 = 0.09. Con el criterio viejo el "mejor epoch" era
ruido; corregirlo subió color de 0.7937 a 0.8124 de macro-F1.

Para reproducir el comportamiento viejo: `--select-min-support 0`.

### 2.3 Evaluar y reportar

```bash
uv run python tools/eval_type_classifier.py \
    --dataset "/home/nina/FNI2025/ProjectParking/datasets/color dataset v5" \
    --checkpoint /home/nina/FNI2025/ProjectParking/datasets/runs/resnet18_final_color_v5_best.pt \
    --split test

uv run python tools/report_dataset.py \
    --dataset "/home/nina/FNI2025/ProjectParking/datasets/type dataset v3" \
    --title "Tipo · v3" --out ../reports/type_v3.md

uv run python tools/report_metrics.py \
    --metrics /home/nina/FNI2025/ProjectParking/datasets/runs/*_metrics.json \
    --title "Clasificadores" --out ../reports/clasificadores.md
```

**Una corrida no es un resultado**: la varianza medida es de 1 a 3,4 puntos de
macro-F1. Para afirmar algo, correr dos o tres veces con nombres distintos y
reportar el rango.

---

## 3. OCR

### 3.1 Ablación (solo si cambió el material)

```bash
uv run python tools/prepare_ocr_ablation.py \
    --dataset /home/nina/FNI2025/ProjectParking
uv run python tools/generate_synth_plates.py \
    --dataset /home/nina/FNI2025/ProjectParking --count 360000 --seed 7
```

Determinístico. `ablation/protocol.json` guarda seed y tamaños para la tesis.

### 3.2 Entrenar

```bash
./train_all.sh ocr        # ~80 min stage 1 + ~10 min stage 2
```

O a mano, con las mismas banderas:

```bash
D=/home/nina/FNI2025/ProjectParking/runs/ocr_v2
uv run python -u tools/train_ocr_crnn.py \
    --dataset /home/nina/FNI2025/ProjectParking \
    --stage stage1_synth --epochs-synth 8 \
    --lr-schedule fixed --save-dir $D

uv run python -u tools/train_ocr_crnn.py \
    --dataset /home/nina/FNI2025/ProjectParking \
    --stage stage2_real --epochs-real 500 \
    --lr-real 1e-3 --lr-schedule fixed --synth-mix-ratio 0 \
    --save-dir $D
```

### 3.3 Por qué esas banderas

Cada una costó una corrida fallida:

- **`--lr-schedule fixed`**: con `plateau`, stage 2 baja el LR a 6e-7 en ~60
  epochs porque el loss es ruidoso. El modelo se congela sin avisar.
- **`--lr-real 1e-3`**: con 2e-4 nunca sale de la cuenca sintética (14% tras 100
  epochs).
- **`--epochs-real 500`**: 395 imágenes ÷ 32 = 12 batches por época. 30 epochs son
  360 pasos de gradiente para 21.7M de parámetros; a 500 epochs la curva todavía
  subía (50.7% → 67.6% entre el 200 y el 500).
- **`--epochs-synth 8`**: la validación sintética mesetea en el epoch 5-6
  (94.7% char / 81% EM) y después solo memoriza el renderer.
- **`--synth-mix-ratio 0`**: el replay sintético no aportó con este dataset.

### 3.4 Dos trampas

1. **CLAHE.** `PlateOcrDataset` normaliza contraste con CLAHE (canal L de LAB,
   clip 2.0, tiles 4×4). Los crops reales son capturas nocturnas: brillo mediano
   76.6 contra 106.4 del sintético. Sin CLAHE el modelo se queda en **14%** de
   char accuracy; con CLAHE llega a **67.6%**. El adapter de producción
   (`backend/adapters/outbound/ml/plate_crnn.py`) aplica el mismo paso: si alguien
   lo saca de ahí, los números de la tesis dejan de describir el sistema.
2. **`crnn_stage1_synth.pt` solo se escribe al terminar el loop.** Si cortás con
   Ctrl-C en el epoch 4, ese archivo no existe y stage 2 arranca en frío **sin
   avisar** (el `if checkpoint_synth.exists()` no imprime nada). Para rescatarlo:

   ```bash
   uv run python - <<'EOF'
   import torch
   D = "/home/nina/FNI2025/ProjectParking/runs/ocr_v2"
   s = torch.load(f"{D}/resume_stage1.pt", map_location="cpu")
   torch.save(s["model"], f"{D}/crnn_stage1_synth.pt")
   print("epoch", s["epoch"])
   EOF
   ```

   Y verificá en el log: `[ocr-train] Loaded Stage 1 weights from ...`.

### 3.5 Detector de placas (localizador)

El CRNN lee recortes ajustados de placa; el worker le entrega recortes de
vehículo. Sin este paso intermedio el OCR recibe un auto entero aplastado a
192x48 y no lee nada.

```bash
# 1. PaddleOCR propone una caja por recorte de vehículo (~6 min, 4.006 crops)
uv run python tools/pseudo_label_plates.py \
    --images /home/nina/FNI2025/ProjectParking/datasets/pool_20261021/images \
    --output ../plates_det/proposed.jsonl

# 2. Revisión humana: aceptar, corregir, descartar (~1 h)
uv run python tools/label_plate_box_gui.py \
    --proposed ../plates_det/proposed.jsonl \
    --output ../plates_det/verified.jsonl \
    --backup-dir ~/backup_plates_det

# 3. Exportar y entrenar
./train_all.sh platedet
```

`verified.jsonl` **no se puede regenerar**: son horas de revisión humana. Las
propuestas de Paddle sí (misma semilla, mismo orden). Por eso el `--backup-dir`.

Controles de la GUI: `drag` redibuja, `flechas` mueven 1 px, `+`/`-` agrandan o
achican, `k` borra y empezás de cero, `u` recupera la propuesta, `Enter` acepta,
`n` no hay placa, `s` saltea.

El split se hace por **bloque de grabación**, no por trayectoria: dos vehículos
de la misma hora no deben quedar a lados opuestos, o el test mide condiciones ya
vistas.

---

## 4. Resultados medidos

| modelo | métrica | valor |
|---|---|---|
| tipo | macro-F1 (test, n=633) | **0.8887** |
| tipo | accuracy | 0.8578 |
| color | macro-F1 (evaluadas, n≥40) | **0.8124** |
| color | accuracy | 0.8022 |
| detector de placas | recall (n=178) | **0.971** |
| detector de placas | mAP@50 | 0.990 |
| OCR | char accuracy (crops, n=101) | **0.831** |
| OCR | exact match (crops) | 0.426 |
| OCR | char accuracy end-to-end (n=88) | **0.782** |
| PaddleOCR | char accuracy end-to-end (n=88) | 0.653 |

Del OCR, por resolución del crop: `<20px` 60.5% (n=17), `20-30px` 82.4%,
`30-40px` 80.3%, `>=40px` 84.7%. En crops `>=20px` el modelo llega a 82.6%.

Dos advertencias para la tesis:

- El OCR se evalúa contra `ablation/test.txt` en **cada** epoch, así que "el mejor
  epoch" sería selectivo. El número reportado es el **epoch 500**, fijado de
  antemano; no se elige el máximo de la curva.
- El post-proceso heurístico boliviano **empeora**: 78.6% contra 78.9% crudo.
  Reescribe letras que son válidas en el bloque numérico (`216OZAC`). En
  producción va apagado (`apply_heuristics=False`).

---

## 5. Desplegar los modelos en el backend

Las rutas viven en `backend/app/core/config.py`:

```
TYPE_MODEL_PATH      datasets/runs/resnet18_final_type_v3_best.pt
COLOR_MODEL_PATH     datasets/runs/resnet18_final_color_v5_best.pt
PLATE_DETECTOR_PATH  plates_det/runs/plate_det/weights/best.pt
OCR_CRNN_CHECKPOINT  runs/ocr_s2_seed7/crnn_stage2_real.pt
```

El OCR viene **encendido** con el lector `crnn`. La confianza mínima es 0.60 y es
deliberadamente permisiva: la medición sobre 88 crops muestra que la confianza del
CRNN separa poco (mediana 0,981 en aciertos contra 0,835 en fallos), y subirla a
0,9 solo reduce la cobertura a 44% para llevar el acierto por lectura de 26% a
51%. La protección real no es el umbral sino el matcheo: una salida solo cierra
una sesión si **ambas** lecturas coinciden exactamente y superan el umbral.

Con 26% de coincidencia exacta, el sistema cierra pocas sesiones y deja el resto
al operador. Eso es intencional: un cierre incorrecto es invisible y descuadra el
conteo; una sesión abierta de más es visible y corregible.

---

## 6. Simulación end-to-end

Procedimiento completo en [`SIMULATION.md`](SIMULATION.md). Resumen:

```bash
docker compose up -d postgres redis        # ojo: el 5432 puede estar ocupado
uv run python -c "..."                     # crear tablas (ver SIMULATION.md)

export CELERY_BROKER_URL=redis://localhost:6379/0
export DATABASE_URL_SYNC=postgresql+psycopg://parking:parking@localhost:5433/parking
uv run celery -A backend.worker.celery_app worker --loglevel=info &
DATABASE_URL=postgresql+asyncpg://parking:parking@localhost:5433/parking \
    uv run uvicorn backend.app.main:app --port 8001 &

uv run python tools/simulate_cameras.py --hour 2026033010 &
uv run python rtsp-capture/capture.py --camera-id gate-1 \
    --rtsp-url rtsp://127.0.0.1:8554/gate-1 --gate entry
```

---

## 7. Orden recomendado

1. **Datasets** (§1): primero, siempre. ~2 min.
2. **Tipo y color** (§2): ~12 min las dos corridas.
3. **OCR** (§3): ~90 min. Independiente de las anteriores.
4. **Simulación** (§6): para verificar la cadena completa.

---

## 8. Lo que todavía no está resuelto

- **El OCR no auto-cierra sesiones.** 17.8% de exact match. Queda como sugerencia
  al operador hasta que haya más placas reales.
- **El techo del OCR son los datos.** 496 placas reales en total, con 17 crops de
  menos de 20 px de alto sobre 101 en test. El 90% de char accuracy que pide la
  tesis no es alcanzable con este material; el aporte medible es el análisis por
  resolución y la ablación de CLAHE.
- **Un stall de `read()` sin atribuir** en el capturador por CLI. Próximo paso:
  loguear el retorno de `read()` y su tiempo dentro de `start()`.
- **No hay esquema para persistir salidas sin match**: hoy se publican por Redis y
  quedan en el log. Reportarlas necesita una columna.
- **El disco externo está en falla.** Si algo quedó solo ahí, copiarlo con
  verificación (`rsync -c`) antes de desconectarlo definitivamente.
