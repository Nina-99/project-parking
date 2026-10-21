# Runbook: reproducir todo desde cero

Todos los comandos, en orden, para dejar los modelos y la ablación listos. Cada
bloque indica qué produce y cuánto tarda.

Trabajar siempre desde `software/`.

---

## 0. Estado de partida (ya hecho)

| artefacto | dónde | estado |
|---|---|---|
| Pool congelado | `/mnt/nina/SmartParking/pool_20261021` | 4.055 crops + `SNAPSHOT.txt` |
| Dataset de tipo | `type dataset v2` | 4.055 crops, 5 de 6 clases reportables |
| Dataset de color | `color dataset v4` | 3.627 crops, 6 de 11 colores reportables |
| Patentes reales | `plates/` | 496 |
| Patentes sintéticas | `plates/**/synth/` | 10.000 |
| Ablación OCR | `ablation/` | preparada, seed 7, test de 73 reales |
| Corridas de clasificadores | `runs/` | 5 corridas + checkpoints |

---

## 1. Ablación OCR

Los dos brazos usan **el mismo test** (`ablation/test.txt`, 73 patentes reales).
Eso es lo que hace comparable la tabla de la tesis.

### 1.1 Regenerar la ablación (solo si cambió el material)

```bash
uv run python tools/prepare_ocr_ablation.py --dataset /mnt/nina/SmartParking
uv run python tools/generate_synth_plates.py \
    --dataset /mnt/nina/SmartParking --count 10000 --seed 7
```

Determinístico: re-ejecutar sobrescribe con salida idéntica. `protocol.json`
guarda videos, seed y tamaños para pegar en la tesis.

### 1.2 Brazo A — solo reales (~2 min)

```bash
uv run python tools/train_ocr_crnn.py \
    --dataset /mnt/nina/SmartParking \
    --stage stage2_real \
    --epochs-real 30 --lr-real 2e-4 \
    --save-dir /mnt/nina/SmartParking/runs/ocr_real_only
```

### 1.3 Brazo B — sintéticas + reales (~15 min)

```bash
uv run python tools/train_ocr_crnn.py \
    --dataset /mnt/nina/SmartParking \
    --stage full_pipeline \
    --epochs-synth 25 --epochs-real 30 \
    --lr-synth 3e-3 --lr-real 2e-4 \
    --save-dir /mnt/nina/SmartParking/runs/ocr_synth_real
```

> **Trampa**: el brazo A **debe** usar un `--save-dir` distinto y vacío. Si ese
> directorio ya tiene `crnn_stage1_synth.pt`, el script lo carga y el brazo A deja
> de ser "solo reales" sin avisar. Un directorio por brazo, siempre.

### 1.4 Resultados

Cada brazo deja `final_metrics.json` (char accuracy y exact match, crudo y con
heurísticas) en su `--save-dir`. La tabla de la tesis sale de comparar los dos
archivos, que comparten test set.

---

## 2. Clasificadores de tipo y color

### 2.1 Congelar el pool (si se etiquetó algo nuevo)

```bash
cp -a "/mnt/nina/SmartParking/car type" "/mnt/nina/SmartParking/pool_$(date +%Y%m%d)"
cd "/mnt/nina/SmartParking/pool_$(date +%Y%m%d)"
sha256sum labels.jsonl | cut -c1-16        # anotar en SNAPSHOT.txt junto a los conteos
```

**Nunca** entrenar desde `car type`: sigue recibiendo etiquetas, y el split —aunque
es estable por hash— dejaría de corresponder al dataset que se publicó.

### 2.2 Reconstruir los datasets

```bash
uv run python tools/build_color_dataset.py \
    --pool /mnt/nina/SmartParking/pool_20261021 \
    --output "/mnt/nina/SmartParking/type dataset v2" \
    --labels-dir labels --data-yaml data.yaml --min-count 10 --force

uv run python tools/build_color_dataset.py \
    --pool /mnt/nina/SmartParking/pool_20261021 \
    --output "/mnt/nina/SmartParking/color dataset v4" \
    --labels-dir labels_color --data-yaml data_color.yaml \
    --exclude-types minibus --min-count 10 --force
```

`--force` reemplaza; sin él el build se niega si el directorio no está vacío, que
es lo que evita mezclar splits viejos.

### 2.3 Composición y clases reportables

```bash
uv run python tools/report_dataset.py --dataset "/mnt/nina/SmartParking/type dataset v2" \
    --title "Tipo · v2" --out /mnt/nina/SmartParking/reports/type_v2.md
uv run python tools/report_dataset.py --dataset "/mnt/nina/SmartParking/color dataset v4" \
    --title "Color · v4" --out /mnt/nina/SmartParking/reports/color_v4.md
```

### 2.4 Entrenar

```bash
# tipo (5 clases evaluadas; furgon entrenada, no evaluada)
uv run python tools/train_type_classifier.py \
    --dataset "/mnt/nina/SmartParking/type dataset v2" \
    --name final_type_v2 --epochs 40 \
    --save-dir /mnt/nina/SmartParking/runs

# color (6 colores evaluados)
uv run python tools/train_color_classifier.py \
    --dataset "/mnt/nina/SmartParking/color dataset v4" \
    --name final_color_v4 --epochs 40 \
    --save-dir /mnt/nina/SmartParking/runs
```

~11 min por corrida. **Una sola corrida no es un resultado**: la varianza medida es
de 1 a 3,4 puntos de macro-F1. Para afirmar algo, correr dos o tres veces con
nombres distintos y reportar el rango.

### 2.5 Evaluar un checkpoint contra cualquier split

```bash
uv run python tools/eval_type_classifier.py \
    --dataset "/mnt/nina/SmartParking/color dataset v4" \
    --checkpoint /mnt/nina/SmartParking/checkpoints/resnet18_color_v3_best.pt \
    --split valid
```

Agregar `--types "<dataset>/types.jsonl"` para el desglose por tipo de vehículo
(diagnóstico: los grupos no alcanzan el soporte mínimo, no se reporta macro).

### 2.6 Tablas para la tesis

```bash
uv run python tools/report_metrics.py \
    --metrics /mnt/nina/SmartParking/runs/*_metrics.json \
    --title "Clasificadores" --out /mnt/nina/SmartParking/reports/clasificadores.md
```

---

## 3. Simulación end-to-end

Procedimiento completo en [`SIMULATION.md`](SIMULATION.md). Resumen:

```bash
# datastores (ojo: el 5432 puede estar ocupado por otra base)
docker compose up -d postgres redis        # o el postgres del simulador en 5433
uv run python -c "..."                     # crear tablas (ver SIMULATION.md)

# worker y API en el host, con los overrides de puertos publicados
export CELERY_BROKER_URL=redis://localhost:6379/0
export DATABASE_URL_SYNC=postgresql+psycopg://parking:parking@localhost:5433/parking
uv run celery -A backend.worker.celery_app worker --loglevel=info &
DATABASE_URL=postgresql+asyncpg://parking:parking@localhost:5433/parking \
    uv run uvicorn backend.app.main:app --port 8001 &

# las dos cámaras con la MISMA hora grabada
uv run python tools/simulate_cameras.py --hour 2026033010 &

# captura
uv run python rtsp-capture/capture.py --camera-id gate-1 \
    --rtsp-url rtsp://127.0.0.1:8554/gate-1 --gate entry
```

---

## 4. Orden recomendado

1. **Ablación OCR** (§1): es la que destraba el matcheo entrada/salida. ~17 min.
2. **Clasificadores** (§2): ya hay resultados; re-correr solo si cambia el dataset.
3. **Simulación** (§3): para verificar la cadena completa con datos reales.

## 5. Lo que todavía no está resuelto

- **El matcheo no cierra sesiones**: no hay checkpoint de OCR hasta que corra §1, y
  el engine todavía no corre OCR. Ver `SIMULATION.md`.
- **Un stall de `read()` sin atribuir** en el capturador por CLI. Próximo paso:
  loguear el retorno de `read()` y su tiempo dentro de `start()`.
- **La selección de época** usa macro-F1 sobre todas las clases de validación,
  mientras el reporte usa solo las que superan el soporte. Documentado, no corregido.
- **No hay esquema para persistir salidas sin match**: hoy se publican por Redis y
  quedan en el log. Reportarlas necesita una columna.
