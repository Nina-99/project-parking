# Vehicle attribute classifiers

Two classifiers run over the same input: a crop holding one vehicle, produced by
the COCO detector at the gate.

| Model | Answers | Used for |
|-------|---------|----------|
| **type** | automovil, camioneta, furgon, minibus, moto, vagoneta | fleet statistics, identifying a vehicle category in a search |
| **colour** | the *base paint* colour | fleet statistics, soft filter in incident search |

Neither model replaces the plate. They describe a vehicle when the plate is
unreadable, and they let an operator narrow a list when all they remember is
"a silver one, around 10:00".

## Data contract

```
car type/                     # flat pool written by label_type_gui.py
├── images/<frame>_b<idx>.jpg
├── labels/<frame>_b<idx>.txt         # type index   (data.yaml)
├── labels_color/<frame>_b<idx>.txt   # colour index (data_color.yaml)
├── labels.jsonl                      # audit: type, colour, confidences
└── reviewed.jsonl                    # per-frame resume state
```

`build_color_dataset.py` and `resplit_by_track.py` turn the pool into the
`train/valid/test` layout both trainers expect. Two rules matter:

- **Split by vehicle, never by crop.** Frames of the same car land in exactly one
  split, otherwise validation measures memorisation.
- **Filter at build time, not by deleting crops.** The pool stays intact and the
  build log records what was excluded.

## Training

```bash
# type
uv run python tools/train_type_classifier.py \
    --dataset "/mnt/nina/SmartParking/type dataset"

# colour (minibuses excluded, see below)
uv run python tools/build_color_dataset.py \
    --pool "/mnt/nina/SmartParking/car type" \
    --output "/mnt/nina/SmartParking/color dataset v2" \
    --exclude-types minibus --min-count 10
uv run python tools/train_color_classifier.py \
    --dataset "/mnt/nina/SmartParking/color dataset v2"

# one checkpoint, same split, comparable numbers
uv run python tools/eval_type_classifier.py \
    --dataset "/mnt/nina/SmartParking/color dataset v2" \
    --checkpoint /tmp/opencode/checkpoints/resnet18_color_best.pt
```

The two trainers share `_train_common.py` and **never share an augmentation
pipeline**. A colour model trained with saturation jitter learns to ignore
saturation, which is its label.

## What the reported metric means

`MIN_TEST_SUPPORT = 40`: a class with fewer test samples is trained but not
reported. Below that the 95% interval is wider than ±0.15 and the per-class F1
says nothing.

The metrics file carries three numbers, and they are not interchangeable:

| Field | Meaning | Use |
|-------|---------|-----|
| `test_macro_f1_evaluated` | mean F1 over reportable classes | the headline |
| `f1_weighted_real` | F1 weighted by real frequency | what deployment feels |
| `test_macro_f1` | legacy: mean over every present class | **do not cite** |

The legacy number is kept so older artefacts stay readable. It is inflated by
tiny classes that happen to be classified correctly; on the colour model it read
0.574 while the same weights score 0.743 over the reportable set.

## Decisions that are easy to undo by accident

| Decision | Why | Undo cost |
|----------|-----|-----------|
| Minibuses excluded from colour | body covered by livery whose dominant colour depends on viewing angle; the gate sees the rear, where most micros look white | rebuild the dataset |
| `gris` and `plata` kept as separate labels | both achromatic, and the split is only ~69% separable by a brightness threshold, but merging destroys the per-colour statistics and moves the confusion to `blanco` | relabel + retrain |
| `amarillo`, `marron`, `beige`, `naranja` not evaluated | 2-13 samples per class; two of them are mostly motorcycles | none, they stay in training |
| `gris+plata` reported as separate counts | the split is ±13% noisy, so publish both with that caveat, never as precise values | none |
| No `otros` label | amarillo (H=87°), naranja (H=32°), marron (72°), beige (85°) sit 30-55° apart; one class would be multi-modal | relabel + retrain |

## Serving

`GET /api/v1/vehicles/search` returns ranked candidates for a time window. Asking
for `plata` also returns `gris`, because the confusion matrix says the classifier
cannot separate them reliably; exact matches rank first.

`GET /api/v1/stats/colors` publishes six colours individually and aggregates the
rest into `minority`. Both endpoints share `core/colours.py`, so the vocabulary
cannot drift.

## Queueing a class the model cannot learn yet

`furgon` sits below the learning floor at 59 labelled tracks. Do not rank
candidates by the model's own prediction: it calls 2 of every 17 furgones
"furgon", so the queue would surface the ones it already gets right. The queue
uses the COCO detector class and the trajectory kind instead -- neither sees the
body type.

```bash
uv run python tools/queue_candidates.py \
    --dataset /mnt/nina/SmartParking --target furgon --kinds entry --limit 400 \
    --exclude-labelled "/mnt/nina/SmartParking/car type/labels.jsonl" \
    --output "/mnt/nina/SmartParking/car type/queue_furgon.jsonl"

uv run python tools/label_type_gui.py \
    --queue "/mnt/nina/SmartParking/car type/queue_furgon.jsonl" \
    --dataset /mnt/nina/SmartParking --output "/mnt/nina/SmartParking/car type" --resume
```

On the current manifest: 836 truck entries, 822 after one crop per track, 812
left to label. A geometric prior was measured and rejected -- furgon boxes are
*less* tall than car boxes, because the gate sees the van from behind.

## Producing the tables

```bash
uv run python tools/report_metrics.py \
    --metrics /tmp/opencode/checkpoints/*_metrics.json \
    --title "Resultados de clasificación" --out docs/results.md
```

Renders the stored JSON, never recomputes it: per-class P/R/F1 with the Wilson
interval and support, the reported/excluded verdict, the confusion matrix, and
one summary row per run.

## Measured limits (put these in the thesis)

- **72% of the fleet is achromatic** (blanco, plata, gris, negro). A colour filter
  can only remove 28% of candidates, whatever the model's accuracy.
- **1 vehicle every 10 minutes** at the gate (median 6/hour, peak 42/hour). In a
  10-minute window the operator sees 1-7 vehicles: the window does the work and
  colour barely matters. Colour earns its keep only over hours or a full day.
- **`gris`/`plata` is near the label's ceiling.** Median lightness differs by 13
  units while each class has a standard deviation of ~16, so signal < noise. Best
  achievable threshold: 69% accuracy.
- **Run-to-run variance is about 1 point of macro-F1.** Same pipeline, same
  split, same seed, two runs: 0.7719 vs 0.7620.
- **What the colour pipeline change was worth**, three arms on the v3 split
  (same seed, 40 epochs, evaluated classes only):

  | pipeline | macro-F1 |
  |---|---|
  | original: RRC 0.7-1.0, hflip, jitter b/c/s 0.3 | 0.8075 |
  | **original minus saturation jitter** | **0.8232** |
  | minus saturation + rotation ±8° + motion blur | 0.7778 |

  Removing saturation jitter is worth about +1.6 points, at the edge of the noise
  floor. Adding rotation and blur on top costs about 4.5. The split between the
  two pipelines was right; the extra augmentation was not.
- **The other gain** came from excluding livery-dominant minibuses: +2 to +3
  points.
- Reference numbers on v3, evaluated classes only: colour **0.8232**
  (weighted 0.8318); type macro-F1 pending.
