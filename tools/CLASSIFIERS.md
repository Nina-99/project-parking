<!--
SPDX-FileCopyrightText: 2026 Nina
SPDX-License-Identifier: LicenseRef-Proprietary
-->

# Vehicle attribute classifiers

Two classifiers run over the same input: a crop holding one vehicle, produced by
the COCO detector at the gate.

| Model | Answers | Used for |
|-------|---------|----------|
| **type** | automovil, camioneta, minibus, moto, vagoneta | fleet statistics, identifying a vehicle category in a search |
| **colour** | the *base paint* colour | fleet statistics, soft filter in incident search |

Neither model replaces the plate. They describe a vehicle when the plate is
unreadable, and they let an operator narrow a list when all they remember is
"a silver one, around 10:00".

## Data contract

```
pool_20261021/                # FROZEN snapshot: the reference for every dataset
├── images/<frame>_b<idx>.jpg
├── labels/<frame>_b<idx>.txt         # type index   (data.yaml)
├── labels_color/<frame>_b<idx>.txt   # colour index (data_color.yaml)
├── labels.jsonl                      # audit: type, colour, confidences
├── reviewed.jsonl                    # per-frame resume state
└── SNAPSHOT.txt                      # counts, labels.jsonl hash, software commit
```

Build every dataset from the snapshot, never from the live `car type` folder,
which keeps receiving labels. Rebuilding is refused over a non-empty output
unless `--force`, and the split is a hash of the frame name, so a new label can
never move an existing crop between splits.

```bash
# type
uv run python tools/build_color_dataset.py \
    --pool /mnt/nina/SmartParking/pool_20261021 \
    --output "/mnt/nina/SmartParking/type dataset v2" \
    --labels-dir labels --data-yaml data.yaml --min-count 10

# colour (minibuses excluded, see below)
uv run python tools/build_color_dataset.py \
    --pool /mnt/nina/SmartParking/pool_20261021 \
    --output "/mnt/nina/SmartParking/color dataset v4" \
    --labels-dir labels_color --data-yaml data_color.yaml \
    --exclude-types minibus --min-count 10
```

The same tool builds both: `--labels-dir` and `--data-yaml` pick which label set
and class names to read. It also writes a `types.jsonl` sidecar so colour metrics
can be broken down by vehicle type.

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

## furgón: merged into vagoneta (2026-10-23)

`furgon` existed as a class until 2026-10-23. It was removed and its crops were
reassigned by hand. The vocabulary is now five classes.

**Why the decision flipped.** The earlier note here kept the class on the argument
that "the distinction is part of what the statistics have to say". Three
measurements defeated that:

| evidence | value |
|---|---|
| test macro F1 of `furgon` (2026-10-18 checkpoint) | **0.16** — lowest by 3.5x (next: camioneta 0.57) |
| test support | 17 crops, and 2 of them predicted correctly |
| share of the dataset | 2.4%, and reaching `MIN_TEST_SUPPORT = 40` needs ~270 tracks |
| **agreement with a human re-audit** | **49 of 98 crops were not furgones at all** |

That last row is the decisive one. A class that a careful human re-labels
correctly only half the time is not a labelling target, it is noise: 23 crops
were pickups (`camioneta`), 20 were panel vans or SUVs (`vagoneta`), 6 were
passenger vans (`minibus`) and 49 were buses, box trucks and pickups that never
belonged in the bucket. No model can be more consistent than its labels.

The feature itself is unobservable at the gate: the presence of rear side windows
is lost at night, when the vehicle enters rear-first, or when the tailgate
occludes it.

**Where the crops went** — human judgement, one pass over all 98, no model
involvement:

| destination | n |
|---|---|
| camioneta | 23 |
| vagoneta | 20 |
| minibus | 6 |
| discarded | 49 |

Note this partly contradicts the earlier "merging into camioneta was rejected"
argument, which relied on nearest-centroid counts (vagoneta 29, camioneta 22):
when a human looked at the actual bodies, pickups were in fact the largest
group. Centroid neighbourhoods on a contaminated class are not evidence about
the true class boundary.

**What the thesis may claim now.** Five classes, all with test support above
`MIN_TEST_SUPPORT`, so all five are evaluated. The macro F1 rises by
construction (a 0.16 class left the average); do not present that as an
improvement in detection quality, report the per-class table instead.

`furgon` counts from any model trained before 2026-10-23 are not comparable with
the current taxonomy.

## Serving

`GET /api/v1/vehicles/search` returns ranked candidates for a time window. Asking
for `plata` also returns `gris`, because the confusion matrix says the classifier
cannot separate them reliably; exact matches rank first.

`GET /api/v1/stats/colors` publishes six colours individually and aggregates the
rest into `minority`. Both endpoints share `core/colours.py`, so the vocabulary
cannot drift.

## Queueing a class the model cannot learn yet — obsolete since 2026-10-23

`queue_candidates.py` was built for `furgon`, its only registered target, and that
class no longer exists. The tool still runs and its tests still pass, but there is
no target to queue. Treat it as a record of a rejected approach, not a live path.

The method finding survives and is worth keeping in the thesis: do not rank
candidates for a rare class by the model's own prediction -- it calls 2 of every
17 furgones "furgon", so the queue surfaces the ones it already gets right. Use an
independent signal (COCO detector class, trajectory kind) instead.

**But measure the yield first.** Reviewing 495 of 822 `truck`-class candidates on
channel 6 produced 3 furgones (0.6%). A hand-curated folder of 40 crops from
`entries_fast` produced 34 (85%). The detector's `truck` class was a poor proxy
for a van on channel 6, and a human looking at candidate crops beat it by two
orders of magnitude. A geometric prior was also measured and rejected: furgon
boxes are *less* tall than car boxes, because the gate sees the van from behind.

The general lesson, now with a second data point: when a class needs 40 test
samples and has 17, the bottleneck is not the sampling tool, it is whether the
class is recognisable at all. Run the 98-crop audit before building the queue —
the audit is what killed the class, and it cost one afternoon.

## Producing the tables

Two generators, both rendering stored artefacts rather than recomputing them:

```bash
# composition: what the trainer will see, and which classes clear the gate
uv run python tools/report_dataset.py \
    --dataset "/mnt/nina/SmartParking/type dataset v2" \
    --out "/mnt/nina/SmartParking/reports/type_v2.md"

# metrics: per-class P/R/F1 with interval and support, plus the confusion matrix
uv run python tools/report_metrics.py \
    --metrics /mnt/nina/SmartParking/runs/*_metrics.json \
    --title "Resultados de clasificación" --out "/mnt/nina/SmartParking/reports/todo.md"
```

## Measured limits (put these in the thesis)

- **72% of the fleet is achromatic** (blanco, plata, gris, negro). A colour filter
  can only remove 28% of candidates, whatever the model's accuracy.
- **1 vehicle every 10 minutes** at the gate (median 6/hour, peak 42/hour). In a
  10-minute window the operator sees 1-7 vehicles: the window does the work and
  colour barely matters. Colour earns its keep only over hours or a full day.
- **`gris`/`plata` is near the label's ceiling.** Median lightness differs by 13
  units while each class has a standard deviation of ~16, so signal < noise. Best
  achievable threshold: 69% accuracy.
- **Run-to-run variance is larger than one point, and single runs prove nothing.**
  Two runs of the *same* colour configuration landed at 0.8232 (v3 dataset) and
  0.7897 (v4 dataset), a 3.4-point spread. The datasets are not the explanation:
  the v3 checkpoint scores 0.8197 on the v4 test split, versus 0.8232 on its own.
  So the gap is the training draw, not the data.
- **What the colour pipeline comparison therefore supports.** Three arms were
  measured on v3 (original 0.8075, minus saturation 0.8232, minus saturation plus
  rotation and blur 0.7778). With the spread above, the first two are
  indistinguishable and the third is clearly worse. The claim that stands is the
  negative one -- **rotation and blur cost performance** -- not that removing
  saturation jitter is worth +1.6 points. That claim needs replicates.
- **The type comparison does stand**, because two independent splits agree:
  original valid 0.8720 / test 0.8801 against augmented valid 0.8550 / test
  0.8707.
- **The only other gain that materialised** came from excluding livery-dominant
  minibuses: +2 to +3 points.
- Reference numbers, evaluated classes only, dataset frozen at `pool_20261021`,
  two runs per configuration:

  | model | runs (macro-F1 evaluadas) | range |
  |---|---|---|
  | type | 0.8801 · 0.8893 | **0.88–0.89** |
  | colour | 0.8232 (v3 data) · 0.7897 · 0.7781 (v4 data) | **0.78–0.82** |

- **Model selection uses a different metric than the report.** The training loop
  picks the best epoch by macro-F1 over *every* class present in validation, while
  the reported number uses only the classes that clear the support gate. A
  checkpoint can therefore be selected for doing well on classes that will not be
  reported. Fixing it means changing `evaluate()` and re-running every experiment,
  so it is recorded as a known limitation rather than changed at the close. Watch
  the tell-tale: the two type runs rank *opposite* on validation (0.8720 vs
  0.8584) and on test (0.8801 vs 0.8893), which also means selection by
  validation did not order them the way the test set does.
