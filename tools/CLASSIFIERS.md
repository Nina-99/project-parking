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

## furgón: kept as a class, reported as insufficient

`furgon` stays in the label set. Furgones do enter the premises, in small numbers
(97 of 4.055 labelled crops, 2.4%), and that count is itself part of what the
statistics have to say. Dropping the class would lose the distinction.

What the thesis may claim is narrower than for the other classes: **five classes
are evaluated, `furgon` is trained but not evaluated** because its test support is
11, below `MIN_TEST_SUPPORT = 40`. The full confusion matrix stays in the annex.

Why it cannot be evaluated, measured rather than assumed:

| evidence | value |
|---|---|
| within-class coherence (similarity to own centroid) | 0.892 — lowest of all six (others 0.907–0.974) |
| linear separability from the rest (5-fold AUC) | 0.867 — compare moto > 0.97 |
| nearest foreign centroid, of 73 furgones | vagoneta 29, camioneta 22, automovil 16, minibus 5, moto 1 |
| prevalence in the fleet | 2.4%, and reaching 40 test samples needs ~270 tracks |

The coherence and neighbourhood figures were computed when the class held 73
crops; it has since grown to 97. The direction has not changed -- furgón remains
the least coherent class and its nearest neighbour remains vagoneta -- but the
numbers above are the 73-crop measurement and should be re-run before being
quoted as final.

**How the class was grown matters, and the automated queue lost.** Reviewing the
822 COCO-`truck` entry candidates on channel 6 yielded 3 furgones in 495 reviews
(0.6%). A hand-curated folder of 40 crops from `entries_fast` yielded 34 (85%).
Both are worth recording: the detector class was a poor proxy for this body type
on that camera, and the human looking at candidate crops beat it by two orders of
magnitude.

**Merging into `camioneta` was rejected on evidence.** The proposal assumed a
furgón is a nearby variant of a pickup; the neighbourhood says otherwise — its
most frequent neighbour is `vagoneta`, not `camioneta`. Merging would have created
a class that matches no region of the feature space and polluted camioneta on the
way.

**What a furgón count means.** Predicted counts undercount, because a model with
recall around 0.5 files half of them under another type. The correction factor
cannot be estimated from 17 test samples, so a report should say "at least N
detected", never "exactly N", and never derive a trend from it.

**Cleanup pending:** five crops have `minibus` as their nearest foreign centroid
and are worth a second look (sim 0.69–0.93). Relabel them through
`label_type_gui.py --review` if the rear windows are there; do not let a model
decide this.

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

**The measured yield says the automated queue was the wrong tool.** Reviewing 495
of those 822 candidates produced 3 furgones (0.6%). A hand-curated folder of 40
crops from `entries_fast` produced 34 (85%). The detector's `truck` class is a
poor proxy for a van on channel 6, and a human looking at candidate crops beat it
by two orders of magnitude.

Keep that in mind before queueing the next rare class: an independent signal is
better than the model's own prediction, but only if the signal actually tracks
the class. Measure the yield on a sample of a hundred before committing hours to
it.

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
