<!--
SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
SPDX-License-Identifier: LicenseRef-Proprietary
-->

# Plate dataset tools

Extracts a license-plate dataset from recorded parking-entrance videos. The
extractor separates vehicles that **enter the premises** from vehicles that
only **pass along the street** by reasoning over each tracked vehicle's
trajectory, not over a single positional rectangle.

## Quick path

1. Calibrate the zones live with the interactive calibrator (run this first):
   ```bash
   cd software
   uv run python tools/calibrate_zones.py \
       --video /mnt/PARKING/ver/C.OCCIDENTE_ch16_main_20260415190012_20260415200012.mp4 \
       --camera channel16
   ```
   The video plays with tracking on: green curves are vehicles entering the
   premises, red curves are street traffic. Adjust the polygons with the mouse
   and press `s` to save them into `software/zones_config.json`.

2. Run the extractor:
   ```bash
   uv run python tools/extract_dataset.py --input /mnt/PARKING/ver/ \
       --output /mnt/Smart_Parking/dataset/
   ```
3. Verify `manifest.jsonl` grew and check the `plates/labels/`, `attributes/`
   and `review/` folders.
4. Label the vehicle attributes (body type and colour) by hand:
   ```bash
   uv run python tools/label_attributes.py --dataset /mnt/Smart_Parking/dataset/
   ```
   Two keystrokes per crop (type, then colour). Resumable: rerun it and it only
   asks for what is still pending. Use `--stats` to see the queue without
   opening a window.
5. Generate synthetic plates for OCR training (optional, train-only):
   ```bash
   uv run python tools/generate_synth_plates.py --dataset /home/nina/Smart_Parking_dataset --count 2000
   ```
   Renders the Bolivian national template and degrades it through the physical
   pipeline (perspective, light, sensor) down to the real size distribution.
   Rows land in `plates/{images,labels}/synth/` with `synth: true` in the
   manifest. Train on real+synthetic, evaluate on real only.

For a headless/batch alternative to step 1 (renders trajectories to a PNG),
use `tools/view_trajectories.py`.

## Details

| Topic | Decision |
|-------|----------|
| Camera selection | Inferred from the filename (`ch16` → `channel16`, `ch6` → `channel6`); override with `--camera`. |
| Zones | Polygons in normalized `[0, 1]` coordinates, edited in `zones_config.json`. |
| Entry rule | A track is an entry when its ground-contact point (bottom-centre of the bbox) reaches `inside_zone`; otherwise it passes. |
| Two-pass scan | Pass 1 samples sparsely and keeps only activity windows. Pass 2 decodes those windows densely — trackers need frame-to-frame continuity. |
| Plate pre-labelling | PaddleOCR (`lang="latin"`, cached models) proposes a plate and text. It is a pseudo-label, not ground truth. |
| Review bucket | Samples below the OCR confidence / plate sharpness thresholds go to `review/`, never to the clean set. |
| Negatives | Passing vehicles are saved as hard negatives. Disable with `--no-negatives`. |
| Attributes | Body type and colour are labelled **by a human**, never pseudo-labelled: an automatic labeller would train the classifier on the answer of the rule it is later compared against. Every classified track is queued, passers included. |
| Vocabulary | Types and colours live in `global.vehicle_types` / `global.vehicle_colors` in `zones_config.json`. They are site-specific and become class/directory names, so renaming a class after labelling invalidates the labels. |
| Synthetic plates | Clean Bolivian template (blue on white, `BOLIVIA` header, dept sticker) with the number filling 84% x 52% of the plate, then degraded in physical order: light wear → geometry → light → sensor. Calibrated against the served crops: brightness 110 vs 101, contrast 37 vs 38, sharpness 277 vs 315, warm cast R/B 1.49 vs 1.55. Text 98.5% `NNNNLLL` / 1.5% `NNNLLL`, number sampled as an integer so a leading zero is impossible; dept all 9 letters with `O` weighted. Train-only, never test. |

### Output layout

```
dataset/
├── vehicles/{images,labels}/   # vehicle crop + YOLO bbox of the plate  -> plate detection
├── plates/{images,labels}/     # tight plate crop + text label          -> OCR
├── review/                     # low-confidence pseudo-labels for human review
├── negatives/images/           # passing vehicles (hard negatives)
├── attributes/
│   ├── images/                 # vehicle crop awaiting a label   -> body type + colour
│   ├── manifest.jsonl          # queue of crops awaiting a label
│   └── labels.jsonl            # human labels, append-only
└── manifest.jsonl              # one row per emitted sample (synthetic rows carry synth: true)
```

### Layout

| File | Role |
|------|------|
| `extract_dataset.py` | Extractor CLI entry point. |
| `generate_synth_plates.py` | Synthetic plate generator CLI (train-only). |
| `prepare_ocr_ablation.py` | Deterministic real/synth ablation splits + protocol. |
| `label_attributes.py` | Interactive body-type and colour labeller (resumable). |
| `calibrate_zones.py` | Interactive real-time zone calibrator (writes the config). |
| `label_plate_box_gui.py` | Plate-box review with the CRNN's own 192x48 view, plus `--export-ocr` to cut OCR crops from the verified boxes. |
| `label_plate_text_gui.py` | Interactive plate-text labeller, prefilled with Paddle's read; records whether each label was accepted or corrected. |
| `view_trajectories.py` | Headless trajectory viewer (renders a PNG). |
| `plate_dataset/attributes.py` | Attribute vocabulary, label queue and append-only label store. |
| `plate_dataset/labelling.py` | Interactive labeller implementation (cv2 window). |
| `plate_dataset/calibrate.py` | Calibrator implementation. |
| `plate_dataset/config.py` | Zones + thresholds; generates `zones_config.json`. |
| `plate_dataset/pipeline.py` | Two-pass orchestration. |
| `plate_dataset/tracking.py` | Per-track trajectory and entry/pass classification. |
| `plate_dataset/synth_plates/` | Synthetic plates (renderer, degradations, pipeline) + ablation split and OCR metrics. |
| `plate_dataset/plate_ocr.py` | PaddleOCR wrapper and plate filter. |
| `plate_dataset/writer.py` | Dataset folders, labels and manifest. |

## Measuring vehicle detection (H1.1)

A separate, much smaller dataset from the one above: it holds **full frames**,
one per vehicle pass, because a detection metric needs the vehicle in context
and the extractor stores crops.

```bash
# 1. One frame per pass, with low-confidence boxes as proposals
uv run python tools/export_detection_frames.py \
    --videos /mnt/nina/PARKING --hour 2026033010 \
    --output ~/Smart_Parking_dataset/detection

# 2. A person accepts, deletes or draws. Resumable, one label file per image.
#    --blind-every 4 opens one image in four with no proposals: that subset is
#    the only one that can measure a miss the detector never proposed.
uv run python tools/label_boxes_gui.py --root ~/Smart_Parking_dataset/detection \
    --blind-every 4

# 3. AP@50 with a bootstrap interval, one row per model
uv run python tools/eval_vehicle_detection.py \
    --root ~/Smart_Parking_dataset/detection \
    --model yolo26n.pt --model yolo26s.pt \
    --json ~/Smart_Parking_dataset/detection/ap50.json
```

| File | Role |
|------|------|
| `export_detection_frames.py` | Samples the footage, tracks vehicles, writes one full frame per confirmed pass plus `proposals.jsonl`. |
| `label_boxes_gui.py` | Multi-box review. `--check` reports how much is left without opening a window. |
| `eval_vehicle_detection.py` | Greedy IoU matching, AP@50, precision/recall at the served threshold, both intervals. |
| `detection_dataset.py` | Label IO, IoU, matching, AP and intervals. Pure: no window, no model. |

Two decisions worth remembering. **One frame per pass, not per second**: frames
of the same car are near-duplicates, and sampling by time yields an interval
that is narrower than the evidence. **One class**: H1.1 asks whether a vehicle is
found; body type is H1.2 and is measured on crops.

**Proposals are visible except on the blind subset, and that is deliberate.** With
a box already drawn, a reviewer stops looking for vehicles the detector missed,
and recall stops being measurable. `review_journal.jsonl` records whether each
label was saved untouched; `--check` prints the acceptance rate, which belongs in
the thesis next to the AP.

## Measuring access events (H1.5)

Ground truth here is a human count of crossings, taken on the recording itself so
it lives in seconds of video and survives any later replay.

```bash
# 1. Watch the recording and mark every entry and exit
uv run python tools/label_events_gui.py \
    --video /mnt/nina/PARKING/C.OCCIDENTE_ch16_main_20260330100012_20260330110012.mp4 \
    --output ~/Smart_Parking_dataset/truth/ch16_2026033010.csv --speed 2

# 2. During the replay, save the history window from the API
curl -s 'http://127.0.0.1:8000/api/v1/stats/history?limit=500' \
    -b cookies.txt > ~/Smart_Parking_dataset/truth/ch16_2026033010_system.json

# 3. Compare, shifting the wall clock onto the video timeline
uv run python tools/compare_events.py \
    --truth ~/Smart_Parking_dataset/truth/ch16_2026033010.csv \
    --system ~/Smart_Parking_dataset/truth/ch16_2026033010_system.json \
    --replay-start 2026-11-02T21:00:00 --tolerance 3
```

| File | Role |
|------|------|
| `label_events_gui.py` | Real-time playback of the recording with entry/exit keys. Speed and frame stepping change while it runs; resumes an existing CSV. |
| `export_system_events.py` | Dumps the crossings the system recorded, from the detection log (includes exits that matched no session). |
| `compare_events.py` | One-to-one matching by direction and proximity, exactness, recall with interval. |
| `event_truth.py` | Timeline shift, matching, metrics. Pure, no window, no database. |

**Exactness is `TP / (TP + FP + FN)`**, not recall: an event the system invents
costs the same as one it misses, which is what an operator feels. Recall gets the
only interval that is honestly binomial — over ground-truth crossings.

## Checklist

- [ ] `zones_config.json` zones match what the trajectory viewer shows.
- [ ] Entry curves (green) end inside `inside_zone`; pass curves (red) never do.
- [ ] `manifest.jsonl` rows have the expected `kind` (`entry` / `pass`).
- [ ] Spot-check a few `plates/labels/*.txt` against the crops before training.
- [ ] `global.vehicle_types` / `global.vehicle_colors` match the vehicles actually seen on site.
- [ ] `attributes/labels.jsonl` covers the whole queue (`label_attributes.py --stats` shows `pending: 0`).

## Next step

Calibrate both cameras (`channel16`, `channel6`), then run the extractor over a
full day, review the `review/` bucket and label the attribute queue with
`tools/label_attributes.py`.

The vehicle type and colour classifiers built from those attributes are
documented separately in [`CLASSIFIERS.md`](CLASSIFIERS.md).
