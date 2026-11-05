#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

# Sequential training runner for the smart-parking classifiers.
#
#   ./train_all.sh datasets   # rebuild type + colour datasets from the frozen pool
#   ./train_all.sh type       # ResNet18 body-type classifier
#   ./train_all.sh color      # ResNet18 colour classifier
#   ./train_all.sh ocr        # CRNN plate OCR (independent of the two above)
#   ./train_all.sh platedet   # plate localiser from the human-reviewed boxes
#   ./train_all.sh all        # datasets -> type -> color -> ocr
#
# Everything runs on one GPU, so the stages are strictly sequential: two
# trainings at once on a 4 GB card is how you get a CUDA OOM at hour six.
#
# Environment:
#   FORCE=1        rebuild the datasets even if they already exist
#   SMARTPARKING   dataset root (default /home/nina/FNI2025/ProjectParking/datasets)
#
# Every stage appends to its own log under $SMARTPARKING/runs/ and fails the
# whole run on a non-zero exit, so a broken stage never gets silently skipped.

set -euo pipefail

# Two roots, because they no longer live on the same disk:
#   DATA   -> frozen pool, type/colour datasets, run artefacts (NVMe)
#   PLATES -> plates/ + ablation/ for the OCR job
#
# Both used to be the USB HDD at /mnt/nina. That disk started returning EIO on
# ~1.3% of the pool crops and silently truncated 25 synthetic plate JPEGs, so
# everything was moved to the NVMe and the HDD is no longer a source of truth.
DATA="${SMARTPARKING:-/home/nina/FNI2025/ProjectParking/datasets}"
PLATES="${PLATES_ROOT:-/home/nina/FNI2025/ProjectParking}"
SW="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$DATA"
POOL="$DATA/pool_20261021"
RUNS="$DATA/runs"
TYPE_DS="$ROOT/type dataset v3"
COLOR_DS="$ROOT/color dataset v5"
TYPE_NAME="final_type_v3"
COLOR_NAME="final_color_v5"
STAMP="$(date +%Y%m%d_%H%M%S)"

# The type vocabulary changed on 2026-10-23: furgon was merged away and the
# indices compacted. Training against a dataset built before that is training a
# 6-class head on 5-class labels, so the counts are asserted, not assumed.
TYPE_CLASSES=5
COLOR_CLASSES=11

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

preflight() {
    command -v uv >/dev/null || die "uv is not on PATH"
    [[ -d "$POOL" ]] || die "frozen pool not found: $POOL"
    [[ -f "$POOL/data.yaml" ]] || die "missing $POOL/data.yaml"
    mkdir -p "$RUNS"

    # Match the actual python entry point, not this script's own command line:
    # `pgrep -f` sees the parent shell, whose argv contains the pattern text.
    if pgrep -f 'tools/train_(type|color)_classifier\.py|tools/train_ocr_crnn\.py' >/dev/null; then
        die "another training run is already active; wait for it or kill it"
    fi

    if command -v nvidia-smi >/dev/null; then
        local used
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
        log "GPU memory in use: ${used} MiB"
        (( used < 500 )) || log "WARNING: GPU is already busy; expect a slower run"
    else
        log "WARNING: no nvidia-smi, training will fall back to CPU"
    fi
}

# Run one python tool, log it, and fail the script if it fails.
run_tool() {
    local name="$1"; shift
    local out="$RUNS/$name.log"
    log "-> $name   (log: $out)"
    ( cd "$SW" && uv run python "$@" ) 2>&1 | tee "$out"
    log "   $name done"
}

check_dataset() {
    local ds="$1" want="$2"
    [[ -d "$ds" ]] || die "dataset not built: $ds"
    local got
    got="$(grep -m1 '^nc:' "$ds/data.yaml" | tr -dc '0-9')"
    [[ "$got" == "$want" ]] || die "$ds has nc=$got, expected $want (stale pre-merge dataset?)"
    if grep -q "furgon" "$ds/data.yaml"; then
        die "$ds still lists 'furgon' in data.yaml"
    fi
    log "   $ds: nc=$got ok"
}

stage_datasets() {
    local force_arg=()
    [[ "${FORCE:-0}" == 1 ]] && force_arg=(--force)

    if [[ -d "$TYPE_DS" && "${FORCE:-0}" != 1 ]]; then
        log "skip: $TYPE_DS already exists (FORCE=1 to rebuild)"
    else
        # Frame-hash split, identical to the method recorded in reports/type_v2.md:
        # a new label can never move an existing crop between splits.
        run_tool "dataset_type_$STAMP" tools/build_color_dataset.py \
            --pool "$POOL" \
            --output "$TYPE_DS" \
            --labels-dir labels \
            --data-yaml data.yaml \
            --min-count 10 \
            "${force_arg[@]}"
    fi

    if [[ -d "$COLOR_DS" && "${FORCE:-0}" != 1 ]]; then
        log "skip: $COLOR_DS already exists (FORCE=1 to rebuild)"
    else
        # Minibuses are excluded from colour: their livery makes the dominant
        # colour depend on the viewing angle (see CLASSIFIERS.md).
        run_tool "dataset_color_$STAMP" tools/build_color_dataset.py \
            --pool "$POOL" \
            --output "$COLOR_DS" \
            --labels-dir labels_color \
            --data-yaml data_color.yaml \
            --exclude-types minibus \
            --min-count 10 \
            "${force_arg[@]}"
    fi
}

stage_type() {
    check_dataset "$TYPE_DS" "$TYPE_CLASSES"
    run_tool "final_type_$STAMP" tools/train_type_classifier.py \
        --dataset "$TYPE_DS" \
        --save-dir "$RUNS" \
        --name "$TYPE_NAME"
}

stage_color() {
    check_dataset "$COLOR_DS" "$COLOR_CLASSES"
    run_tool "final_color_$STAMP" tools/train_color_classifier.py \
        --dataset "$COLOR_DS" \
        --save-dir "$RUNS" \
        --name "$COLOR_NAME"
}

stage_platedet() {
    # Needs the reviewed boxes: the pseudo-labels alone are not training data.
    local verified="$PLATES/plates_det/verified.jsonl"
    [[ -f "$verified" ]] || die "missing $verified (run the review GUI first)"
    run_tool "platedet_dataset_$STAMP" tools/build_plate_detection_dataset.py \
        --verified "$verified" --output "$PLATES/plates_det/dataset" --force
    run_tool "platedet_train_$STAMP" tools/train_plate_detector.py \
        --data "$PLATES/plates_det/dataset/data.yaml" \
        --save-dir "$PLATES/plates_det/runs" --name plate_det \
        --epochs "${PLATEDET_EPOCHS:-100}" --batch "${PLATEDET_BATCH:-8}"
}

stage_ocr() {
    # Without ablation/ the trainer reads an empty stem list and SILENTLY skips
    # stage 1, then fine-tunes on 395 real plates alone. Fail here instead.
    [[ -f "$PLATES/ablation/train_synth.txt" ]] \
        || die "missing $PLATES/ablation/train_synth.txt (stage 1 would be skipped)"
    [[ -d "$PLATES/plates/images" ]] \
        || die "missing $PLATES/plates/images"
    log "OCR root: $PLATES"
    log "NOTE: stage 1 is ~360k synthetic plates; stage 2 is 500 epochs on 395 real crops."
    # These are the measured settings, not defaults: lr-real 2e-4 with the
    # plateau scheduler never left the synthetic basin (14% char accuracy after
    # 100 epochs), and 30 real epochs is ~360 gradient steps for a 21.7M-param
    # model. See tools/train_ocr_crnn.py and the RUNBOOK for the numbers.
    run_tool "ocr_$STAMP" tools/train_ocr_crnn.py \
        --dataset "$PLATES" \
        --stage full_pipeline \
        --epochs-synth 8 \
        --epochs-real 500 \
        --lr-synth 3e-3 \
        --lr-real 1e-3 \
        --lr-schedule fixed \
        --synth-mix-ratio 0 \
        --crop-jitter \
        --amp "${AMP:-bf16}" \
        --save-dir "$RUNS/ocr_$STAMP"
}

main() {
    local stage="${1:-all}"
    preflight
    case "$stage" in
        datasets) stage_datasets ;;
        type)     stage_type ;;
        color)    stage_color ;;
        ocr)      stage_ocr ;;
        platedet) stage_platedet ;;
        all)
            stage_datasets
            stage_type
            stage_color
            stage_ocr
            stage_platedet
            ;;
        *) die "unknown stage '$stage' (datasets|type|color|ocr|platedet|all)" ;;
    esac
    log "finished: $stage"
    log "artifacts under $RUNS"
}

main "$@"
