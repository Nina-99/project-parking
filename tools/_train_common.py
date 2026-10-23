"""Shared training core for the vehicle type and colour classifiers.

Both are transfer-learning runs over the same kind of input: a crop holding one
vehicle, produced by the COCO detector, with the class of its largest box as the
label. Almost everything is shared -- the dataset reader, the model factory, the
training loop, the metric summary.

What is deliberately *not* shared is the augmentation pipeline. A type model is
judged on silhouette and can survive saturation jitter; a colour model cannot,
because saturation is the label. ``build_transforms`` therefore takes the task
and refuses to build anything it does not recognise.

The two thin entry points are ``train_type_classifier.py`` and
``train_color_classifier.py``. Evaluation lives in ``eval_type_classifier.py``,
which imports from here so training-time and standalone numbers come from the
same code.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SPLITS = ("train", "valid", "test")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# A class needs this many test samples before its F1 is reported. Below it the
# 95% interval is wider than +/-0.15 and the per-class number is noise, not a
# result. Fixed before the experiment ran: see the methodology section.
MIN_TEST_SUPPORT = 40


def read_class(label_path: Path) -> int | None:
    """Class of the largest box in a YOLO label file (the main vehicle)."""
    best_class: int | None = None
    best_area = -1.0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            area = float(parts[3]) * float(parts[4])
        except ValueError:
            continue
        if area > best_area:
            best_area = area
            best_class = cls
    return best_class


class VehicleCropDataset(Dataset):
    """Vehicle crops with one label per crop, read from a YOLO-style export."""

    def __init__(self, root: Path, split: str, transform) -> None:
        self.transform = transform
        self.items: list[tuple[Path, int]] = []
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            return
        for img in sorted(img_dir.glob("*.jpg")):
            label = lbl_dir / f"{img.stem}.txt"
            if not label.exists():
                continue
            cls = read_class(label)
            if cls is not None:
                self.items.append((img, cls))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, cls = self.items[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), cls

    def class_counts(self, num_classes: int) -> list[int]:
        counts = Counter(c for _, c in self.items)
        return [counts.get(i, 0) for i in range(num_classes)]


def build_transforms(img_size: int, task: str = "type") -> tuple[transforms.Compose, transforms.Compose]:
    """Train/eval pipelines, split by what the model must not learn to ignore.

    Both pipelines are deliberately plain: crop, flip, exposure. Rotation and
    motion blur were measured on both tasks and cost more than they bought, so
    they are gone rather than left available.

    The one real difference is saturation jitter, and it runs the opposite way
    for the two labels: ``type`` is judged on silhouette, so colour jitter is
    harmless and helps; ``colour`` is judged on hue, so jittering saturation
    teaches the network to ignore part of its own signal.

    Two tasks, two pipelines. Sharing one is how a colour model ends up unable
    to separate red from grey.
    """
    if task not in ("type", "color"):
        raise ValueError(f"unknown task {task!r}")
    if task == "type":
        augmentation: list = [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ]
    else:
        augmentation = [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # Exposure only: no saturation, no hue. Never add them here.
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    train_tf = transforms.Compose([
        *augmentation,
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def build_model(backbone: str, num_classes: int) -> nn.Module:
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone {backbone!r}")
    return model


def confusion(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[t, p] += 1
    return matrix


def per_class_metrics(matrix: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    """Precision, recall and F1 per class from a confusion matrix."""
    precisions, recalls, f1s = [], [], []
    for i in range(matrix.shape[0]):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s


def _wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson score interval for a proportion.

    Wilson instead of the normal approximation because the normal one collapses
    to zero width at p = 1.0, which would report a class with two samples and
    one hit as perfectly estimated -- the opposite of the truth.
    """
    if n == 0:
        return 1.0
    p = min(max(p, 0.0), 1.0)
    denominator = 1.0 + z * z / n
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return half


def summarise(
    matrix: np.ndarray,
    names: list[str],
    min_support: int = MIN_TEST_SUPPORT,
) -> dict:
    """Per-class metrics plus an evaluation verdict per class.

    A class with fewer than ``min_support`` test samples is excluded from the
    reported macro average: its F1 confidence interval is too wide to support
    any claim. The excluded class still counts as a false positive against the
    classes it steals predictions from, so nothing is hidden -- only unclaimed.

    ``ci95`` is the Wilson score interval half-width, using the F1 as the
    proportion and the class support as n.
    """
    precisions, recalls, f1s = per_class_metrics(matrix)
    supports = [int(matrix[i, :].sum()) for i in range(len(names))]
    evaluated = [support >= min_support for support in supports]
    ci95 = [_wilson_halfwidth(f1, support) for f1, support in zip(f1s, supports, strict=True)]
    evaluated_idx = [i for i, ok in enumerate(evaluated) if ok]
    present_idx = [i for i in range(len(names)) if supports[i] > 0]
    evaluated_support = sum(supports[i] for i in evaluated_idx)
    return {
        "precision": {names[i]: precisions[i] for i in range(len(names))},
        "recall": {names[i]: recalls[i] for i in range(len(names))},
        "f1": {names[i]: f1s[i] for i in range(len(names))},
        "ci95": {names[i]: ci95[i] for i in range(len(names))},
        "support": {names[i]: supports[i] for i in range(len(names))},
        "evaluated": {names[i]: evaluated[i] for i in range(len(names))},
        "macro_f1_present": (
            float(np.mean([f1s[i] for i in present_idx])) if present_idx else 0.0
        ),
        "macro_f1_evaluated": (
            float(np.mean([f1s[i] for i in evaluated_idx])) if evaluated_idx else 0.0
        ),
        "f1_weighted_real": (
            sum(f1s[i] * supports[i] for i in evaluated_idx) / evaluated_support
            if evaluated_support else 0.0
        ),
        "evaluated_support": evaluated_support,
        "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
        "excluded_classes": [names[i] for i in range(len(names)) if not evaluated[i]],
        "min_test_support": min_support,
    }


def print_summary(summary: dict, names: list[str]) -> None:
    """One table: per class, with the excluded ones visibly marked."""
    print("\n=== TEST per-class ===")
    print(f"  {'clase':>12}  {'P':>5}  {'R':>5}  {'F1':>5}  {'±ci95':>6}  {'n':>5}  eval")
    for name in names:
        mark = "si" if summary["evaluated"][name] else "NO"
        print(
            f"  {name:>12}  {summary['precision'][name]:.3f}  {summary['recall'][name]:.3f}  "
            f"{summary['f1'][name]:.3f}  {summary['ci95'][name]:6.3f}  "
            f"{summary['support'][name]:5d}  {mark}"
        )
    print(
        f"\n  macro-F1 (evaluadas) = {summary['macro_f1_evaluated']:.4f}"
        f"   F1 ponderado por frecuencia real = {summary['f1_weighted_real']:.4f}"
    )
    print(
        f"  macro-F1 (todas las presentes, legado) = {summary['macro_f1_present']:.4f}"
        f"   accuracy = {summary['accuracy']:.4f}"
    )
    if summary["excluded_classes"]:
        print(
            f"  no evaluadas (n < {summary['min_test_support']}): "
            f"{', '.join(summary['excluded_classes'])}"
        )


@torch.no_grad()
def evaluate(model, loader, device, num_classes, amp: str = "bf16") -> tuple[float, np.ndarray, list[int], list[int]]:
    """Macro-F1 over the classes present, used for model selection.

    Deliberately keeps the legacy definition: changing the selection criterion
    mid-training would change the result for reasons unrelated to the fix under
    test. Only the reported test metrics use ``summarise``.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in loader:
        images = images.to(device)
        with autocast_ctx(device, amp):
            logits = model(images)
        preds = logits.argmax(1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(targets.tolist())
    matrix = confusion(y_true, y_pred, num_classes)
    _, _, f1s = per_class_metrics(matrix)
    present = [i for i in range(num_classes) if matrix[i, :].sum() > 0]
    macro_f1 = float(np.mean([f1s[i] for i in present])) if present else 0.0
    return macro_f1, matrix, y_true, y_pred


def autocast_ctx(device, amp: str = "bf16"):
    """Autocast context for the training step, or a no-op.
    bf16 only, never fp16: fp16 needs a GradScaler to avoid underflow, bf16 has
    the same exponent range as fp32 and does not. Ampere (the RTX 3050) executes
    bf16 natively, so the ~6% that fp16 would add over bf16 is not worth an extra
    moving part. Measured on this card: ResNet18 224 at batch 32 goes from 293
    (fp32) to 505 img/s (bf16) at the same batch size.

    Set ``--amp off`` when a number has to be comparable with a run trained
    before autocast was introduced.
    """
    if amp == "off" or torch.device(device).type != "cuda":
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.bfloat16)


def train_one_epoch(model, loader, optimizer, criterion, device, amp: str = "bf16") -> float:
    model.train()
    total = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with autocast_ctx(device, amp):
            loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))




def build_parser(task: str) -> argparse.ArgumentParser:
    """CLI shared by both trainers. ``task`` only sets the defaults.

    The augmentation pipeline is not a flag: a colour model must never be able
    to run with the type pipeline by accident, so the entry point decides.
    """
    descriptions = {
        "type": "Train the vehicle body-type classifier.",
        "color": "Train the vehicle colour classifier.",
    }
    parser = argparse.ArgumentParser(description=descriptions.get(task, task))
    parser.add_argument("--dataset", type=Path, required=True,
                        help="YOLO-style export root (with data.yaml, train/valid/test).")
    parser.add_argument("--backbone", default="resnet18",
                        choices=("resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=10,
                        help="Decoding workers. One per physical core up to "
                             "len(os.sched_getaffinity(0)), minus the main process.")
    parser.add_argument("--amp", choices=("bf16", "off"), default="bf16",
                        help="Autocast precision. 'off' to reproduce a pre-autocast run.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-dir", type=Path, default=Path("/tmp/opencode/checkpoints"))
    parser.add_argument("--name", default=task,
                        help="Checkpoint/metrics prefix (defaults to the task name).")
    parser.add_argument("--min-test-support", type=int, default=MIN_TEST_SUPPORT,
                        help="Classes with fewer test samples are trained but not "
                             "reported (see MIN_TEST_SUPPORT).")
    return parser


def main_for_task(task: str, argv: list[str] | None = None) -> None:
    """Train and evaluate one task. ``task`` selects the augmentation pipeline."""
    args = build_parser(task).parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = yaml.safe_load((args.dataset / "data.yaml").read_text(encoding="utf-8"))
    names = list(data["names"])
    num_classes = len(names)
    print(f"[{task}] classes ({num_classes}): {names}")

    train_tf, eval_tf = build_transforms(args.img_size, task=task)
    train_ds = VehicleCropDataset(args.dataset, "train", train_tf)
    valid_ds = VehicleCropDataset(args.dataset, "valid", eval_tf)
    test_ds = VehicleCropDataset(args.dataset, "test", eval_tf)
    print(f"[{task}] samples -> train {len(train_ds)}, valid {len(valid_ds)}, test {len(test_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{task}] device: {device}  amp: {args.amp}  workers: {args.num_workers}")
    # Fixed input size for every batch: let cuDNN pick its fastest kernels once.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    counts = train_ds.class_counts(num_classes)
    print(f"[{task}] train counts:", {names[i]: counts[i] for i in range(num_classes)})
    weights = torch.tensor(
        [len(train_ds) / (num_classes * c) if c else 0.0 for c in counts], dtype=torch.float32
    ).to(device)

    # persistent_workers keeps the decode processes alive across epochs; without
    # it, 40 epochs means 160 worker respawns for nothing.
    workers = max(0, args.num_workers)
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=workers, pin_memory=True,
                            persistent_workers=workers > 0),
        "valid": DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True,
                            persistent_workers=workers > 0),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=workers, pin_memory=True,
                           persistent_workers=workers > 0),
    }

    model = build_model(args.backbone, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.save_dir / f"{args.backbone}_{args.name}_best.pt"
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.perf_counter()
        loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device,
                               amp=args.amp)
        scheduler.step()
        f1, _, _, _ = evaluate(model, loaders["valid"], device, num_classes, amp=args.amp)
        epoch_s = time.perf_counter() - t_epoch
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), checkpoint)
            flag = "  <- best"
        else:
            flag = ""
        # flush: without it a piped stdout buffers for the whole run and the
        # training looks hung instead of simply quiet.
        # The timestamp is the point: without it a log cannot be timed after the
        # fact, only guessed from file mtimes.
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | {epoch_s:5.1f}s "
              f"| {len(train_ds) / epoch_s:6.1f} img/s | loss {loss:.4f} | "
              f"valid macro-F1 {f1:.4f}{flag}", flush=True)

    print(f"\n[{task}] best valid macro-F1: {best_f1:.4f}  ({checkpoint})")
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    _, matrix, _, _ = evaluate(model, loaders["test"], device, num_classes, amp=args.amp)
    summary = summarise(matrix, names, args.min_test_support)
    print_summary(summary, names)
    print("\n=== confusion (rows=true) ===")
    print("        " + " ".join(f"{n[:6]:>7s}" for n in names))
    for i, name in enumerate(names):
        print(f"  {name[:6]:>6s} " + " ".join(f"{v:7d}" for v in matrix[i]))

    metrics_file = args.save_dir / f"{args.backbone}_{args.name}_metrics.json"
    metrics_file.write_text(json.dumps({
        "backbone": args.backbone,
        "task": task,
        "amp": args.amp,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "dataset": str(args.dataset),
        "classes": names,
        # Legacy key: macro over every present class. Kept so older artefacts
        # stay readable; the reported number is ``test_macro_f1_evaluated``.
        "test_macro_f1": summary["macro_f1_present"],
        "test_macro_f1_evaluated": summary["macro_f1_evaluated"],
        "f1_weighted_real": summary["f1_weighted_real"],
        "test_accuracy": summary["accuracy"],
        "min_test_support": summary["min_test_support"],
        "excluded_classes": summary["excluded_classes"],
        "per_class_f1": summary["f1"],
        "per_class_precision": summary["precision"],
        "per_class_recall": summary["recall"],
        "per_class_ci95": summary["ci95"],
        "per_class_support": summary["support"],
        "per_class_evaluated": summary["evaluated"],
        "confusion": matrix.tolist(),
    }, indent=4))
    print(f"[{task}] metrics saved to {metrics_file}")
