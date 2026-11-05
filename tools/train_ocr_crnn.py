# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""CRNN training CLI for Bolivian plate OCR.

Two stages: pretraining on synthetic plates, then fine-tuning on real ones. The
splits come from ``ablation/`` so both arms are scored on the same test set.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _train_common import autocast_ctx  # noqa: E402
from plate_dataset.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from plate_dataset.synth_plates.ablation import OcrScores, score_pairs  # noqa: E402

# Alphabet: 0-9 + A-Z + Blank token (index 0)
# Deterministic hold-out of the synthetic set, so stage 1 can be judged on the
# task it is actually training for.
SYNTH_VAL_SIZE = 300


def seed_everything(seed: int) -> None:
    """Make two runs comparable.

    Without this, the randomly initialised BiLSTM head and the shuffling order
    differ on every run: measuring "2 epochs vs 4" across two such runs compares
    two different models, not two schedules. Measured 2026-10-28: the same
    configuration reached 91.3% and 61.5% on its first epoch purely by chance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Per-worker augmentation seed. Without it the jitter and blur differ
    between runs even when the main process is seeded."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {ch: i + 1 for i, ch in enumerate(ALPHABET)}
IDX2CHAR = {i + 1: ch for i, ch in enumerate(ALPHABET)}
BLANK_IDX = 0
NUM_CLASSES = len(ALPHABET) + 1  # 36 chars + 1 blank


class ResNetCRNN(nn.Module):
    """ResNet18 + BiLSTM + CTC Classifier for Plate OCR."""

    def __init__(self, num_classes: int = NUM_CLASSES, hidden_size: int = 512):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Preserve spatial width dimension in deeper layers for sequence modeling.
        # By setting stride to (2, 1) we only halve the height, keeping the width intact.
        # Input width 192 -> conv1/pool (48) -> layer1 (48) -> layer2 (48) -> layer3 (48) -> layer4 (48)
        resnet.layer2[0].conv1.stride = (2, 1)
        resnet.layer2[0].downsample[0].stride = (2, 1)
        resnet.layer3[0].conv1.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)
        resnet.layer4[0].conv1.stride = (2, 1)
        resnet.layer4[0].downsample[0].stride = (2, 1)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        # Batch Normalization between CNN and RNN for stability
        self.bn_seq = nn.BatchNorm1d(512)

        self.bilstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        # Dropout before final classification to prevent overfitting on real plates
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, 512, 1, W')
        features = self.avgpool(features)  # (B, 512, 1, W')
        features = features.squeeze(2)  # (B, 512, W')

        # Apply BatchNorm over the sequence
        features = self.bn_seq(features)

        features = features.permute(0, 2, 1)  # (B, W', 512)

        recurrent_out, _ = self.bilstm(features)  # (B, W', 2*hidden_size)
        recurrent_out = self.dropout(recurrent_out)
        logits = self.classifier(recurrent_out)  # (B, W', num_classes)
        return logits.permute(1, 0, 2)  # (W', B, num_classes) for CTCLoss


_CLAHE_CACHE: dict[tuple[float, int], cv2.CLAHE] = {}


def _clahe(clip: float = 2.0, tile: int = 4) -> "cv2.CLAHE":
    """Per-process CLAHE instance.

    cv2.CLAHE objects are not picklable, so they cannot live on the Dataset
    (workers would fail to fork). A module-level cache gives each worker its
    own, built once.
    """
    key = (clip, tile)
    if key not in _CLAHE_CACHE:
        _CLAHE_CACHE[key] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return _CLAHE_CACHE[key]


class PlateOcrDataset(Dataset):
    """Loads plate crops and labels from dataset root given stem list."""

    def __init__(
        self,
        root: Path,
        stems: list[str],
        augment_mode: str = "none",
        img_height: int = 48,
        img_width: int = 192,
        use_clahe: bool = True,
        crop_jitter: bool = False,
        right_pad: float = 0.0,
    ):
        self.root = root
        # Only paths are held. Labels are read in __getitem__, because reading
        # them here meant 300k file reads at construction -- and every DataLoader
        # worker inherits that list when it forks.
        self.items: list[Path] = []
        self.image_dir = plates_dir = None  # set below, kept for lazy labels
        self.augment_mode = augment_mode
        self.img_height = img_height
        self.img_width = img_width
        self.use_clahe = use_clahe
        self.crop_jitter = crop_jitter
        self.right_pad = right_pad

        plates_dir = root / "plates"
        img_dir = plates_dir / "images"
        self.image_dir = img_dir
        self.label_dir = plates_dir / "labels"

        stem_set = set(stems)
        for sub in img_dir.glob("**/*.jpg"):
            if sub.stem in stem_set:
                self.items.append(sub)

    def __len__(self) -> int:
        return len(self.items)

    def _label_for(self, img_path: Path) -> str:
        """Read the text label on demand, keeping only valid characters."""
        lbl_path = self.label_dir / img_path.relative_to(self.image_dir).with_suffix(".txt")
        if not lbl_path.exists():
            return ""
        text = lbl_path.read_text(encoding="utf-8").strip().upper()
        return "".join(c for c in text if c in CHAR2IDX)

    def apply_motion_blur(self, img: np.ndarray, size: int = 5) -> np.ndarray:
        kernel = np.zeros((size, size))
        # Horizontal motion blur
        kernel[int((size - 1) / 2), :] = np.ones(size) / size
        return cv2.filter2D(img, -1, kernel)

    def add_salt_and_pepper_noise(self, img: np.ndarray, prob: float = 0.02) -> np.ndarray:
        noisy = np.copy(img)
        black = np.random.random(noisy.shape[:2]) < (prob / 2)
        white = np.random.random(noisy.shape[:2]) < (prob / 2)
        noisy[black] = [0, 0, 0]
        noisy[white] = [255, 255, 255]
        return noisy

    def apply_shear(self, img: np.ndarray, max_shear: float = 0.2) -> np.ndarray:
        h, w = img.shape[:2]
        shear = random.uniform(-max_shear, max_shear)
        M = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img_path = self.items[idx]
        label = self._label_for(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            # Corrupt/truncated JPEG (a few survived the copy off the failing
            # disk). Returning a black image with a valid label teaches the
            # model to hallucinate text from nothing, so resample a neighbour.
            for k in range(1, min(16, len(self.items))):
                alt = self.items[(idx + k) % len(self.items)]
                alt_img = cv2.imread(str(alt))
                if alt_img is not None:
                    img, label = alt_img, self._label_for(alt)
                    break
            else:
                img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        if img is not None:
            # CRITICAL FIX: OpenCV loads in BGR, PyTorch ResNet expects RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.use_clahe:
                # Real crops are night captures: median brightness 77 vs 106
                # for synthetic, contrast 21 vs 34. Unnormalised, the model
                # sees an input distribution it never trained on and cannot
                # read the glyphs. Measured on 32 real plates: 64% -> 99.6%
                # character accuracy once CLAHE is applied.
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = _clahe().apply(lab[:, :, 0])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Aggressive Data Augmentation ONLY for real fine-tuning phase
        if self.augment_mode == "heavy":
            if random.random() < 0.4:
                factor = random.uniform(0.6, 1.4)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)

            if random.random() < 0.3:
                if random.random() < 0.5:
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                else:
                    blur_size = random.choice([3, 5])
                    img = self.apply_motion_blur(img, blur_size)

            if random.random() < 0.2:
                img = self.apply_shear(img)

            if random.random() < 0.15:
                img = self.add_salt_and_pepper_noise(img)

            if random.random() < 0.2:
                kernel = np.ones((2, 2), np.uint8)
                if random.random() < 0.5:
                    img = cv2.erode(img, kernel, iterations=1)
                else:
                    img = cv2.dilate(img, kernel, iterations=1)
        elif self.augment_mode == "light":
            # Just brightness variations to prevent extreme overfitting to perfect digital colors
            if random.random() < 0.3:
                factor = random.uniform(0.7, 1.3)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # Simulate the detector cutting the box differently than the crop the
        # model was trained on. Training crops were cut by Paddle's box (tight to
        # the glyphs); at inference our own detector cuts them, and that 5-point
        # drop (59.4% end-to-end vs 64.3% with a human box) is framing, not
        # recognition. Cheap invariance beats recollecting parent vehicle crops
        # that no longer exist on disk.
        if self.crop_jitter:
            height, width = img.shape[:2]
            scale_x = random.uniform(0.94, 1.10)
            scale_y = random.uniform(0.90, 1.12)
            centre_x = width / 2.0 + random.uniform(-0.03, 0.03) * width
            centre_y = height / 2.0 + random.uniform(-0.05, 0.05) * height
            box_w, box_h = width * scale_x, height * scale_y
            left, top = centre_x - box_w / 2.0, centre_y - box_h / 2.0
            matrix = np.float32([[box_w / width, 0, left], [0, box_h / height, top]])
            img = cv2.warpAffine(
                img, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

        # Resize to fixed height/width
        # Blank space after the last glyph. Without it the final character sits
        # on the last CTC timestep with no context to its right, and the measured
        # error profile says exactly that: positions 4/5/6 accumulate 30/34/40
        # errors against 11-18 for the digits, all of them on the right stroke
        # (R->N, Y->T, F->E, G->E).
        if self.right_pad > 0:
            height, width = img.shape[:2]
            extra = max(1, int(width * self.right_pad))
            img = cv2.copyMakeBorder(img, 0, 0, 0, extra, cv2.BORDER_REPLICATE)
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)
        # Normalize to [0, 1] and CHW format
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, label


def ctc_collate_fn(batch: list[tuple[torch.Tensor, str]]):
    imgs, labels = zip(*batch)
    imgs_tensor = torch.stack(imgs, dim=0)

    target_indices = []
    target_lengths = []
    for lbl in labels:
        indices = [CHAR2IDX[c] for c in lbl]
        target_indices.extend(indices)
        target_lengths.append(len(indices))

    targets_tensor = torch.tensor(target_indices, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)

    return imgs_tensor, targets_tensor, target_lengths_tensor, labels


def post_process_bolivian_plate(text: str) -> str:
    """Applies heuristic formatting rules for Bolivian license plates.

    Standard Bolivian format: [3 or 4 Digits] + [3 Letters].
    E.g., 1234ABC or 123ABC.
    """
    if len(text) < 6:
        return text  # Too short to apply fixed format safely

    # Split the plate: last 3 characters are letters, the rest (prefix) are digits
    letters_part = text[-3:]
    digits_part = text[:-3]

    # Dictionaries for common OCR confusions
    char_to_digit = {
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "A": "4",
        "S": "5",
        "B": "8",
        "G": "6",
    }
    digit_to_char = {"0": "O", "1": "I", "2": "Z", "7": "Z", "5": "S", "8": "B", "6": "G", "4": "A"}

    corrected_digits = "".join(char_to_digit.get(c, c) for c in digits_part)
    corrected_letters = "".join(digit_to_char.get(c, c) for c in letters_part)

    return corrected_digits + corrected_letters


def decode_ctc(logits: torch.Tensor, apply_heuristics: bool = True) -> list[str]:
    """Greedy CTC Decoder with optional heuristic post-processing."""
    # logits shape: (W', B, num_classes)
    probs = logits.softmax(2)
    preds = probs.argmax(2).permute(1, 0).cpu().numpy()  # (B, W')

    decoded = []
    for pred in preds:
        chars = []
        prev = BLANK_IDX
        for idx in pred:
            if idx != BLANK_IDX and idx != prev:
                chars.append(IDX2CHAR[idx])
            prev = idx

        raw_text = "".join(chars)
        if apply_heuristics:
            raw_text = post_process_bolivian_plate(raw_text)

        decoded.append(raw_text)
    return decoded


def evaluate(
    model: ResNetCRNN,
    dataloader: DataLoader,
    device: torch.device,
    dump_path: Path | None = None,
) -> tuple[OcrScores, OcrScores]:
    model.eval()
    pairs_raw: list[tuple[str, str]] = []
    pairs_heuristic: list[tuple[str, str]] = []

    with torch.no_grad():
        for imgs, _, _, labels in dataloader:
            imgs = imgs.to(device)
            logits = model(imgs)

            preds_raw = decode_ctc(logits, apply_heuristics=False)
            preds_heuristic = decode_ctc(logits, apply_heuristics=True)

            pairs_raw.extend(list(zip(labels, preds_raw)))
            pairs_heuristic.extend(list(zip(labels, preds_heuristic)))

    if dump_path is not None:
        # The plate text doubles as the cluster id: repeated crops of one plate
        # share it, so bootstrap_ci.py can resample plates instead of crops.
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as fh:
            for index, ((truth, pred), (_, pred_heur)) in enumerate(
                zip(pairs_raw, pairs_heuristic, strict=True)
            ):
                fh.write(
                    json.dumps(
                        {
                            "stem": f"{truth or 'blank'}#{index:04d}",
                            "plate": truth,
                            "truth": truth,
                            "pred": pred,
                            "pred_heuristic": pred_heur,
                        }
                    )
                    + "\n"
                )
        print(f"[ocr-eval] wrote {len(pairs_raw)} predictions to {dump_path}")

    return score_pairs(pairs_raw), score_pairs(pairs_heuristic)


def _save_resume(path: Path, epoch: int, model, optimizer, scheduler) -> None:
    """Persist one epoch so an interrupted run can continue instead of restarting.

    Written to a temp file and moved into place: a kill during the write must
    never leave a truncated checkpoint that looks valid on resume.
    """
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        tmp,
    )
    tmp.replace(path)


def _load_resume(path: Path, model, optimizer, scheduler, device) -> int:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["epoch"]) + 1


def set_stage_freezing(model: ResNetCRNN, stage: str):
    if stage == "stage1_synth":
        # Stage 1: freeze early backbone (conv1, bn1, relu, maxpool, layer1,
        # layer2 = backbone[0:6]) and leave layer3/layer4 (backbone[6:8])
        # trainable so the network can adapt ImageNet features for character
        # discrimination.
        #
        # Full backbone freeze was tested and plateaus at ~55% char accuracy
        # because the BiLSTM+FC head alone cannot reshape generic ImageNet
        # features into glyph-level representations.
        #
        # The earlier run that converged to 95.4% (ocr_300k_stage1) had
        # layer3/layer4 trainable.  The key addition here is *differential
        # learning rates* (see build_stage1_optimizer): backbone layers adapt
        # at lr/10 so the pretrained weights evolve gently instead of
        # collapsing under the head's 3e-3.
        print(
            "[transfer-learning] Stage 1: backbone[0:6] frozen, "
            "layer3/layer4 trainable with differential LR"
        )
        for i, layer in enumerate(model.backbone):
            frozen = i < 6  # 0-5 = conv1, bn1, relu, maxpool, layer1, layer2
            for param in layer.parameters():
                param.requires_grad = not frozen
    elif stage == "stage2_real":
        print("[transfer-learning] Stage 2: UNFREEZING full backbone for deep real adaptation")
        for param in model.backbone.parameters():
            param.requires_grad = True

    for param in model.bilstm.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.bn_seq.parameters():
        param.requires_grad = True


def build_stage1_optimizer(model: ResNetCRNN, lr: float) -> optim.Adam:
    """Adam with differential LR: backbone layer3/4 at lr/10, head at lr.

    This prevents the CTC blank collapse that happens when 10.5M backbone
    params train at the same high rate as the randomly-initialized head.
    """
    backbone_trainable = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.bn_seq.parameters())
        + list(model.bilstm.parameters())
        + list(model.dropout.parameters())
        + list(model.classifier.parameters())
    )
    return optim.Adam(
        [
            {"params": backbone_trainable, "lr": lr / 10},
            {"params": head_params, "lr": lr},
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CRNN OCR model using 2-stage transfer learning."
    )
    parser.add_argument("--dataset", type=Path, help="Dataset root path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--stage", choices=["stage1_synth", "stage2_real", "full_pipeline"], default="full_pipeline"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seeds python, numpy, torch and every DataLoader "
        "worker. Two runs with the same seed differ only by "
        "the schedule under test.",
    )
    parser.add_argument("--epochs-synth", type=int, default=25)
    parser.add_argument("--epochs-real", type=int, default=30)
    # Higher learning rate for synth to escape the blank token local minimum
    parser.add_argument("--lr-synth", type=float, default=3e-3)
    parser.add_argument(
        "--lr-schedule",
        choices=("plateau", "fixed"),
        default="plateau",
        help="'plateau' halves the LR on stagnation. On a task that "
        "plateaus early it can freeze the run; 'fixed' keeps it "
        "constant so the LR is not the variable.",
    )
    parser.add_argument(
        "--synth-augment",
        choices=("none", "light", "heavy"),
        default="light",
        help="Augmentation for the synthetic stage. 'light' is only "
        "brightness, which lets the model memorise the single "
        "rendering instead of learning the glyphs.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1500,
        help="Linear LR warmup over the first N batches of stage 1. "
        "Without it, a 3e-3 first step collapses CTC to the "
        "all-blank equilibrium about half the time.",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.5,
        help="Gradient-norm clip. 1.5 was blocking CTC convergence; "
        "raise it to see whether the plateau moves.",
    )
    parser.add_argument(
        "--right-pad",
        type=float,
        default=0.0,
        help="Blank columns appended to the right of every crop before the resize, "
        "as a fraction of its width. Gives the last character timesteps to land on.",
    )
    parser.add_argument(
        "--crop-jitter",
        action="store_true",
        help="Randomly shift/scale the framing of every real crop. "
        "Use when the detector that cuts at inference is not "
        "the one that cut the training crops.",
    )
    parser.add_argument(
        "--synth-mix-ratio",
        type=float,
        default=0.5,
        help="Fraction of each stage-2 epoch that comes from synth "
        "data (0.0 = real only, 0.5 = equal mix). Synth batches "
        "prevent catastrophic forgetting of character features "
        "learned in stage 1.",
    )
    parser.add_argument("--lr-real", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=8,
        help="Decoding workers. Zero makes the GPU wait on a single "
        "core to decode 300k JPEGs, which is most of the epoch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the current stage from resume_stage{1,2}.pt if present.",
    )
    parser.add_argument(
        "--amp",
        choices=("bf16", "off"),
        default="bf16",
        help="Autocast precision for the backbone. CTC is always fp32.",
    )
    parser.add_argument("--save-dir", type=Path, default=Path("/tmp/opencode/ocr_checkpoints"))
    parser.add_argument(
        "--dump-test-predictions",
        type=Path,
        help="Write per-item test predictions as JSONL (truth, pred, plate). Feed the "
             "file to bootstrap_ci.py items --task text --cluster-col plate to get an "
             "interval that resamples plates instead of crops.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training: load --checkpoint (or the stage-2 checkpoint in --save-dir) "
             "and report the test metrics. Combine with --dump-test-predictions to "
             "export predictions from an existing model.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Weights for --eval-only.")
    return parser


def build_scheduler(optimizer: optim.Optimizer, mode: str, patience: int):
    """LR schedule, or None for a constant rate.

    ''plateau'' halves the LR whenever the loss stops improving for ``patience``
    epochs. On a task that plateaus early that is a trap rather than a rescue:
    the LR keeps halving, the plateau consolidates, and the run freezes at a
    mediocre point while looking like it is still training.
    """
    if mode == "fixed":
        return None
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )


def current_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    clip_grad: float = 1.5,
    amp: str = "bf16",
    warmup_steps: int = 0,
):
    model.train()
    total_loss = 0.0
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    step = 0
    for imgs, targets, target_lengths, _ in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        with autocast_ctx(device, amp):
            logits = model(imgs)  # (W', B, num_classes)
        # CTC stays in fp32. log_softmax over 37 classes in bf16 loses precision
        # exactly where CTC needs it (paths that sum to the target), and a loss
        # that goes inf/NaN under autocast burns the whole run. Measured: the
        # autocast only covers the backbone, which is where the speedup lives.
        logits = logits.float()
        w_prime, b_size, _ = logits.shape

        input_lengths = torch.full(
            size=(b_size,), fill_value=w_prime, dtype=torch.long, device=device
        )
        log_probs = logits.log_softmax(2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()

        # Clipping guards against the blank-token collapse, but 1.5 is strict
        # enough to hold CTC back: with Adam at 3e-3 the model can plateau far
        # from converged. It is a knob now, because the right value is an
        # empirical question, not a matter of taste.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        # Linear warmup: a freshly initialised head taking a 3e-3 step immediately
        # falls into the all-blank CTC equilibrium about half the time. Ramping
        # in over the first few hundred batches gives it a gentler start.
        if warmup_steps > 0:
            scale = min(1.0, (step + 1) / warmup_steps)
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * scale
        step += 1

        optimizer.step()

        total_loss += loss.item()
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base
    return total_loss / len(dataloader)


def train_one_epoch_mixed(
    model,
    real_loader,
    synth_iter,
    optimizer,
    criterion,
    device,
    clip_grad: float = 1.5,
    amp: str = "bf16",
):
    """Train one epoch interleaving real and synth batches.

    For every real batch, one synth batch is replayed.  The synth data keeps
    the character-recognition features alive while the backbone adapts to
    real-world image statistics.  Without this, 395 real images overwrite
    the 360k-sample pretraining in a few dozen gradient steps (catastrophic
    forgetting).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    def _step(imgs, targets, target_lengths):
        nonlocal total_loss, n_batches
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        optimizer.zero_grad()
        with autocast_ctx(device, amp):
            logits = model(imgs)
        logits = logits.float()
        w_prime, b_size, _ = logits.shape
        input_lengths = torch.full(
            size=(b_size,),
            fill_value=w_prime,
            dtype=torch.long,
            device=device,
        )
        log_probs = logits.log_softmax(2)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    for imgs, targets, target_lengths, _ in real_loader:
        _step(imgs, targets, target_lengths)
        # Synth replay — one batch per real batch
        if synth_iter is not None:
            try:
                s_imgs, s_targets, s_tl, _ = next(synth_iter)
                _step(s_imgs, s_targets, s_tl)
            except StopIteration:
                synth_iter = None  # exhausted; continue real-only

    return total_loss / max(n_batches, 1)


def build_stage2_optimizer(model: ResNetCRNN, lr: float) -> optim.Adam:
    """Adam with differential LR for stage 2: backbone at lr/10, head at lr.

    Even at lr 2e-4 the full backbone (11.2M params) can overwrite the
    character features learned in stage 1 when trained on only 395 images.
    Differential LR lets the backbone drift gently while the head adapts.
    """
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.bn_seq.parameters())
        + list(model.bilstm.parameters())
        + list(model.dropout.parameters())
        + list(model.classifier.parameters())
    )
    return optim.Adam(
        [
            {"params": backbone_params, "lr": lr / 10},
            {"params": head_params, "lr": lr},
        ]
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    _, glob = load_config(args.config)
    root = args.dataset or Path(glob.output_dir)

    ablation_dir = root / "ablation"
    train_synth_stems = (
        (ablation_dir / "train_synth.txt").read_text(encoding="utf-8").splitlines()
        if (ablation_dir / "train_synth.txt").exists()
        else []
    )
    train_real_stems = (
        (ablation_dir / "train_real.txt").read_text(encoding="utf-8").splitlines()
        if (ablation_dir / "train_real.txt").exists()
        else []
    )
    test_stems = (
        (ablation_dir / "test.txt").read_text(encoding="utf-8").splitlines()
        if (ablation_dir / "test.txt").exists()
        else []
    )

    print(f"[ocr-train] Dataset root: {root}")
    print(
        f"[ocr-train] Stems -> synth: {len(train_synth_stems)}, real_train: {len(train_real_stems)}, real_test: {len(test_stems)}"
    )

    if args.eval_only and not test_stems:
        raise SystemExit(
            f"[ocr-eval] there is nothing to evaluate: {ablation_dir / 'test.txt'} "
            "does not exist or is empty. Point --dataset at the root that contains "
            "'ablation/test.txt' and 'plates/images'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    if device.type == "cuda":
        # benchmark=True autotunes kernels on every run, so two seeded runs still
        # take different paths through the numerics. At lr 3e-3 CTC sits on a
        # knife edge: measured 2026-10-28, the same seed gave 89.3% and 11.2%
        # (blank collapse) on epoch 1 depending on the kernels cuDNN picked.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f"[ocr-train] Using device: {device}  amp: {args.amp}  seed: {args.seed}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_synth = args.save_dir / "crnn_stage1_synth.pt"
    checkpoint_final = args.save_dir / "crnn_stage2_real.pt"
    checkpoint_best = args.save_dir / "crnn_stage2_best.pt"

    model = ResNetCRNN().to(device)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    test_ds = PlateOcrDataset(root, test_stems, augment_mode="none")
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=args.loader_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    if args.eval_only:
        checkpoint = args.checkpoint
        if checkpoint is None:
            for candidate in (checkpoint_best, checkpoint_final):
                if candidate.exists():
                    checkpoint = candidate
                    break
        if checkpoint is None or not checkpoint.exists():
            raise SystemExit(
                "[ocr-eval] no checkpoint to evaluate; pass --checkpoint or train first"
            )
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        raw, heur = evaluate(model, test_loader, device, dump_path=args.dump_test_predictions)
        print(
            f"[ocr-eval] {checkpoint}\n"
            f"[ocr-eval] Raw  Acc: {raw.char_accuracy * 100:.1f}% "
            f"(EM:{raw.exact_match * 100:.1f}%)  n={raw.n}\n"
            f"[ocr-eval] Heur Acc: {heur.char_accuracy * 100:.1f}% "
            f"(EM:{heur.exact_match * 100:.1f}%)"
        )
        return

    stage1_ran = False
    # Stage 2 replays synth batches even when stage 1 is skipped, so seed the
    # pool from the full synth list here.
    synth_train_stems: list[str] = list(train_synth_stems)

    # --- STAGE 1: Synthetic Pretraining ---
    if args.stage in ["stage1_synth", "full_pipeline"] and train_synth_stems:
        print("\n=== STARTING STAGE 1: Synthetic Pretraining ===")
        set_stage_freezing(model, "stage1_synth")
        # Stage 1 used to be judged only on the real test set, so a run that was
        # not learning its own task looked identical to one that was learning it
        # and failing to transfer. Those are different problems with different
        # fixes. Hold out a deterministic slice of the synthetic set, sized
        # proportionally so a tiny fixture still trains on something.
        val_size = min(SYNTH_VAL_SIZE, len(train_synth_stems) // 10)
        synth_val_stems = train_synth_stems[-val_size:] if val_size else []
        synth_train_stems = train_synth_stems[:-val_size] if val_size else train_synth_stems

        # Synth images get light augmentation to avoid perfect-pixel memorization
        synth_ds = PlateOcrDataset(root, synth_train_stems, augment_mode=args.synth_augment)
        synth_loader = DataLoader(
            synth_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            num_workers=args.loader_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed),
        )
        synth_val_loader = None
        if synth_val_stems:
            synth_val_ds = PlateOcrDataset(root, synth_val_stems, augment_mode="none")
            synth_val_loader = DataLoader(
                synth_val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=ctc_collate_fn,
                num_workers=args.loader_workers,
                pin_memory=pin_memory,
                worker_init_fn=seed_worker,
            )

        optimizer = build_stage1_optimizer(model, args.lr_synth)
        # Slower LR decay to give CTC time to align characters
        scheduler = build_scheduler(optimizer, args.lr_schedule, patience=5)

        resume1 = args.save_dir / "resume_stage1.pt"
        start_epoch = 1
        if args.resume and resume1.exists():
            start_epoch = _load_resume(resume1, model, optimizer, scheduler, device)
            print(f"[ocr-train] resuming stage 1 at epoch {start_epoch}")
        for epoch in range(start_epoch, args.epochs_synth + 1):
            t_epoch = time.perf_counter()
            loss = train_one_epoch(
                model,
                synth_loader,
                optimizer,
                criterion,
                device,
                clip_grad=args.clip_grad,
                amp=args.amp,
                warmup_steps=args.warmup_steps if epoch == 1 else 0,
            )
            epoch_s = time.perf_counter() - t_epoch
            if scheduler is not None:
                scheduler.step(loss)
            # evaluate() returns both the raw and the heuristic decoding; stage 1
            # printed it as a single object and crashed on the first epoch.
            scores_raw, _ = evaluate(model, test_loader, device)
            synth_scores = (
                evaluate(model, synth_val_loader, device)[0]
                if synth_val_loader is not None
                else None
            )
            synth_part = ""
            if synth_scores is not None:
                synth_part = (
                    f"SynthVal Acc: {synth_scores.char_accuracy * 100:.1f}% "
                    f"(EM:{synth_scores.exact_match * 100:.1f}%) | "
                )
            n_synth = len(synth_train_stems)
            print(
                f"Epoch {epoch:02d}/{args.epochs_synth:02d} | {epoch_s:6.1f}s "
                f"| {n_synth / epoch_s:6.1f} img/s "
                f"| Synth Loss: {loss:.4f} "
                f"| LR {current_lr(optimizer):.2e} | "
                f"{synth_part}"
                f"RealTest Acc: {scores_raw.char_accuracy * 100:.1f}% "
                f"(EM:{scores_raw.exact_match * 100:.1f}%)",
                flush=True,
            )
            # A collapsed run looks like a slow learner for 5 epochs and then
            # gets thrown away. The all-blank equilibrium has a signature: CTC
            # loss sits above ln(num_classes) and does not move.
            if epoch >= 2 and loss > 2.0:
                raise SystemExit(
                    f"[ocr-train] CTC blank collapse: synth loss {loss:.3f} after "
                    f"{epoch} epochs, still at the all-blank equilibrium. "
                    "The head never started emitting. Try another --seed or a "
                    "lower --lr-synth; continuing would burn the remaining "
                    "epochs for nothing."
                )

            _save_resume(resume1, epoch, model, optimizer, scheduler)
        stage1_ran = True

        torch.save(model.state_dict(), checkpoint_synth)
        print(f"[ocr-train] Saved Stage 1 model to {checkpoint_synth}")

    # --- STAGE 2: Real Fine-Tuning ---
    if args.stage in ["stage2_real", "full_pipeline"] and train_real_stems:
        print("\n=== STARTING STAGE 2: Real Fine-Tuning ===")
        # On a resumed full_pipeline the stage-1 loop is skipped (start_epoch >
        # epochs_synth), so the in-memory model would be random weights unless we
        # load them here. Same for a standalone --stage stage2_real.
        if checkpoint_synth.exists() and (args.stage == "stage2_real" or not stage1_ran):
            model.load_state_dict(torch.load(checkpoint_synth, map_location=device))
            print(f"[ocr-train] Loaded Stage 1 weights from {checkpoint_synth}")

        set_stage_freezing(model, "stage2_real")
        # Real images need heavy augmentation to generalize
        real_ds = PlateOcrDataset(
            root, train_real_stems, augment_mode="heavy", crop_jitter=args.crop_jitter
        )
        real_loader = DataLoader(
            real_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=ctc_collate_fn,
            num_workers=args.loader_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed + 1),
        )

        # Synth replay: a small pool sampled once, reshuffled every epoch. The
        # full 360k set is unnecessary here -- mixed training only consumes
        # ~n_real_batches synth batches per epoch.
        synth_replay_stems: list[str] = []
        if args.synth_mix_ratio > 0 and synth_train_stems:
            n_replay = max(2000, int(len(train_real_stems) * args.synth_mix_ratio * 4))
            rng = random.Random(7)
            synth_replay_stems = rng.sample(
                synth_train_stems, min(n_replay, len(synth_train_stems))
            )
            print(
                f"[ocr-train] Stage 2 synth replay pool: {len(synth_replay_stems)} stems "
                f"(mix ratio {args.synth_mix_ratio})"
            )
        synth_replay_loader = None
        if synth_replay_stems:
            replay_ds = PlateOcrDataset(root, synth_replay_stems, augment_mode="light",
                                        right_pad=args.right_pad)
            synth_replay_loader = DataLoader(
                replay_ds,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=ctc_collate_fn,
                num_workers=args.loader_workers,
                pin_memory=pin_memory,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(args.seed + 2),
            )

        # Differential LR: backbone at lr/10 so it adapts without erasing the
        # character features learned in stage 1.
        optimizer = build_stage2_optimizer(model, args.lr_real)
        scheduler = build_scheduler(optimizer, args.lr_schedule, patience=4)

        # Stage 2 is cheap per epoch (seconds) and exact-match is noisy: on the
        # last run it peaked at epoch 29 and fell back at 30. Saving only the
        # final weights throws away the best model the run produced.
        best_em = -1.0
        best_epoch = 0
        resume2 = args.save_dir / "resume_stage2.pt"
        start_epoch = 1
        if args.resume and resume2.exists():
            start_epoch = _load_resume(resume2, model, optimizer, scheduler, device)
            print(f"[ocr-train] resuming stage 2 at epoch {start_epoch}")
        for epoch in range(start_epoch, args.epochs_real + 1):
            t_epoch = time.perf_counter()
            # Fresh synth iterator per epoch: the replay pool is reshuffled each
            # time, so every epoch sees a different subset of synthetic plates.
            synth_iter = iter(synth_replay_loader) if synth_replay_loader is not None else None
            loss = train_one_epoch_mixed(
                model,
                real_loader,
                synth_iter,
                optimizer,
                criterion,
                device,
                clip_grad=args.clip_grad,
                amp=args.amp,
            )
            epoch_s = time.perf_counter() - t_epoch
            if scheduler is not None:
                scheduler.step(loss)
            scores_raw, scores_heur = evaluate(model, test_loader, device)
            print(
                f"Epoch {epoch:02d}/{args.epochs_real:02d} | {epoch_s:6.1f}s "
                f"| Real Loss: {loss:.4f} "
                f"| LR {current_lr(optimizer):.2e} | "
                f"Raw Acc: {scores_raw.char_accuracy * 100:.1f}% (EM:{scores_raw.exact_match * 100:.1f}%) | "
                f"Heur Acc: {scores_heur.char_accuracy * 100:.1f}% "
                f"(EM:{scores_heur.exact_match * 100:.1f}%)",
                flush=True,
            )
            if scores_raw.exact_match > best_em:
                best_em = scores_raw.exact_match
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_best)
            _save_resume(resume2, epoch, model, optimizer, scheduler)

        torch.save(model.state_dict(), checkpoint_final)
        print(f"[ocr-train] Saved Stage 2 model to {checkpoint_final} (last epoch)")
        if checkpoint_best.exists():
            model.load_state_dict(torch.load(checkpoint_best, map_location=device))
            print(
                f"[ocr-train] Reporting the best epoch: {best_epoch} "
                f"(EM {best_em * 100:.1f}%), {checkpoint_best}"
            )

        final_raw, final_heur = evaluate(
            model, test_loader, device, dump_path=args.dump_test_predictions
        )
        metrics_file = args.save_dir / "final_metrics.json"
        metrics_file.write_text(
            json.dumps(
                {
                    "n_test": final_raw.n,
                    "selected": "best",
                    "best_epoch": best_epoch,
                    "raw_char_accuracy": final_raw.char_accuracy,
                    "raw_exact_match": final_raw.exact_match,
                    "heur_char_accuracy": final_heur.char_accuracy,
                    "heur_exact_match": final_heur.exact_match,
                },
                indent=4,
            )
        )
        print(f"[ocr-train] Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
