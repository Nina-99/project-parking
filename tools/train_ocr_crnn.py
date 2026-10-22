"""CRNN PyTorch Training CLI for Bolivian License Plate OCR.

Implements Two-Stage Transfer Learning (Synthetic Pretraining + Real Fine-Tuning)
aligned with thesis ablation splits in ``ablation/``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plate_dataset.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from plate_dataset.synth_plates.ablation import OcrScores, score_pairs  # noqa: E402

# Alphabet: 0-9 + A-Z + Blank token (index 0)
# Deterministic hold-out of the synthetic set, so stage 1 can be judged on the
# task it is actually training for.
SYNTH_VAL_SIZE = 300

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


class PlateOcrDataset(Dataset):
    """Loads plate crops and labels from dataset root given stem list."""

    def __init__(
        self,
        root: Path,
        stems: list[str],
        augment_mode: str = "none",
        img_height: int = 48,
        img_width: int = 192,
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
            img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            # CRITICAL FIX: OpenCV loads in BGR, PyTorch ResNet expects RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

        # Resize to fixed height/width
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


def evaluate(model: ResNetCRNN, dataloader: DataLoader, device: torch.device) -> tuple[OcrScores, OcrScores]:
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

    return score_pairs(pairs_raw), score_pairs(pairs_heuristic)


def set_stage_freezing(model: ResNetCRNN, stage: str):
    if stage == "stage1_synth":
        print(
            "[transfer-learning] Stage 1: Freezing early ResNet layers (layer1/layer2) to prevent domain overfitting"
        )
        # Freeze conv1, bn1, layer1, layer2
        for name, param in model.backbone.named_parameters():
            if any(prefix in name for prefix in ["0.", "1.", "4.", "5."]):
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif stage == "stage2_real":
        print("[transfer-learning] Stage 2: UNFREEZING full backbone for deep real adaptation")
        for param in model.backbone.parameters():
            param.requires_grad = True

    for param in model.bilstm.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CRNN OCR model using 2-stage transfer learning."
    )
    parser.add_argument("--dataset", type=Path, help="Dataset root path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--stage", choices=["stage1_synth", "stage2_real", "full_pipeline"], default="full_pipeline"
    )
    parser.add_argument("--epochs-synth", type=int, default=25)
    parser.add_argument("--epochs-real", type=int, default=30)
    # Higher learning rate for synth to escape the blank token local minimum
    parser.add_argument("--lr-synth", type=float, default=3e-3)
    parser.add_argument("--lr-schedule", choices=("plateau", "fixed"), default="plateau",
                        help="'plateau' halves the LR on stagnation. On a task that "
                             "plateaus early it can freeze the run; 'fixed' keeps it "
                             "constant so the LR is not the variable.")
    parser.add_argument("--synth-augment", choices=("none", "light", "heavy"),
                        default="light",
                        help="Augmentation for the synthetic stage. 'light' is only "
                             "brightness, which lets the model memorise the single "
                             "rendering instead of learning the glyphs.")
    parser.add_argument("--clip-grad", type=float, default=1.5,
                        help="Gradient-norm clip. 1.5 was blocking CTC convergence; "
                             "raise it to see whether the plateau moves.")
    parser.add_argument("--lr-real", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--loader-workers", type=int, default=8,
                        help="Decoding workers. Zero makes the GPU wait on a single "
                             "core to decode 300k JPEGs, which is most of the epoch.")
    parser.add_argument("--save-dir", type=Path, default=Path("/tmp/opencode/ocr_checkpoints"))
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
        optimizer, mode="min", factor=0.5, patience=patience)


def current_lr(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_grad: float = 1.5):
    model.train()
    total_loss = 0.0
    for imgs, targets, target_lengths, _ in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        logits = model(imgs)  # (W', B, num_classes)
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
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    print(f"[ocr-train] Using device: {device}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_synth = args.save_dir / "crnn_stage1_synth.pt"
    checkpoint_final = args.save_dir / "crnn_stage2_real.pt"

    model = ResNetCRNN().to(device)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    test_ds = PlateOcrDataset(root, test_stems, augment_mode="none")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=ctc_collate_fn,
        num_workers=args.loader_workers, pin_memory=pin_memory,
    )

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
        synth_train_stems = (
            train_synth_stems[:-val_size] if val_size else train_synth_stems)

        # Synth images get light augmentation to avoid perfect-pixel memorization
        synth_ds = PlateOcrDataset(root, synth_train_stems,
                                   augment_mode=args.synth_augment)
        synth_loader = DataLoader(
            synth_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ctc_collate_fn,
            num_workers=args.loader_workers, pin_memory=pin_memory,
        )
        synth_val_loader = None
        if synth_val_stems:
            synth_val_ds = PlateOcrDataset(root, synth_val_stems, augment_mode="none")
            synth_val_loader = DataLoader(
                synth_val_ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=ctc_collate_fn,
                num_workers=args.loader_workers, pin_memory=pin_memory,
            )

        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr_synth)
        # Slower LR decay to give CTC time to align characters
        scheduler = build_scheduler(optimizer, args.lr_schedule, patience=5)

        for epoch in range(1, args.epochs_synth + 1):
            loss = train_one_epoch(model, synth_loader, optimizer, criterion, device,
                                   clip_grad=args.clip_grad)
            if scheduler is not None:
                scheduler.step(loss)
            # evaluate() returns both the raw and the heuristic decoding; stage 1
            # printed it as a single object and crashed on the first epoch.
            scores_raw, _ = evaluate(model, test_loader, device)
            synth_scores = (
                evaluate(model, synth_val_loader, device)[0]
                if synth_val_loader is not None else None)
            synth_part = ""
            if synth_scores is not None:
                synth_part = (
                    f"SynthVal Acc: {synth_scores.char_accuracy * 100:.1f}% "
                    f"(EM:{synth_scores.exact_match * 100:.1f}%) | ")
            print(
                f"Epoch {epoch:02d}/{args.epochs_synth:02d} | Synth Loss: {loss:.4f} "
                f"| LR {current_lr(optimizer):.2e} | "
                f"{synth_part}"
                f"RealTest Acc: {scores_raw.char_accuracy * 100:.1f}% "
                f"(EM:{scores_raw.exact_match * 100:.1f}%)",
                flush=True,
            )

        torch.save(model.state_dict(), checkpoint_synth)
        print(f"[ocr-train] Saved Stage 1 model to {checkpoint_synth}")

    # --- STAGE 2: Real Fine-Tuning ---
    if args.stage in ["stage2_real", "full_pipeline"] and train_real_stems:
        print("\n=== STARTING STAGE 2: Real Fine-Tuning ===")
        if checkpoint_synth.exists() and args.stage == "stage2_real":
            model.load_state_dict(torch.load(checkpoint_synth, map_location=device))
            print(f"[ocr-train] Loaded Stage 1 weights from {checkpoint_synth}")

        set_stage_freezing(model, "stage2_real")
        # Real images need heavy augmentation to generalize
        real_ds = PlateOcrDataset(root, train_real_stems, augment_mode="heavy")
        real_loader = DataLoader(
            real_ds, batch_size=args.batch_size, shuffle=True, collate_fn=ctc_collate_fn,
            num_workers=args.loader_workers, pin_memory=pin_memory,
        )

        # Higher LR for Stage 2 adaptation (2e-4) with scheduler
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr_real)
        scheduler = build_scheduler(optimizer, args.lr_schedule, patience=4)

        for epoch in range(1, args.epochs_real + 1):
            loss = train_one_epoch(model, real_loader, optimizer, criterion, device,
                                   clip_grad=args.clip_grad)
            if scheduler is not None:
                scheduler.step(loss)
            scores_raw, scores_heur = evaluate(model, test_loader, device)
            print(
                f"Epoch {epoch:02d}/{args.epochs_real:02d} | Real Loss: {loss:.4f} "
                f"| LR {current_lr(optimizer):.2e} | "
                f"Raw Acc: {scores_raw.char_accuracy * 100:.1f}% (EM:{scores_raw.exact_match * 100:.1f}%) | "
                f"Heur Acc: {scores_heur.char_accuracy * 100:.1f}% "
                f"(EM:{scores_heur.exact_match * 100:.1f}%)",
                flush=True,
            )

        torch.save(model.state_dict(), checkpoint_final)
        print(f"[ocr-train] Saved Stage 2 model to {checkpoint_final}")

        final_raw, final_heur = evaluate(model, test_loader, device)
        metrics_file = args.save_dir / "final_metrics.json"
        metrics_file.write_text(
            json.dumps(
                {
                    "n_test": final_raw.n,
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
