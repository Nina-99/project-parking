# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Contract tests for the OCR crop export and the model-view preview.

These guard the two properties that make the framing ablation meaningful: the
crop convention is a pure function of the verified box (reproducible, never a
per-crop human choice), and the preview shows the real 192x48 geometry after
CLAHE. A preview with the wrong height is worse than none -- the reviewer would
approve crops the network sees differently.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from label_plate_box_gui import (
    EXPORT_VERSION,
    MODEL_H,
    MODEL_W,
    build_parser,
    crop_variants,
    export_ocr,
    inset_box,
    model_view,
    pad_box,
)


def _image(width: int = 200, height: int = 120) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


class TestModelView:
    def test_has_the_network_input_geometry(self) -> None:
        view = model_view(_image())
        assert view.shape == (MODEL_H, MODEL_W, 3)
        assert view.dtype == np.uint8

    def test_raises_contrast_the_way_the_runtime_does(self) -> None:
        flat = np.full((40, 120, 3), 90, dtype=np.uint8)
        flat[10:30, 20:100] = 40
        plain = cv2.resize(flat, (MODEL_W, MODEL_H), interpolation=cv2.INTER_CUBIC)
        enhanced = model_view(flat)
        assert np.std(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)) > np.std(
            cv2.cvtColor(plain, cv2.COLOR_BGR2GRAY)
        )

    def test_empty_input_does_not_crash(self) -> None:
        assert model_view(np.empty((0, 0, 3), dtype=np.uint8)).shape == (MODEL_H, MODEL_W, 3)


class TestBoxMath:
    def test_pad_matches_the_runtime_crop_plate(self) -> None:
        # crop_plate pads 4% in x and 8% in y; the export must not drift from it
        # or the training crops and the served crops stop describing each other.
        assert pad_box((100, 100, 200, 150), 0.04, 0.08) == (96, 96, 204, 154)

    def test_inset_is_symmetric_and_never_inverts(self) -> None:
        left, top, right, bottom = inset_box((100, 100, 200, 150), 0.10, 0.22)
        assert left - 100 == 200 - right
        assert top - 100 == 150 - bottom
        assert right > left and bottom > top

    def test_inset_clamps_absurd_fractions(self) -> None:
        x1, y1, x2, y2 = inset_box((0, 0, 40, 40), 0.9, 0.9)
        assert x2 - x1 >= 2 and y2 - y1 >= 2


class TestCropVariants:
    def test_box_variant_is_the_padded_crop(self) -> None:
        image = _image()
        variants = crop_variants(image, (50, 40, 150, 80), conventions=("box",))
        crop = variants["box"]
        x1, y1, x2, y2 = pad_box((50, 40, 150, 80), 0.04, 0.08)
        assert np.array_equal(crop, image[y1:y2, x1:x2])

    def test_inset_variant_is_smaller_than_box(self) -> None:
        variants = crop_variants(_image(), (50, 40, 150, 80), conventions=("box", "inset"))
        assert variants["inset"].shape[0] < variants["box"].shape[0]
        assert variants["inset"].shape[1] < variants["box"].shape[1]

    def test_box_is_the_default_arm(self) -> None:
        # The served framing is the one to export by default; the inset arm is
        # opt-in because a trim larger than the whitespace amputates glyphs.
        assert set(crop_variants(_image(), (50, 40, 150, 80))) == {"box"}

    def test_degenerate_crops_are_dropped_not_written(self) -> None:
        variants = crop_variants(_image(), (10, 10, 22, 12))
        assert variants == {}

    def test_unknown_convention_is_an_error(self) -> None:
        try:
            crop_variants(_image(), (50, 40, 150, 80), conventions=("tight",))
        except ValueError as error:
            assert "tight" in str(error)
        else:  # pragma: no cover - the raise is the contract
            raise AssertionError("a typo in a convention must not pass silently")


class TestExport:
    SESSION = "C.OCCIDENTE_ch16_main_20260427090012_20260427100012"
    STEM = SESSION + "_track1866_f0048936_b00"

    def _write(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        image_path = tmp_path / "crop.jpg"
        cv2.imwrite(str(image_path), _image())
        verified = tmp_path / "verified.jsonl"
        rows = [
            {
                "stem": self.STEM,
                "image": str(image_path),
                "box": [50, 40, 150, 80],
                "source": "proposal",
            },
            {
                "stem": "no_text",
                "image": str(image_path),
                "box": [50, 40, 150, 80],
                "source": "drawn",
            },
            {"stem": "negative", "image": str(image_path), "box": None, "source": "negative"},
            {
                "stem": "gone",
                "image": str(tmp_path / "gone.jpg"),
                "box": [50, 40, 150, 80],
                "source": "proposal",
            },
            # A second review of the same stem must win: the file is append-only.
            {
                "stem": self.STEM,
                "image": str(image_path),
                "box": [60, 45, 140, 75],
                "source": "drawn",
            },
        ]
        verified.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        text = tmp_path / "text.jsonl"
        # "gone" is labelled on purpose: without a label it would be filtered
        # before the unreadable-image branch and that path would go untested.
        text.write_text(
            json.dumps({"stem": self.STEM, "text": "4852phd"})
            + "\n"
            + json.dumps({"stem": "gone", "text": "1234abc"})
            + "\n",
            encoding="utf-8",
        )
        return verified, text, tmp_path / "crops"

    def _args(self, verified: Path, text: Path, root: Path, conventions: str = "box") -> object:
        return build_parser().parse_args(
            [
                "--export-ocr",
                "--output",
                str(verified),
                "--crop-root",
                str(root),
                "--text",
                str(text),
                "--conventions",
                conventions,
            ]
        )

    def _manifest(self, root: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    def test_writes_the_layout_the_trainer_reads(self, tmp_path: Path) -> None:
        verified, text, root = self._write(tmp_path)
        written = export_ocr(self._args(verified, text, root))
        assert written == 1
        crop = root / "plates" / "images" / self.SESSION / f"{self.STEM}.jpg"
        label = root / "plates" / "labels" / self.SESSION / f"{self.STEM}.txt"
        assert crop.exists(), "PlateOcrDataset globs plates/images/**/*.jpg"
        assert label.read_text(encoding="utf-8").strip() == "4852PHD"
        rows = self._manifest(root)
        assert rows[0]["session"] == self.SESSION, (
            "the session directory is what prepare_ocr_ablation splits by"
        )
        assert rows[0]["convention"] == "box"
        assert rows[0]["version"] == EXPORT_VERSION

    def test_last_review_of_a_stem_wins(self, tmp_path: Path) -> None:
        verified, text, root = self._write(tmp_path)
        export_ocr(self._args(verified, text, root))
        assert self._manifest(root)[0]["box"] == [60, 45, 140, 75]

    def test_crops_without_text_are_left_out(self, tmp_path: Path, capsys: object) -> None:
        verified, text, root = self._write(tmp_path)
        export_ocr(self._args(verified, text, root))
        stems = {row["stem"] for row in self._manifest(root)}
        assert "no_text" not in stems, "an unlabelled crop teaches CTC to emit nothing"
        assert "have no text label" in capsys.readouterr().out

    def test_negatives_and_unreadable_images_are_counted(
        self, tmp_path: Path, capsys: object
    ) -> None:
        verified, text, root = self._write(tmp_path)
        export_ocr(self._args(verified, text, root))
        stems = {row["stem"] for row in self._manifest(root)}
        assert stems == {self.STEM}
        assert "1 rows skipped" in capsys.readouterr().out

    def test_refuses_two_conventions_in_one_root(self, tmp_path: Path) -> None:
        verified, text, root = self._write(tmp_path)
        try:
            export_ocr(self._args(verified, text, root, conventions="box,inset"))
        except SystemExit as error:
            assert "one convention per dataset root" in str(error)
        else:  # pragma: no cover - the refusal is the contract
            raise AssertionError("mixing framings behind the same stems must fail")

    def test_refuses_to_reuse_a_root_with_another_convention(self, tmp_path: Path) -> None:
        verified, text, root = self._write(tmp_path)
        export_ocr(self._args(verified, text, root))
        try:
            export_ocr(self._args(verified, text, root, conventions="inset"))
        except SystemExit as error:
            assert "already holds" in str(error)
        else:  # pragma: no cover
            raise AssertionError("two conventions in one root silently double samples")

    def test_a_failed_write_keeps_no_row(
        self, tmp_path: Path, monkeypatch: object, capsys: object
    ) -> None:
        # imwrite returns False on a full disk; the manifest must not advertise
        # a crop that is not there.
        verified, text, root = self._write(tmp_path)
        monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)
        written = export_ocr(self._args(verified, text, root))
        assert written == 0
        assert self._manifest(root) == []
        assert "could not be written" in capsys.readouterr().out

    def _malformed_case(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        image = tmp_path / "c.jpg"
        cv2.imwrite(str(image), _image())
        verified = tmp_path / "v.jsonl"
        verified.write_text(
            "".join(
                json.dumps(
                    {
                        "stem": "s" + str(i),
                        "image": str(image),
                        "box": [50, 40, 150, 80],
                        "source": "proposal",
                    }
                )
                + "\n"
                for i in range(2)
            ),
            encoding="utf-8",
        )
        text = tmp_path / "t.jsonl"
        text.write_text(
            json.dumps({"stem": "s0", "text": "4250L"})
            + "\n"
            + json.dumps({"stem": "s1", "text": "4852PHD"})
            + "\n",
            encoding="utf-8",
        )
        return verified, text, tmp_path / "crops"

    def test_malformed_labels_never_reach_the_trainer(self, tmp_path: Path, capsys: object) -> None:
        verified, text, root = self._malformed_case(tmp_path)
        written = export_ocr(self._args(verified, text, root))
        assert written == 1
        assert [r["text"] for r in self._manifest(root)] == ["4852PHD"]
        out = capsys.readouterr().out
        assert "not NNNLLL / NNNNLLL" in out and "4250L" in out

    def test_allow_nonconforming_keeps_them(self, tmp_path: Path) -> None:
        verified, text, root = self._malformed_case(tmp_path)
        args = build_parser().parse_args(
            [
                "--export-ocr",
                "--output",
                str(verified),
                "--crop-root",
                str(root),
                "--text",
                str(text),
                "--allow-nonconforming",
            ]
        )
        assert export_ocr(args) == 2

    def test_stale_crops_are_pruned_when_a_label_becomes_invalid(self, tmp_path: Path) -> None:
        # The trainer globs the directory. A crop whose label later fails
        # validation must not survive as an orphan next to valid samples.
        verified, text, root = self._malformed_case(tmp_path)
        permissive = build_parser().parse_args(
            [
                "--export-ocr",
                "--output",
                str(verified),
                "--crop-root",
                str(root),
                "--text",
                str(text),
                "--allow-nonconforming",
            ]
        )
        export_ocr(permissive)
        good = root / "plates" / "images" / "s1" / "s1.jpg"
        bad = root / "plates" / "images" / "s0" / "s0.jpg"
        assert good.exists() and bad.exists()
        export_ocr(self._args(verified, text, root))
        assert good.exists()
        assert not bad.exists(), "an orphaned crop defeats the label validation"
        assert not (root / "plates" / "labels" / "s0" / "s0.txt").exists()

    def test_never_prunes_files_from_another_producer(self, tmp_path: Path) -> None:
        # Crops copied in from another dataset root (and the synthetic set) are
        # not this tool's to delete: pruning is scoped to the manifest's rows.
        verified, text, root = self._write(tmp_path)
        foreign = root / "plates" / "images" / "S1__tight" / "other.jpg"
        foreign.parent.mkdir(parents=True)
        cv2.imwrite(str(foreign), _image())
        export_ocr(self._args(verified, text, root))
        assert foreign.exists()

    def test_never_prunes_the_synthetic_set(self, tmp_path: Path) -> None:
        # Synth shares the root (the trainer needs one) but belongs to the
        # generator; deleting it would wipe stage 1 on the next export.
        verified, text, root = self._write(tmp_path)
        synth_img = root / "plates" / "images" / "synth" / "synth_000001.jpg"
        synth_lbl = root / "plates" / "labels" / "synth" / "synth_000001.txt"
        synth_img.parent.mkdir(parents=True)
        synth_lbl.parent.mkdir(parents=True)
        cv2.imwrite(str(synth_img), _image())
        synth_lbl.write_text("4852PHD\n", encoding="utf-8")
        export_ocr(self._args(verified, text, root))
        assert synth_img.exists() and synth_lbl.exists()

    def test_rerun_is_idempotent_and_rewrites_the_manifest(self, tmp_path: Path) -> None:
        verified, text, root = self._write(tmp_path)
        export_ocr(self._args(verified, text, root))
        crop = root / "plates" / "images" / self.SESSION / f"{self.STEM}.jpg"
        before = crop.stat().st_mtime_ns
        export_ocr(self._args(verified, text, root))
        assert crop.stat().st_mtime_ns == before
        assert len(self._manifest(root)) == 1, "the manifest must describe the disk, not accumulate"
