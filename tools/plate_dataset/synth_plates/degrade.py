# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Physical degradation pipeline: geometry, light, sensor — in that order.

The order is the physics of the capture: the plate has a pose (perspective +
scale for distance), the light hits it (exposure, shadow, glare, night), and
only then does the sensor turn it into pixels (downsample, blur, noise, JPEG).
Shuffling the order produces artefacts no camera ever makes, e.g. glare that
survives pixelation instead of being pixelated with everything else.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


def _perspective(image: np.ndarray, rng: random.Random, amount: float) -> np.ndarray:
    """Tilt the master as the fixed gate camera sees it: mostly yaw."""
    h, w = image.shape[:2]
    dx, dy = w * amount, h * amount
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.uniform(0, dx), rng.uniform(0, dy)],
        [w - rng.uniform(0, dx), rng.uniform(0, dy)],
        [w - rng.uniform(0, dx * 0.6), h - rng.uniform(0, dy)],
        [rng.uniform(0, dx), h - rng.uniform(0, dy)],
    ])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderMode=cv2.BORDER_REPLICATE)


def _exposure(image: np.ndarray, rng: random.Random, night: bool) -> np.ndarray:
    """Gamma + gain, biased dark: the gate reads gray, never studio white.

    Real crops cluster dark (night, booth shadow, overcast): even the day
    branch stays under full exposure so the sheeting lands gray.
    """
    img = image.astype(np.float32) / 255.0
    if night:
        img *= rng.uniform(0.30, 0.55)
        img[:, :, 2] *= rng.uniform(1.0, 1.25)  # sodium lamps run warm
        gamma = rng.uniform(1.3, 1.8)
    else:
        img *= rng.uniform(0.65, 1.0)
        gamma = rng.uniform(0.9, 1.4)
    img = np.clip(img, 0.0, 1.0) ** gamma
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * rng.uniform(0.45, 0.85)  # desaturate: ink reads near-black
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _shadow(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """One soft diagonal shadow, like the booth casts across the lane."""
    if rng.random() < 0.4:
        return image
    h, w = image.shape[:2]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    mask = (x * rng.uniform(0.5, 1.0) + rng.uniform(0.0, 0.5)).clip(0, 1)
    strength = rng.uniform(0.35, 0.65)
    shadow = (1.0 - strength * mask)[:, :, None]
    return (image.astype(np.float32) * shadow).astype(np.uint8)


def _glare(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Specular streak on the reflective sheeting; rare but blinding."""
    if rng.random() < 0.75:
        return image
    h, w = image.shape[:2]
    overlay = np.zeros_like(image)
    cx, cy = rng.randint(0, w), rng.randint(0, h)
    cv2.ellipse(overlay, (cx, cy), (w // 3, h // 2),
                rng.uniform(0, 180), 0, 360, (255, 255, 255), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=w / 8.0)
    alpha = rng.uniform(0.25, 0.55)
    return cv2.addWeighted(image, 1.0, overlay, alpha, 0)


def _motion_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Directional blur along the lane: the car moves, the camera does not."""
    if rng.random() < 0.5:
        return image
    length = rng.randint(3, 9)
    kernel = np.zeros((length, length), dtype=np.float32)
    if rng.random() < 0.7:
        kernel[length // 2, :] = 1.0  # horizontal drive-by
    else:
        kernel = cv2.getRotationMatrix2D((length / 2, length / 2), rng.uniform(-25, 25), 1.0)
        base = np.zeros((length, length), dtype=np.float32)
        base[length // 2, :] = 1.0
        kernel = cv2.warpAffine(base, kernel, (length, length))
        kernel /= kernel.sum() + 1e-9
        return cv2.filter2D(image, -1, kernel)
    kernel /= float(length)
    return cv2.filter2D(image, -1, kernel)


def _to_target_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Downsample to the real-world size. This step IS the pixelation."""
    return cv2.resize(image, (max(8, width), max(6, height)), interpolation=cv2.INTER_AREA)


def _sensor_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    std = rng.uniform(1.5, 6.0)
    noise = np.random.default_rng(rng.randint(0, 2 ** 31 - 1)).normal(
        0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Recorder-style recompression: blocky, always colour."""
    quality = rng.randint(55, 90)
    ok, buf = cv2.imencode(".jpg", image,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(np.frombuffer(buf.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)


def degrade(master_bgr: np.ndarray, rng: random.Random,
            width: int, height: int, night_prob: float = 0.35) -> tuple[np.ndarray, dict]:
    """Run the full pipeline; returns the final crop and the params used."""
    night = rng.random() < night_prob
    img = _perspective(master_bgr, rng, amount=rng.uniform(0.02, 0.10))
    img = _exposure(img, rng, night)
    img = _shadow(img, rng)
    img = _glare(img, rng)
    img = _to_target_size(img, width, height)
    img = _motion_blur(img, rng)
    img = _sensor_noise(img, rng)
    img = _jpeg(img, rng)
    return img, {"night": night, "target_hw": [int(img.shape[0]), int(img.shape[1])]}
