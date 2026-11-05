# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
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
    dst = np.float32(
        [
            [rng.uniform(0, dx), rng.uniform(0, dy)],
            [w - rng.uniform(0, dx), rng.uniform(0, dy)],
            [w - rng.uniform(0, dx * 0.6), h - rng.uniform(0, dy)],
            [rng.uniform(0, dx), h - rng.uniform(0, dy)],
        ]
    )
    return cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE
    )


def _exposure(image: np.ndarray, rng: random.Random, night: bool) -> np.ndarray:
    """Gamma + gain, biased dark: the gate reads gray, never studio white.

    Real crops cluster dark (night, booth shadow, overcast): even the day
    branch stays under full exposure so the sheeting lands gray.
    """
    img = image.astype(np.float32) / 255.0
    if night:
        img *= rng.uniform(0.50, 0.66)
        gamma = rng.uniform(1.3, 1.8)
    else:
        img *= rng.uniform(0.65, 1.0)
        gamma = rng.uniform(0.9, 1.4)
    img = np.clip(img, 0.0, 1.0) ** gamma
    # Flatten and darken toward the real capture. Without this the synthetic
    # stays ~45% more contrasty and ~35% brighter than the served crops even
    # after the runtime CLAHE, which is a distribution the model never meets.
    flat = rng.uniform(0.62, 0.86)
    img = (img - img.mean()) * flat + img.mean()
    img *= rng.uniform(0.62, 0.80)
    # White balance: the recorder and the booth lighting make every real crop
    # warm (measured R/B = 1.41 against 0.96 for a neutral render). Grey-level
    # statistics cannot see this, but the CRNN consumes RGB.
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * rng.uniform(0.45, 0.85)  # desaturate: ink reads near-black
    # White balance goes AFTER the desaturation on purpose: desaturating pulls
    # every channel toward grey, which cancelled most of the warmth when it ran
    # last. Real crops are strongly warm (R/B 1.55 against 0.96 neutral).
    img[:, :, 2] *= rng.uniform(1.30, 1.55)
    img[:, :, 0] *= rng.uniform(0.82, 0.94)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _shadow(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """One soft diagonal shadow, like the booth casts across the lane."""
    if rng.random() < 0.4:
        return image
    h, w = image.shape[:2]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    mask = (x * rng.uniform(0.5, 1.0) + rng.uniform(0.0, 0.5)).clip(0, 1)
    strength = rng.uniform(0.30, 0.55)
    shadow = (1.0 - strength * mask)[:, :, None]
    return (image.astype(np.float32) * shadow).astype(np.uint8)


def _glare(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Specular streak on the reflective sheeting; rare but blinding."""
    if rng.random() < 0.75:
        return image
    h, w = image.shape[:2]
    overlay = np.zeros_like(image)
    cx, cy = rng.randint(0, w), rng.randint(0, h)
    cv2.ellipse(
        overlay, (cx, cy), (w // 3, h // 2), rng.uniform(0, 180), 0, 360, (255, 255, 255), -1
    )
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


def _wear(image: np.ndarray, rng: random.Random, strength: float = 0.22) -> np.ndarray:
    """Light scratches and dirt, blended at low alpha.

    Kept deliberately faint (``strength`` ~0.2): a plate on the street is dirty,
    not destroyed. The point is to break the perfectly uniform sheeting that no
    capture ever produces, without inventing damage the model would learn to
    expect. The blend is what keeps it "almost intact" instead of "ruined".
    """
    if rng.random() < 0.30:
        return image
    h, w = image.shape[:2]
    damage = image.copy()
    for _ in range(rng.randint(1, 2)):  # scratches
        x0, y0 = rng.randint(0, w), rng.randint(0, h)
        x1 = x0 + rng.randint(-w // 2, w // 2)
        y1 = y0 + rng.randint(-h // 4, h // 4)
        colour = rng.choice([(250, 250, 250), (40, 40, 40), (110, 110, 110)])
        cv2.line(damage, (x0, y0), (x1, y1), colour, rng.randint(1, 2))
    for _ in range(rng.randint(2, 8)):  # specks of dirt
        centre = (rng.randint(0, w), rng.randint(0, h))
        cv2.circle(
            damage,
            centre,
            rng.randint(1, max(2, w // 60)),
            rng.choice([(60, 60, 60), (200, 200, 200)]),
            -1,
        )
    if rng.random() < 0.5:  # a soft smudge, like a thumbprint
        overlay = np.zeros_like(image)
        cv2.ellipse(
            overlay,
            (rng.randint(0, w), rng.randint(0, h)),
            (max(3, w // 12), max(2, h // 8)),
            rng.uniform(0, 180),
            0,
            360,
            (30, 30, 30),
            -1,
        )
        overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=max(1.0, w / 60.0))
        damage = cv2.addWeighted(damage, 1.0, overlay, 0.35, 0)
    return cv2.addWeighted(damage, strength, image, 1.0 - strength, 0)


def _lens_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Optical softness at the final resolution.

    Measured on the served crops, real plates are ~5x less sharp than a clean
    downscale, and no amount of CLAHE brings that texture back: CLAHE stretches
    local range, it cannot invent the missing blur. Sigma scales with the crop
    width so a 60 px plate and a 150 px plate look equally soft.
    """
    height, width = image.shape[:2]
    sigma = rng.uniform(0.42, 0.95) * (width / 100.0)
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma * rng.uniform(0.8, 1.2))


def _to_target_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Downsample to the real-world size. This step IS the pixelation."""
    return cv2.resize(image, (max(8, width), max(6, height)), interpolation=cv2.INTER_AREA)


def _sensor_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    std = rng.uniform(1.5, 6.0)
    noise = (
        np.random.default_rng(rng.randint(0, 2**31 - 1))
        .normal(0, std, image.shape)
        .astype(np.float32)
    )
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Recorder-style recompression: blocky, always colour."""
    quality = rng.randint(32, 78)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    return cv2.imdecode(np.frombuffer(buf.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)


def degrade(
    master_bgr: np.ndarray,
    rng: random.Random,
    width: int,
    height: int,
    night_prob: float = 0.35,
    wear_strength: float = 0.22,
) -> tuple[np.ndarray, dict]:
    """Run the full pipeline; returns the final crop and the params used."""
    night = rng.random() < night_prob
    img = _perspective(master_bgr, rng, amount=rng.uniform(0.02, 0.10))
    img = _wear(img, rng, wear_strength)
    img = _exposure(img, rng, night)
    img = _shadow(img, rng)
    img = _glare(img, rng)
    img = _to_target_size(img, width, height)
    img = _lens_blur(img, rng)
    img = _motion_blur(img, rng)
    img = _sensor_noise(img, rng)
    img = _jpeg(img, rng)
    return img, {
        "night": night,
        "wear": wear_strength,
        "target_hw": [int(img.shape[0]), int(img.shape[1])],
    }
