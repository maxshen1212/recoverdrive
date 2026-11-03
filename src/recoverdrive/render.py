"""Images for the VLM: a labeled mosaic of all cameras and a LiDAR bird's-eye view."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

MOSAIC_ORDER = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


def camera_mosaic(images: dict[str, np.ndarray], order: list[str] | None = None,
                  tile_wh: tuple[int, int] = (480, 270), cols: int = 3) -> np.ndarray:
    """Grid of downscaled camera images, each with its name in a black bar at the top-left."""
    order = [c for c in (order or MOSAIC_ORDER) if c in images]
    rows = max(1, -(-len(order) // cols))
    tw, th = tile_wh
    canvas = Image.new("RGB", (tw * cols, th * rows))
    draw = ImageDraw.Draw(canvas)
    for i, cam in enumerate(order):
        x, y = (i % cols) * tw, (i // cols) * th
        canvas.paste(Image.fromarray(images[cam]).resize((tw, th), Image.BILINEAR), (x, y))
        draw.rectangle([x, y, x + 12 * len(cam) + 10, y + 24], fill=(0, 0, 0))
        draw.text((x + 5, y + 3), cam, fill=(255, 255, 0), font_size=18)
    return np.asarray(canvas)


def lidar_bev(points: np.ndarray, extent_m: float = 50.0, size_px: int = 448) -> np.ndarray:
    """Top-down LiDAR image: ego at the centre facing up, `extent_m` to each edge, colour = height."""
    img = np.zeros((size_px, size_px, 3), np.uint8)
    pts = np.asarray(points, float).reshape(-1, points.shape[1] if points.ndim == 2 and len(points) else 4)
    if len(pts):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        keep = (np.abs(x) < extent_m) & (np.abs(y) < extent_m)
        x, y, z = x[keep], y[keep], z[keep]
        row = ((extent_m - x) / (2 * extent_m) * (size_px - 1)).astype(int)
        col = ((extent_m - y) / (2 * extent_m) * (size_px - 1)).astype(int)
        t = np.clip((z + 2.0) / 4.0, 0, 1)  # -2 m .. +2 m relative to the sensor
        color = np.stack([255 * t, 255 * (1 - np.abs(t - 0.5) * 2) * 0.8 + 50, 255 * (1 - t)], axis=1)
        img[row, col] = color.astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    c = size_px / 2
    r25 = size_px / 4
    draw.ellipse([c - r25, c - r25, c + r25, c + r25], outline=(90, 90, 90))
    draw.polygon([(c, c - 8), (c - 5, c + 6), (c + 5, c + 6)], fill=(255, 255, 255))
    draw.text((5, 3), "LiDAR top view, ego at centre facing up, ring = 25 m", fill=(200, 200, 200), font_size=14)
    return np.asarray(pil)
