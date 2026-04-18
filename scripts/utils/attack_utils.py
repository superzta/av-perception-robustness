from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def apply_camera_attack(image_bgr: np.ndarray, attack_cfg: Dict) -> Tuple[np.ndarray, Dict]:
    if not attack_cfg.get("enabled", False):
        return image_bgr.copy(), {"camera_attack_enabled": False, "camera_attack_type": "none"}

    attack_type = str(attack_cfg.get("type", "glare")).lower()
    if attack_type == "patch":
        attacked = _apply_patch_attack(image_bgr, attack_cfg)
    else:
        attacked = _apply_glare_attack(image_bgr, attack_cfg)
        attack_type = "glare"

    meta = {
        "camera_attack_enabled": True,
        "camera_attack_type": attack_type,
        "camera_attack_strength": float(attack_cfg.get("strength", 1.0)),
    }
    return attacked, meta


def _apply_glare_attack(image_bgr: np.ndarray, attack_cfg: Dict) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    cx = float(attack_cfg.get("center_x_ratio", 0.65)) * w
    cy = float(attack_cfg.get("center_y_ratio", 0.3)) * h
    radius = max(5.0, float(attack_cfg.get("radius_ratio", 0.22)) * min(h, w))
    glare_strength = float(attack_cfg.get("glare_strength", 1.8))

    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mask = np.clip(1.0 - (dist / radius), 0.0, 1.0)
    boost = 1.0 + glare_strength * mask
    boost = np.repeat(boost[:, :, None], 3, axis=2)

    attacked = image_bgr.astype(np.float32) * boost
    attacked = np.clip(attacked, 0, 255).astype(np.uint8)
    return attacked


def _apply_patch_attack(image_bgr: np.ndarray, attack_cfg: Dict) -> np.ndarray:
    attacked = image_bgr.copy()
    h, w = attacked.shape[:2]
    px = float(attack_cfg.get("x_ratio", 0.45))
    py = float(attack_cfg.get("y_ratio", 0.15))
    pw = float(attack_cfg.get("w_ratio", 0.12))
    ph = float(attack_cfg.get("h_ratio", 0.18))
    x1 = max(0, min(w - 1, int(px * w)))
    y1 = max(0, min(h - 1, int(py * h)))
    x2 = max(x1 + 2, min(w, int((px + pw) * w)))
    y2 = max(y1 + 2, min(h, int((py + ph) * h)))
    color = attack_cfg.get("color_bgr", [0, 0, 255])
    attacked[y1:y2, x1:x2] = np.array(color, dtype=np.uint8)
    return attacked


def apply_lidar_attack(points_xyzi: np.ndarray, attack_cfg: Dict, frame_id: int) -> Tuple[np.ndarray, Dict]:
    if not attack_cfg.get("enabled", False):
        return points_xyzi.copy(), {"lidar_attack_enabled": False, "lidar_attack_type": "none", "injected_points": 0}

    attack_type = str(attack_cfg.get("type", "phantom_cluster")).lower()
    if attack_type != "phantom_cluster":
        attack_type = "phantom_cluster"

    num_points = int(attack_cfg.get("num_points", 220))
    x = float(attack_cfg.get("x_m", 10.0))
    y = float(attack_cfg.get("y_m", 0.0))
    z = float(attack_cfg.get("z_m", -1.0))
    std = float(attack_cfg.get("std_m", 0.5))
    intensity = float(attack_cfg.get("intensity", 0.8))

    rng = np.random.default_rng(seed=int(frame_id) + int(attack_cfg.get("seed_offset", 12345)))
    cluster_xyz = rng.normal(loc=[x, y, z], scale=[std, std, max(0.15, std * 0.3)], size=(num_points, 3))
    cluster_i = np.full((num_points, 1), intensity, dtype=np.float32)
    injected = np.concatenate([cluster_xyz.astype(np.float32), cluster_i], axis=1)
    attacked = np.concatenate([points_xyzi.astype(np.float32), injected], axis=0)

    meta = {
        "lidar_attack_enabled": True,
        "lidar_attack_type": attack_type,
        "injected_points": int(num_points),
        "phantom_center_xyz": [x, y, z],
    }
    return attacked, meta


def save_lidar_bev(points_xyzi: np.ndarray, output_path: Path, x_lim=(-5.0, 30.0), y_lim=(-12.0, 12.0)) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 500
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    # Draw reference grid every 5 meters to make scene changes obvious.
    x_span = x_lim[1] - x_lim[0]
    y_span = y_lim[1] - y_lim[0]
    if x_span > 0 and y_span > 0:
        for x_m in range(int(np.ceil(x_lim[0])), int(np.floor(x_lim[1])) + 1):
            if x_m % 5 != 0:
                continue
            px = int((x_m - x_lim[0]) / x_span * (width - 1))
            if 0 <= px < width:
                canvas[:, px] = (38, 38, 38)
        for y_m in range(int(np.ceil(y_lim[0])), int(np.floor(y_lim[1])) + 1):
            if y_m % 5 != 0:
                continue
            py = int((1.0 - (y_m - y_lim[0]) / y_span) * (height - 1))
            if 0 <= py < height:
                canvas[py, :] = (38, 38, 38)

    x = points_xyzi[:, 0]
    y = points_xyzi[:, 1]
    z = points_xyzi[:, 2]
    in_range = (x >= x_lim[0]) & (x <= x_lim[1]) & (y >= y_lim[0]) & (y <= y_lim[1])
    x = x[in_range]
    y = y[in_range]
    z = z[in_range]
    if x.size > 0:
        px = ((x - x_lim[0]) / (x_lim[1] - x_lim[0]) * (width - 1)).astype(int)
        py = ((1.0 - (y - y_lim[0]) / (y_lim[1] - y_lim[0])) * (height - 1)).astype(int)
        # Color by distance in front of ego: near=yellow, far=blue.
        dist = np.sqrt(x**2 + y**2)
        dist_norm = np.clip(dist / max(1.0, x_lim[1]), 0.0, 1.0)
        b = (255.0 * dist_norm).astype(np.uint8)
        g = (220.0 * (1.0 - dist_norm)).astype(np.uint8)
        r = (255.0 * (1.0 - dist_norm)).astype(np.uint8)
        canvas[py, px] = np.stack([b, g, r], axis=1)

        # Highlight near-ground obstacle points in red for quick sanity checks.
        obstacle_mask = (x > 0.5) & (x < 25.0) & (np.abs(y) < 3.0) & (z > -2.3) & (z < 0.8)
        if np.any(obstacle_mask):
            canvas[py[obstacle_mask], px[obstacle_mask]] = (40, 40, 255)

    ego_x = int((0.0 - x_lim[0]) / (x_lim[1] - x_lim[0]) * (width - 1))
    ego_y = int((1.0 - (0.0 - y_lim[0]) / (y_lim[1] - y_lim[0])) * (height - 1))
    if 0 <= ego_x < width and 0 <= ego_y < height:
        canvas[max(0, ego_y - 4) : min(height, ego_y + 5), max(0, ego_x - 4) : min(width, ego_x + 5)] = (
            0,
            255,
            0,
        )

    try:
        import cv2

        cv2.imwrite(str(output_path), canvas)
    except ImportError:
        from PIL import Image

        Image.fromarray(canvas[:, :, ::-1]).save(output_path)
