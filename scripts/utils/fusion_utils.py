import math
from typing import Dict, List, Tuple

import numpy as np


def get_synced_rgb_lidar(world, rgb_queue, lidar_queue, timeout_seconds: float = 2.0):
    target_frame = world.tick()
    rgb = _get_first_at_or_after(rgb_queue, target_frame, timeout_seconds)
    lidar = _get_first_at_or_after(lidar_queue, target_frame, timeout_seconds)

    while rgb.frame != lidar.frame:
        common = max(rgb.frame, lidar.frame)
        if rgb.frame < common:
            rgb = _get_first_at_or_after(rgb_queue, common, timeout_seconds)
        if lidar.frame < common:
            lidar = _get_first_at_or_after(lidar_queue, common, timeout_seconds)
    return rgb, lidar


def _get_first_at_or_after(data_queue, min_frame: int, timeout_seconds: float):
    while True:
        data = data_queue.get(timeout=timeout_seconds)
        if data.frame >= min_frame:
            return data


def build_camera_intrinsics(width: int, height: int, fov_degrees: float) -> np.ndarray:
    focal = width / (2.0 * math.tan(math.radians(fov_degrees / 2.0)))
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = focal
    intrinsics[1, 1] = focal
    intrinsics[0, 2] = width / 2.0
    intrinsics[1, 2] = height / 2.0
    return intrinsics


def project_lidar_to_image(
    lidar_points_xyzi: np.ndarray,
    camera_actor,
    lidar_actor,
    camera_intrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if lidar_points_xyzi.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    lidar_xyz = lidar_points_xyzi[:, :3]
    ones = np.ones((lidar_xyz.shape[0], 1), dtype=np.float32)
    lidar_h = np.concatenate([lidar_xyz, ones], axis=1)

    lidar_to_world = np.array(lidar_actor.get_transform().get_matrix(), dtype=np.float32)
    world_to_camera = np.array(camera_actor.get_transform().get_inverse_matrix(), dtype=np.float32)

    world_points = (lidar_to_world @ lidar_h.T)
    camera_points = (world_to_camera @ world_points).T[:, :3]

    # CARLA -> conventional camera coordinates.
    points = np.stack(
        [
            camera_points[:, 1],
            -camera_points[:, 2],
            camera_points[:, 0],
        ],
        axis=1,
    )
    depths = points[:, 2]
    valid_depth = depths > 0.1

    points = points[valid_depth]
    depths = depths[valid_depth]
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    u = (camera_intrinsics[0, 0] * points[:, 0] / depths) + camera_intrinsics[0, 2]
    v = (camera_intrinsics[1, 1] * points[:, 1] / depths) + camera_intrinsics[1, 2]
    uv = np.stack([u, v], axis=1).astype(np.float32)
    return uv, depths.astype(np.float32)


def apply_late_fusion(
    detections: List[dict],
    projected_uv: np.ndarray,
    projected_depths: np.ndarray,
    fusion_cfg: Dict,
) -> List[dict]:
    min_points = int(fusion_cfg.get("min_lidar_points_in_bbox", 8))
    min_depth = float(fusion_cfg.get("min_depth_m", 0.5))
    max_depth = float(fusion_cfg.get("max_depth_m", 60.0))
    unconfirmed_penalty = float(fusion_cfg.get("unconfirmed_confidence_penalty", 0.6))

    fused = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        mask = (
            (projected_uv[:, 0] >= x1)
            & (projected_uv[:, 0] <= x2)
            & (projected_uv[:, 1] >= y1)
            & (projected_uv[:, 1] <= y2)
            & (projected_depths >= min_depth)
            & (projected_depths <= max_depth)
        )
        lidar_points_in_bbox = int(np.count_nonzero(mask))
        lidar_confirmed = lidar_points_in_bbox >= min_points
        fused_confidence = float(det["confidence"] if lidar_confirmed else det["confidence"] * unconfirmed_penalty)

        fused_det = dict(det)
        fused_det["lidar_points_in_bbox"] = lidar_points_in_bbox
        fused_det["lidar_confirmed"] = lidar_confirmed
        fused_det["fused_confidence"] = fused_confidence
        fused.append(fused_det)

    return fused


def draw_fusion_detections(image_bgr: np.ndarray, fused_detections: List[dict], header_text: str = "") -> np.ndarray:
    annotated = image_bgr.copy()
    try:
        import cv2

        for det in fused_detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            if det.get("lidar_confirmed", False):
                color = (0, 220, 0)
                status = "CONF"
            else:
                color = (0, 165, 255)
                status = "RGB_ONLY"

            label = (
                f"{det['class_name']} y={det['confidence']:.2f} "
                f"f={det.get('fused_confidence', det['confidence']):.2f} "
                f"lp={det.get('lidar_points_in_bbox', 0)} {status}"
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        if header_text:
            cv2.putText(
                annotated,
                header_text,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated
    except ImportError:
        from PIL import Image, ImageDraw

        image_rgb = annotated[:, :, ::-1]
        pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil)
        for det in fused_detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            if det.get("lidar_confirmed", False):
                color = (0, 220, 0)
                status = "CONF"
            else:
                color = (255, 165, 0)
                status = "RGB_ONLY"
            label = (
                f"{det['class_name']} y={det['confidence']:.2f} "
                f"f={det.get('fused_confidence', det['confidence']):.2f} "
                f"lp={det.get('lidar_points_in_bbox', 0)} {status}"
            )
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            draw.text((x1, max(0, y1 - 12)), label, fill=color)
        if header_text:
            draw.text((10, 10), header_text, fill=(255, 255, 255))
        rgb_array = np.array(pil)
        return rgb_array[:, :, ::-1]
