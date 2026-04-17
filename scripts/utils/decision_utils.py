from collections import Counter
from typing import Dict, List

import numpy as np


CRITICAL_CLASSES = {"person", "stop sign"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def detection_changed(clean_dets: List[dict], attacked_dets: List[dict], conf_delta_threshold: float = 0.08) -> bool:
    clean_classes = Counter(det["class_name"] for det in clean_dets)
    attacked_classes = Counter(det["class_name"] for det in attacked_dets)
    if clean_classes != attacked_classes:
        return True

    clean_mean = float(np.mean([det.get("confidence", det.get("fused_confidence", 0.0)) for det in clean_dets])) if clean_dets else 0.0
    attacked_mean = (
        float(np.mean([det.get("confidence", det.get("fused_confidence", 0.0)) for det in attacked_dets]))
        if attacked_dets
        else 0.0
    )
    return abs(clean_mean - attacked_mean) >= conf_delta_threshold


def camera_decision_from_detections(detections: List[dict]) -> str:
    for det in detections:
        if det["class_name"] in CRITICAL_CLASSES and float(det.get("confidence", 0.0)) >= 0.30:
            return "BRAKE"
    for det in detections:
        if det["class_name"] in VEHICLE_CLASSES and float(det.get("confidence", 0.0)) >= 0.45:
            return "SLOW_DOWN"
    return "KEEP_LANE"


def fusion_decision_from_outputs(
    fused_detections: List[dict],
    lidar_points_xyzi,
    cfg: Dict,
) -> str:
    for det in fused_detections:
        if (
            det["class_name"] in CRITICAL_CLASSES
            and float(det.get("fused_confidence", 0.0)) >= 0.25
            and bool(det.get("lidar_confirmed", False))
        ):
            return "BRAKE"

    obstacle_threshold = int(cfg.get("fusion_obstacle_point_threshold", 45))
    close_obstacle_points = count_close_front_points(lidar_points_xyzi)
    if close_obstacle_points >= obstacle_threshold:
        return "BRAKE"

    for det in fused_detections:
        if det["class_name"] in VEHICLE_CLASSES and float(det.get("fused_confidence", 0.0)) >= 0.40:
            return "SLOW_DOWN"
    return "KEEP_LANE"


def count_close_front_points(points_xyzi) -> int:
    if points_xyzi is None or len(points_xyzi) == 0:
        return 0
    x = points_xyzi[:, 0]
    y = points_xyzi[:, 1]
    z = points_xyzi[:, 2]
    mask = (x >= 1.5) & (x <= 12.0) & (np.abs(y) <= 1.8) & (z >= -2.5) & (z <= 1.2)
    return int(np.count_nonzero(mask))
