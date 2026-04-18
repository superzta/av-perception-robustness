import math
from typing import Dict, List, Optional, Tuple

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


def _map_detection_to_semantic_ids(class_name: str) -> set:
    name = str(class_name).lower()
    # PASCAL VOC ids from torchvision DeepLabV3 head:
    # 0 background, 7 car, 14 motorbike, 15 person, 19 train, 6 bus (approx via "bus"/"train")
    if name in {"person"}:
        return {15}
    if name in {"motorcycle"}:
        return {14}
    if name in {"car", "truck", "bus"}:
        return {7, 19, 6}
    if name in {"bicycle"}:
        return {2}
    return set()


def apply_pointpainting_fusion(
    detections: List[dict],
    projected_uv: np.ndarray,
    projected_depths: np.ndarray,
    point_semantic_ids: np.ndarray,
    point_semantic_scores: np.ndarray,
    fusion_cfg: Dict,
) -> List[dict]:
    min_points = int(fusion_cfg.get("min_lidar_points_in_bbox", 8))
    min_depth = float(fusion_cfg.get("min_depth_m", 0.5))
    max_depth = float(fusion_cfg.get("max_depth_m", 60.0))
    unconfirmed_penalty = float(fusion_cfg.get("unconfirmed_confidence_penalty", 0.6))
    semantic_match_min_ratio = float(fusion_cfg.get("semantic_match_min_ratio", 0.15))
    semantic_match_min_score = float(fusion_cfg.get("semantic_match_min_score", 0.30))
    semantic_bonus = float(fusion_cfg.get("semantic_bonus_confidence", 0.08))

    fused = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        bbox_mask = (
            (projected_uv[:, 0] >= x1)
            & (projected_uv[:, 0] <= x2)
            & (projected_uv[:, 1] >= y1)
            & (projected_uv[:, 1] <= y2)
            & (projected_depths >= min_depth)
            & (projected_depths <= max_depth)
        )
        lidar_points_in_bbox = int(np.count_nonzero(bbox_mask))
        lidar_confirmed = lidar_points_in_bbox >= min_points

        if lidar_points_in_bbox > 0:
            sem_ids = point_semantic_ids[bbox_mask]
            sem_scores = point_semantic_scores[bbox_mask]
            target_ids = _map_detection_to_semantic_ids(det["class_name"])
            if target_ids:
                semantic_match = np.isin(sem_ids, list(target_ids))
                semantic_match_points = int(np.count_nonzero(semantic_match))
                semantic_match_ratio = float(semantic_match_points / max(1, lidar_points_in_bbox))
                semantic_match_score = float(np.mean(sem_scores[semantic_match])) if semantic_match_points > 0 else 0.0
            else:
                semantic_match_points = 0
                semantic_match_ratio = 0.0
                semantic_match_score = 0.0
        else:
            semantic_match_points = 0
            semantic_match_ratio = 0.0
            semantic_match_score = 0.0

        semantic_confirmed = (
            semantic_match_ratio >= semantic_match_min_ratio and semantic_match_score >= semantic_match_min_score
        )
        pointpainting_confirmed = bool(lidar_confirmed and semantic_confirmed)

        if pointpainting_confirmed:
            fused_confidence = min(0.999, float(det["confidence"] + semantic_bonus))
        elif lidar_confirmed:
            fused_confidence = float(det["confidence"])
        else:
            fused_confidence = float(det["confidence"] * unconfirmed_penalty)

        fused_det = dict(det)
        fused_det["lidar_points_in_bbox"] = lidar_points_in_bbox
        fused_det["lidar_confirmed"] = lidar_confirmed
        fused_det["semantic_match_points"] = semantic_match_points
        fused_det["semantic_match_ratio"] = round(semantic_match_ratio, 6)
        fused_det["semantic_match_score"] = round(semantic_match_score, 6)
        fused_det["pointpainting_confirmed"] = pointpainting_confirmed
        fused_det["fused_confidence"] = fused_confidence
        fused.append(fused_det)

    return fused


class SemanticPointPainter:
    def __init__(self, fusion_cfg: Dict, logger):
        self.logger = logger
        self.enabled = str(fusion_cfg.get("mode", "")).lower().startswith("pointpainting")
        self.require_pointpainting = bool(fusion_cfg.get("require_pointpainting", False))
        self.device = str(fusion_cfg.get("segmentation_device", "auto")).lower()
        self.model_name = str(fusion_cfg.get("segmentation_model", "deeplabv3_resnet50"))
        self._model = None
        self._torch = None
        if self.enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from torchvision.models.segmentation import (
                DeepLabV3_ResNet50_Weights,
                deeplabv3_resnet50,
            )
        except Exception as exc:  # noqa: BLE001
            if self.require_pointpainting:
                raise RuntimeError(
                    "PointPainting requires torch + torchvision. Install with "
                    "`pip install torch torchvision`."
                ) from exc
            self.logger.warning("PointPainting unavailable (missing torch/torchvision). Falling back to late fusion.")
            self.enabled = False
            return

        if self.model_name != "deeplabv3_resnet50":
            self.logger.warning("Unsupported segmentation model '%s'. Using deeplabv3_resnet50.", self.model_name)

        cuda_available = bool(torch.cuda.is_available())
        requested_cuda = self.device.startswith("cuda") or self.device == "gpu"
        if self.device == "auto":
            requested_cuda = True

        if requested_cuda and (not cuda_available):
            msg = (
                "PointPainting requested CUDA, but torch.cuda.is_available() is False. "
                f"Installed torch={torch.__version__} (cuda={torch.version.cuda}). "
                "Install a CUDA-enabled torch/torchvision build."
            )
            if self.require_pointpainting:
                raise RuntimeError(msg)
            self.logger.warning("%s Falling back to CPU.", msg)

        if requested_cuda and cuda_available:
            self._device_obj = torch.device("cuda:0" if self.device in {"auto", "cuda", "gpu"} else self.device)
        else:
            self._device_obj = torch.device("cpu")

        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self._preprocess = weights.transforms()
        self._model = deeplabv3_resnet50(weights=weights).to(self._device_obj)
        self._model.eval()
        self._torch = torch
        self.logger.info(
            "Loaded PointPainting segmentation backend=%s device=%s torch=%s torch_cuda=%s cuda_available=%s",
            self.model_name,
            self._device_obj,
            torch.__version__,
            torch.version.cuda,
            cuda_available,
        )

    def paint_points(
        self,
        image_bgr: np.ndarray,
        projected_uv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if (not self.enabled) or self._model is None:
            n = projected_uv.shape[0]
            return np.zeros((n,), dtype=np.int32), np.zeros((n,), dtype=np.float32)

        torch = self._torch
        image_rgb = image_bgr[:, :, ::-1]
        pil = None
        try:
            from PIL import Image

            pil = Image.fromarray(image_rgb)
        except Exception as exc:  # noqa: BLE001
            if self.require_pointpainting:
                raise RuntimeError("PointPainting requires Pillow for RGB preprocessing.") from exc
            n = projected_uv.shape[0]
            return np.zeros((n,), dtype=np.int32), np.zeros((n,), dtype=np.float32)

        xyt = self._preprocess(pil).unsqueeze(0).to(self._device_obj)
        with torch.no_grad():
            logits = self._model(xyt)["out"][0]  # [C,H,W]
            probs = torch.softmax(logits, dim=0)
            sem_ids = torch.argmax(probs, dim=0).cpu().numpy().astype(np.int32)
            sem_scores = torch.max(probs, dim=0).values.cpu().numpy().astype(np.float32)

        h, w = sem_ids.shape
        u = np.round(projected_uv[:, 0]).astype(np.int32)
        v = np.round(projected_uv[:, 1]).astype(np.int32)
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        out_ids = np.zeros((projected_uv.shape[0],), dtype=np.int32)
        out_scores = np.zeros((projected_uv.shape[0],), dtype=np.float32)
        out_ids[valid] = sem_ids[v[valid], u[valid]]
        out_scores[valid] = sem_scores[v[valid], u[valid]]
        return out_ids, out_scores


def draw_fusion_detections(image_bgr: np.ndarray, fused_detections: List[dict], header_text: str = "") -> np.ndarray:
    annotated = image_bgr.copy()
    try:
        import cv2

        for det in fused_detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            if det.get("pointpainting_confirmed", False):
                color = (255, 80, 0)
                status = "PP_CONF"
            elif det.get("lidar_confirmed", False):
                color = (0, 220, 0)
                status = "CONF"
            else:
                color = (0, 165, 255)
                status = "RGB_ONLY"

            label = (
                f"{det['class_name']} y={det['confidence']:.2f} "
                f"f={det.get('fused_confidence', det['confidence']):.2f} "
                f"lp={det.get('lidar_points_in_bbox', 0)} "
                f"sp={det.get('semantic_match_points', 0)} {status}"
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
            if det.get("pointpainting_confirmed", False):
                color = (255, 80, 0)
                status = "PP_CONF"
            elif det.get("lidar_confirmed", False):
                color = (0, 220, 0)
                status = "CONF"
            else:
                color = (255, 165, 0)
                status = "RGB_ONLY"
            label = (
                f"{det['class_name']} y={det['confidence']:.2f} "
                f"f={det.get('fused_confidence', det['confidence']):.2f} "
                f"lp={det.get('lidar_points_in_bbox', 0)} "
                f"sp={det.get('semantic_match_points', 0)} {status}"
            )
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
            draw.text((x1, max(0, y1 - 12)), label, fill=color)
        if header_text:
            draw.text((10, 10), header_text, fill=(255, 255, 255))
        rgb_array = np.array(pil)
        return rgb_array[:, :, ::-1]
