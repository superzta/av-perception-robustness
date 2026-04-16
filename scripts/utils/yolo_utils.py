from pathlib import Path
from typing import List

import numpy as np
import shutil


YOLO_INSTALL_HINT = (
    "Ultralytics YOLO is not installed. Install dependencies in your venv with:\n"
    "pip install ultralytics opencv-python numpy pillow"
)


class YoloDetector:
    def __init__(self, detector_cfg: dict, logger):
        self.logger = logger
        self.model_name = detector_cfg.get("model", "models/weights/yolo/yolov8n.pt")
        self.confidence_threshold = float(detector_cfg.get("confidence_threshold", 0.25))
        self.target_classes = set(detector_cfg.get("target_classes", []))
        self.require_yolo = bool(detector_cfg.get("require_yolo", True))
        self._model = None

        try:
            from ultralytics import YOLO
        except ImportError:
            if self.require_yolo:
                raise RuntimeError(YOLO_INSTALL_HINT)
            self.logger.warning("%s", YOLO_INSTALL_HINT)
            return

        model_source = self._resolve_local_model_source(Path(self.model_name).resolve(), YOLO)
        self._model = YOLO(str(model_source))
        self.logger.info("Loaded YOLO model: %s", str(model_source))

    @property
    def enabled(self) -> bool:
        return self._model is not None

    def _resolve_local_model_source(self, model_path: Path, yolo_cls):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists():
            return model_path

        default_name = model_path.name
        if default_name == "yolov8n.pt":
            self.logger.info("Model not found locally. Downloading default %s ...", default_name)
            bootstrap_model = yolo_cls(default_name)
            source_path = Path(getattr(bootstrap_model, "ckpt_path", default_name))
            if source_path.exists():
                shutil.copy2(source_path, model_path)
                self.logger.info("Copied YOLO weight to %s", str(model_path))
                return model_path

        return default_name

    def detect(self, frame_bgr: np.ndarray) -> List[dict]:
        if not self.enabled:
            return []

        results = self._model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        detections = []
        if result.boxes is None:
            return detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for idx in range(len(boxes_xyxy)):
            cls_id = int(classes[idx])
            cls_name = str(names.get(cls_id, str(cls_id))).lower()
            if self.target_classes and cls_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = boxes_xyxy[idx].tolist()
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": float(confs[idx]),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
        return detections


def save_bgr_image(image_bgr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cv2.imwrite(str(output_path), image_bgr)
    except ImportError:
        from PIL import Image

        image_rgb = image_bgr[:, :, ::-1]
        Image.fromarray(image_rgb).save(output_path)


def draw_detections(image_bgr: np.ndarray, detections: List[dict]) -> np.ndarray:
    annotated = image_bgr.copy()
    try:
        import cv2

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated
    except ImportError:
        from PIL import Image, ImageDraw

        image_rgb = annotated[:, :, ::-1]
        pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil)
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
            draw.text((x1, max(0, y1 - 12)), f"{det['class_name']} {det['confidence']:.2f}", fill=(0, 255, 0))
        rgb_array = np.array(pil)
        return rgb_array[:, :, ::-1]
