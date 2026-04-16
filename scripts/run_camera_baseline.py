import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

from utils.carla_utils import (
    apply_weather,
    attach_rgb_camera,
    cleanup_actor_ids,
    cleanup_actors,
    connect_to_carla,
    get_synced_camera_image,
    image_to_bgr_array,
    set_world_synchronous,
    spawn_ego_vehicle,
    spawn_npc_vehicles,
)
from utils.io_utils import ensure_directories, make_run_id, save_metrics_json
from utils.yolo_utils import YoloDetector, draw_detections, save_bgr_image


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("camera_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def write_detection_rows(csv_writer, jsonl_fp, run_id: str, frame_info: dict, detections: list) -> int:
    timestamp = frame_info["sim_time_seconds"]
    frame_id = frame_info["frame_id"]
    image_path = frame_info["raw_image_path"]

    rows_written = 0
    if not detections:
        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "frame_id": frame_id,
            "class_name": "__none__",
            "confidence": 0.0,
            "bbox_x1": "",
            "bbox_y1": "",
            "bbox_x2": "",
            "bbox_y2": "",
            "raw_image_path": image_path,
        }
        csv_writer.writerow(row)
        jsonl_fp.write(json.dumps(row) + "\n")
        return 1

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "frame_id": frame_id,
            "class_name": det["class_name"],
            "confidence": round(det["confidence"], 6),
            "bbox_x1": round(x1, 2),
            "bbox_y1": round(y1, 2),
            "bbox_x2": round(x2, 2),
            "bbox_y2": round(y2, 2),
            "raw_image_path": image_path,
        }
        csv_writer.writerow(row)
        jsonl_fp.write(json.dumps(row) + "\n")
        rows_written += 1
    return rows_written


def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a.union(b)
    if not union:
        return 1.0
    return len(a.intersection(b)) / len(union)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2: camera-only CARLA + YOLO baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage2_camera_daylight.json",
        help="Path to baseline JSON config.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional fixed run id for reproducible scenario sweeps.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1
    config = load_config(config_path)

    run_id = args.run_id.strip() or make_run_id(config.get("experiment_name", "camera_baseline"))
    output_dirs = ensure_directories(
        {
            "logs": "outputs/logs",
            "metrics": "outputs/metrics",
            "raw_root": "outputs/images/raw",
            "annotated_root": "outputs/images/annotated",
        }
    )
    raw_dir = output_dirs["raw_root"] / run_id
    annotated_dir = output_dirs["annotated_root"] / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dirs["logs"] / f"{run_id}.log"
    logger = build_logger(log_path)
    logger.info("Starting Stage 2 camera baseline run_id=%s", run_id)
    logger.info("Using config: %s", config_path)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    metrics = {
        "run_id": run_id,
        "experiment_name": config.get("experiment_name", "camera_baseline"),
        "status": "started",
        "config_path": str(config_path),
        "raw_frames_dir": str(raw_dir),
        "annotated_frames_dir": str(annotated_dir),
        "scenario_name": config.get("scenario", {}).get("name", "default"),
        "weather_conditions": config.get("weather", {}),
        "vehicle_count": int(config.get("traffic", {}).get("vehicle_count", 0)),
        "pedestrian_count": int(config.get("pedestrians", {}).get("walker_count", 0)),
    }

    actors = []
    npc_vehicle_ids = []
    original_settings = None
    client = None
    world = None
    csv_fp = None
    jsonl_fp = None
    start_time = time.time()

    try:
        detector = YoloDetector(config.get("detector", {}), logger)
        client, world = connect_to_carla(config.get("carla", {}), logger)
        original_settings = world.get_settings()

        sim_cfg = config.get("simulation", {})
        fixed_delta_seconds = float(sim_cfg.get("fixed_delta_seconds", 0.05))
        set_world_synchronous(world, enabled=True, fixed_delta_seconds=fixed_delta_seconds)
        apply_weather(world, config.get("weather", {}), logger)

        traffic_cfg = config.get("traffic", {})
        tm_port = int(config.get("carla", {}).get("traffic_manager_port", 8000))
        npc_vehicle_ids = spawn_npc_vehicles(
            client=client,
            world=world,
            vehicle_count=int(traffic_cfg.get("vehicle_count", 0)),
            traffic_manager_port=tm_port,
            seed=seed,
            logger=logger,
        )

        ego_cfg = config.get("ego_vehicle", {})
        ego = spawn_ego_vehicle(
            world=world,
            vehicle_filter=ego_cfg.get("blueprint_filter", "vehicle.tesla.model3"),
            spawn_point_index=int(ego_cfg.get("spawn_point_index", 0)),
            logger=logger,
        )
        actors.append(ego)

        if bool(ego_cfg.get("autopilot_enabled", True)):
            ego.set_autopilot(True, tm_port)

        camera = attach_rgb_camera(world, ego, config.get("sensors", {}).get("camera", {}), logger)
        actors.append(camera)

        import queue

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        csv_path = output_dirs["logs"] / f"{run_id}_detections.csv"
        jsonl_path = output_dirs["logs"] / f"{run_id}_detections.jsonl"
        frame_metrics_path = output_dirs["logs"] / f"{run_id}_frame_metrics.csv"
        csv_fp = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_fp,
            fieldnames=[
                "run_id",
                "timestamp",
                "frame_id",
                "class_name",
                "confidence",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "raw_image_path",
            ],
        )
        csv_writer.writeheader()
        jsonl_fp = jsonl_path.open("w", encoding="utf-8")
        frame_metrics_fp = frame_metrics_path.open("w", newline="", encoding="utf-8")
        frame_metrics_writer = csv.DictWriter(
            frame_metrics_fp,
            fieldnames=[
                "run_id",
                "scenario_name",
                "timestamp",
                "frame_id",
                "detection_count",
                "unique_classes",
                "missed_detection_flag",
                "unstable_classification_flag",
                "raw_image_path",
                "annotated_image_path",
            ],
        )
        frame_metrics_writer.writeheader()

        total_frames = int(sim_cfg.get("total_frames", 120))
        warmup_frames = int(sim_cfg.get("warmup_frames", 10))
        save_every_n = max(1, int(sim_cfg.get("save_every_n", 1)))
        rows_logged = 0
        frames_processed = 0
        missed_frames = 0
        unstable_frames = 0
        previous_classes = set()
        previous_detection_count = 0
        scenario_name = config.get("scenario", {}).get("name", "default")
        stability_jaccard_threshold = float(config.get("scenario", {}).get("stability_jaccard_threshold", 0.3))

        logger.info(
            "Capturing frames: total=%d, warmup=%d, save_every_n=%d",
            total_frames,
            warmup_frames,
            save_every_n,
        )

        for step in range(total_frames + warmup_frames):
            image = get_synced_camera_image(world, image_queue, timeout_seconds=2.0)
            if step < warmup_frames:
                continue
            if ((step - warmup_frames) % save_every_n) != 0:
                continue

            frame_bgr = image_to_bgr_array(image)
            detections = detector.detect(frame_bgr)
            annotated = draw_detections(frame_bgr, detections)

            frame_id = int(image.frame)
            sim_time = float(image.timestamp)
            raw_path = raw_dir / f"frame_{frame_id:06d}.png"
            ann_path = annotated_dir / f"frame_{frame_id:06d}.png"

            save_bgr_image(frame_bgr, raw_path)
            save_bgr_image(annotated, ann_path)

            frame_info = {
                "frame_id": frame_id,
                "sim_time_seconds": round(sim_time, 6),
                "raw_image_path": str(raw_path),
            }
            rows_logged += write_detection_rows(csv_writer, jsonl_fp, run_id, frame_info, detections)
            frames_processed += 1

            current_classes = {det["class_name"] for det in detections}
            missed_flag = int(previous_detection_count > 0 and len(detections) == 0)
            unstable_flag = int(
                bool(previous_classes)
                and bool(current_classes)
                and jaccard_similarity(previous_classes, current_classes) < stability_jaccard_threshold
            )
            missed_frames += missed_flag
            unstable_frames += unstable_flag
            frame_metrics_writer.writerow(
                {
                    "run_id": run_id,
                    "scenario_name": scenario_name,
                    "timestamp": round(sim_time, 6),
                    "frame_id": frame_id,
                    "detection_count": len(detections),
                    "unique_classes": "|".join(sorted(current_classes)) if current_classes else "__none__",
                    "missed_detection_flag": missed_flag,
                    "unstable_classification_flag": unstable_flag,
                    "raw_image_path": str(raw_path),
                    "annotated_image_path": str(ann_path),
                }
            )
            previous_classes = current_classes
            previous_detection_count = len(detections)

            if frames_processed % 20 == 0:
                logger.info("Processed frames=%d, detection_rows=%d", frames_processed, rows_logged)

        frame_metrics_fp.close()
        metrics.update(
            {
                "status": "completed",
                "frames_processed": frames_processed,
                "detection_rows": rows_logged,
                "detections_csv": str(csv_path),
                "detections_jsonl": str(jsonl_path),
                "frame_metrics_csv": str(frame_metrics_path),
                "missed_detection_frames": missed_frames,
                "unstable_classification_frames": unstable_frames,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Camera baseline failed: %s", exc)
        metrics["status"] = "failed"
        metrics["error"] = str(exc)
        return_code = 1
    else:
        return_code = 0
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()
        if csv_fp is not None:
            csv_fp.close()
        if "frame_metrics_fp" in locals() and frame_metrics_fp is not None and not frame_metrics_fp.closed:
            frame_metrics_fp.close()

        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        cleanup_actors(actors, logger)
        if client is not None and npc_vehicle_ids:
            cleanup_actor_ids(client, npc_vehicle_ids, logger)

        metrics["duration_seconds"] = round(time.time() - start_time, 3)
        metrics_path = output_dirs["metrics"] / f"{run_id}_stage2_camera_baseline.json"
        save_metrics_json(metrics, metrics_path)
        logger.info("Saved run metrics: %s", metrics_path)
        logger.info("Run status: %s", metrics["status"])

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
