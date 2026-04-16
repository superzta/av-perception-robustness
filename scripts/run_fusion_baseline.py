import argparse
import csv
import json
import logging
import queue
import random
import sys
import time
from pathlib import Path

import numpy as np

from utils.carla_utils import (
    apply_weather,
    attach_lidar,
    attach_rgb_camera,
    cleanup_actor_ids,
    cleanup_actors,
    connect_to_carla,
    image_to_bgr_array,
    lidar_measurement_to_array,
    set_world_synchronous,
    spawn_ego_vehicle,
    spawn_npc_vehicles,
)
from utils.fusion_utils import (
    apply_late_fusion,
    build_camera_intrinsics,
    draw_fusion_detections,
    get_synced_rgb_lidar,
    project_lidar_to_image,
)
from utils.io_utils import ensure_directories, make_run_id, save_metrics_json
from utils.yolo_utils import YoloDetector, save_bgr_image


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("fusion_baseline")
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


def write_fusion_rows(csv_writer, jsonl_fp, run_id: str, frame_info: dict, fused_detections: list) -> int:
    rows = 0
    base = {
        "run_id": run_id,
        "timestamp": frame_info["sim_time_seconds"],
        "frame_id": frame_info["frame_id"],
        "rgb_path": frame_info["rgb_path"],
        "lidar_path": frame_info["lidar_path"],
        "fusion_annotated_path": frame_info["fusion_annotated_path"],
        "lidar_total_points": frame_info["lidar_total_points"],
        "projected_points_in_image": frame_info["projected_points_in_image"],
    }
    if not fused_detections:
        row = dict(base)
        row.update(
            {
                "class_name": "__none__",
                "yolo_confidence": 0.0,
                "fused_confidence": 0.0,
                "lidar_confirmed": False,
                "lidar_points_in_bbox": 0,
                "bbox_x1": "",
                "bbox_y1": "",
                "bbox_x2": "",
                "bbox_y2": "",
            }
        )
        csv_writer.writerow(row)
        jsonl_fp.write(json.dumps(row) + "\n")
        return 1

    for det in fused_detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        row = dict(base)
        row.update(
            {
                "class_name": det["class_name"],
                "yolo_confidence": round(det["confidence"], 6),
                "fused_confidence": round(det["fused_confidence"], 6),
                "lidar_confirmed": bool(det["lidar_confirmed"]),
                "lidar_points_in_bbox": int(det["lidar_points_in_bbox"]),
                "bbox_x1": round(x1, 2),
                "bbox_y1": round(y1, 2),
                "bbox_x2": round(x2, 2),
                "bbox_y2": round(y2, 2),
            }
        )
        csv_writer.writerow(row)
        jsonl_fp.write(json.dumps(row) + "\n")
        rows += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3: camera + LiDAR late-fusion baseline for CARLA."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage3_fusion_daylight.json",
        help="Path to Stage 3 fusion config JSON.",
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

    run_id = args.run_id.strip() or make_run_id(config.get("experiment_name", "fusion_baseline"))
    output_dirs = ensure_directories(
        {
            "root": "outputs/fusion_baseline",
            "metrics": "outputs/metrics",
        }
    )
    run_root = output_dirs["root"] / run_id
    rgb_dir = run_root / "rgb_raw"
    lidar_dir = run_root / "lidar_points"
    ann_dir = run_root / "fusion_annotated"
    logs_dir = run_root / "logs"
    for directory in (rgb_dir, lidar_dir, ann_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger = build_logger(logs_dir / f"{run_id}.log")
    logger.info("Starting Stage 3 fusion baseline run_id=%s", run_id)
    logger.info("Using config: %s", str(config_path))
    logger.info("Fusion mode: late_fusion_rgb_with_lidar_confirmation")

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    metrics = {
        "run_id": run_id,
        "experiment_name": config.get("experiment_name", "fusion_baseline"),
        "status": "started",
        "fusion_mode": "late_fusion_rgb_with_lidar_confirmation",
        "config_path": str(config_path),
        "output_root": str(run_root),
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

        tm_port = int(config.get("carla", {}).get("traffic_manager_port", 8000))
        npc_vehicle_ids = spawn_npc_vehicles(
            client=client,
            world=world,
            vehicle_count=int(config.get("traffic", {}).get("vehicle_count", 0)),
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

        sensors_cfg = config.get("sensors", {})
        camera_cfg = sensors_cfg.get("camera", {})
        lidar_cfg = sensors_cfg.get("lidar", {})
        if not bool(lidar_cfg.get("enabled", True)):
            raise RuntimeError("Fusion baseline requires sensors.lidar.enabled=true.")

        camera = attach_rgb_camera(world, ego, camera_cfg, logger)
        lidar = attach_lidar(world, ego, lidar_cfg, logger)
        actors.extend([camera, lidar])

        rgb_queue = queue.Queue()
        lidar_queue = queue.Queue()
        camera.listen(rgb_queue.put)
        lidar.listen(lidar_queue.put)

        width = int(camera_cfg.get("image_width", 1280))
        height = int(camera_cfg.get("image_height", 720))
        fov = float(camera_cfg.get("fov", 90))
        intrinsics = build_camera_intrinsics(width, height, fov)

        csv_path = logs_dir / f"{run_id}_fusion_detections.csv"
        jsonl_path = logs_dir / f"{run_id}_fusion_detections.jsonl"
        frame_metrics_path = logs_dir / f"{run_id}_frame_metrics.csv"
        csv_fp = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_fp,
            fieldnames=[
                "run_id",
                "timestamp",
                "frame_id",
                "class_name",
                "yolo_confidence",
                "fused_confidence",
                "lidar_confirmed",
                "lidar_points_in_bbox",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "rgb_path",
                "lidar_path",
                "fusion_annotated_path",
                "lidar_total_points",
                "projected_points_in_image",
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
                "yolo_detection_count",
                "fused_detection_count",
                "lidar_confirmed_count",
                "lidar_unconfirmed_count",
                "unique_classes",
                "missed_detection_flag",
                "unstable_classification_flag",
                "rgb_path",
                "fusion_annotated_path",
            ],
        )
        frame_metrics_writer.writeheader()

        total_frames = int(sim_cfg.get("total_frames", 200))
        warmup_frames = int(sim_cfg.get("warmup_frames", 20))
        save_every_n = max(1, int(sim_cfg.get("save_every_n", 1)))

        frames_processed = 0
        rows_logged = 0
        total_confirmed = 0
        total_unconfirmed = 0
        missed_frames = 0
        unstable_frames = 0
        previous_classes = set()
        previous_detection_count = 0
        scenario_name = config.get("scenario", {}).get("name", "default")
        stability_jaccard_threshold = float(config.get("scenario", {}).get("stability_jaccard_threshold", 0.3))

        fusion_cfg = config.get("fusion", {})
        logger.info(
            "Capturing synchronized RGB+LiDAR: total=%d warmup=%d save_every_n=%d",
            total_frames,
            warmup_frames,
            save_every_n,
        )

        for step in range(total_frames + warmup_frames):
            rgb_image, lidar_data = get_synced_rgb_lidar(world, rgb_queue, lidar_queue, timeout_seconds=2.0)
            if step < warmup_frames:
                continue
            if ((step - warmup_frames) % save_every_n) != 0:
                continue

            frame_id = int(rgb_image.frame)
            sim_time = float(rgb_image.timestamp)
            frame_bgr = image_to_bgr_array(rgb_image)
            lidar_points = lidar_measurement_to_array(lidar_data)

            detections = detector.detect(frame_bgr)
            projected_uv, projected_depths = project_lidar_to_image(
                lidar_points_xyzi=lidar_points,
                camera_actor=camera,
                lidar_actor=lidar,
                camera_intrinsics=intrinsics,
            )
            fused_detections = apply_late_fusion(detections, projected_uv, projected_depths, fusion_cfg)

            confirmed = sum(1 for det in fused_detections if det["lidar_confirmed"])
            unconfirmed = len(fused_detections) - confirmed
            total_confirmed += confirmed
            total_unconfirmed += unconfirmed

            rgb_path = rgb_dir / f"frame_{frame_id:06d}.png"
            lidar_path = lidar_dir / f"frame_{frame_id:06d}.npy"
            ann_path = ann_dir / f"frame_{frame_id:06d}.png"

            save_bgr_image(frame_bgr, rgb_path)
            np.save(lidar_path, lidar_points)

            annotated = draw_fusion_detections(
                frame_bgr,
                fused_detections,
                header_text=(
                    f"frame={frame_id} yolo={len(detections)} "
                    f"confirmed={confirmed} unconfirmed={unconfirmed}"
                ),
            )
            save_bgr_image(annotated, ann_path)

            frame_info = {
                "frame_id": frame_id,
                "sim_time_seconds": round(sim_time, 6),
                "rgb_path": str(rgb_path),
                "lidar_path": str(lidar_path),
                "fusion_annotated_path": str(ann_path),
                "lidar_total_points": int(lidar_points.shape[0]),
                "projected_points_in_image": int(projected_uv.shape[0]),
            }
            rows_logged += write_fusion_rows(csv_writer, jsonl_fp, run_id, frame_info, fused_detections)
            frames_processed += 1

            current_classes = {det["class_name"] for det in fused_detections}
            missed_flag = int(previous_detection_count > 0 and len(fused_detections) == 0)
            if not previous_classes and not current_classes:
                class_jaccard = 1.0
            else:
                union = previous_classes.union(current_classes)
                class_jaccard = (len(previous_classes.intersection(current_classes)) / len(union)) if union else 1.0
            unstable_flag = int(bool(previous_classes) and bool(current_classes) and class_jaccard < stability_jaccard_threshold)
            missed_frames += missed_flag
            unstable_frames += unstable_flag
            frame_metrics_writer.writerow(
                {
                    "run_id": run_id,
                    "scenario_name": scenario_name,
                    "timestamp": round(sim_time, 6),
                    "frame_id": frame_id,
                    "yolo_detection_count": len(detections),
                    "fused_detection_count": len(fused_detections),
                    "lidar_confirmed_count": confirmed,
                    "lidar_unconfirmed_count": unconfirmed,
                    "unique_classes": "|".join(sorted(current_classes)) if current_classes else "__none__",
                    "missed_detection_flag": missed_flag,
                    "unstable_classification_flag": unstable_flag,
                    "rgb_path": str(rgb_path),
                    "fusion_annotated_path": str(ann_path),
                }
            )
            previous_classes = current_classes
            previous_detection_count = len(fused_detections)

            if frames_processed % 20 == 0:
                logger.info(
                    "Processed=%d rows=%d confirmed=%d unconfirmed=%d",
                    frames_processed,
                    rows_logged,
                    total_confirmed,
                    total_unconfirmed,
                )

        metrics.update(
            {
                "status": "completed",
                "frames_processed": frames_processed,
                "rows_logged": rows_logged,
                "total_confirmed_detections": total_confirmed,
                "total_unconfirmed_detections": total_unconfirmed,
                "fusion_csv": str(csv_path),
                "fusion_jsonl": str(jsonl_path),
                "frame_metrics_csv": str(frame_metrics_path),
                "missed_detection_frames": missed_frames,
                "unstable_classification_frames": unstable_frames,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fusion baseline failed: %s", exc)
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
        metrics_path = output_dirs["metrics"] / f"{run_id}_stage3_fusion_baseline.json"
        save_metrics_json(metrics, metrics_path)
        logger.info("Saved run metrics: %s", str(metrics_path))
        logger.info("Run status: %s", metrics["status"])

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
