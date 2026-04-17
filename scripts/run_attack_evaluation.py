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

from utils.attack_utils import apply_camera_attack, apply_lidar_attack, save_lidar_bev
from utils.carla_utils import (
    apply_weather,
    attach_lidar,
    attach_rgb_camera,
    cleanup_actor_ids,
    cleanup_actors,
    connect_to_carla,
    get_synced_camera_image,
    image_to_bgr_array,
    lidar_measurement_to_array,
    set_world_synchronous,
    spawn_ego_vehicle,
    spawn_npc_vehicles,
)
from utils.decision_utils import (
    camera_decision_from_detections,
    count_close_front_points,
    detection_changed,
    fusion_decision_from_outputs,
)
from utils.fusion_utils import (
    apply_late_fusion,
    build_camera_intrinsics,
    draw_fusion_detections,
    get_synced_rgb_lidar,
    project_lidar_to_image,
)
from utils.io_utils import ensure_directories, make_run_id, save_metrics_json
from utils.yolo_utils import YoloDetector, draw_detections, save_bgr_image


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_logger(log_path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def add_header_text(image_bgr: np.ndarray, header: str) -> np.ndarray:
    out = image_bgr.copy()
    try:
        import cv2

        cv2.putText(out, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    except ImportError:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(out[:, :, ::-1])
        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), header, fill=(255, 255, 255))
        arr = np.array(pil)
        return arr[:, :, ::-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5 adversarial attack evaluation for camera-only and fusion pipelines.")
    parser.add_argument("--config", type=str, default="configs/stage5_camera_attack.json")
    parser.add_argument("--pipeline", type=str, choices=["camera_only", "fusion"], default="camera_only")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--frames", type=int, default=0, help="Optional override of total_frames for quick tests.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1
    config = load_config(config_path)

    run_id = args.run_id.strip() or make_run_id(f"stage5_{args.pipeline}_{config.get('experiment_name', 'attack_eval')}")
    out_dirs = ensure_directories(
        {
            "root": f"outputs/attacks/{args.pipeline}/{run_id}",
            "metrics": "outputs/metrics",
        }
    )
    root = out_dirs["root"]
    rgb_clean_dir = root / "rgb_clean"
    rgb_attack_dir = root / "rgb_attacked"
    ann_clean_dir = root / "annotated_clean"
    ann_attack_dir = root / "annotated_attacked"
    lidar_clean_dir = root / "lidar_clean"
    lidar_attack_dir = root / "lidar_attacked"
    lidar_bev_clean_dir = root / "lidar_bev_clean"
    lidar_bev_attack_dir = root / "lidar_bev_attacked"
    logs_dir = root / "logs"
    for d in [
        rgb_clean_dir,
        rgb_attack_dir,
        ann_clean_dir,
        ann_attack_dir,
        lidar_clean_dir,
        lidar_attack_dir,
        lidar_bev_clean_dir,
        lidar_bev_attack_dir,
        logs_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    logger = build_logger(logs_dir / f"{run_id}.log", f"stage5_{args.pipeline}")
    logger.info("Starting Stage 5 attack evaluation run_id=%s pipeline=%s", run_id, args.pipeline)
    logger.info("Config: %s", str(config_path))

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    metrics = {
        "run_id": run_id,
        "pipeline": args.pipeline,
        "experiment_name": config.get("experiment_name", "stage5_attack"),
        "scenario_name": config.get("scenario", {}).get("name", "default"),
        "weather_conditions": config.get("weather", {}),
        "vehicle_count": int(config.get("traffic", {}).get("vehicle_count", 0)),
        "pedestrian_count": int(config.get("pedestrians", {}).get("walker_count", 0)),
        "status": "started",
        "output_root": str(root),
    }

    actors = []
    npc_vehicle_ids = []
    client = None
    world = None
    original_settings = None
    frame_csv_fp = None
    frame_jsonl_fp = None
    start = time.time()

    try:
        detector = YoloDetector(config.get("detector", {}), logger)
        client, world = connect_to_carla(config.get("carla", {}), logger)
        original_settings = world.get_settings()

        sim_cfg = config.get("simulation", {})
        total_frames = int(sim_cfg.get("total_frames", 150))
        if args.frames > 0:
            total_frames = int(args.frames)
        warmup_frames = int(sim_cfg.get("warmup_frames", 20))
        save_every_n = max(1, int(sim_cfg.get("save_every_n", 1)))

        set_world_synchronous(world, enabled=True, fixed_delta_seconds=float(sim_cfg.get("fixed_delta_seconds", 0.05)))
        apply_weather(world, config.get("weather", {}), logger)

        tm_port = int(config.get("carla", {}).get("traffic_manager_port", 8000))
        npc_vehicle_ids = spawn_npc_vehicles(
            client,
            world,
            vehicle_count=int(config.get("traffic", {}).get("vehicle_count", 0)),
            traffic_manager_port=tm_port,
            seed=seed,
            logger=logger,
        )

        ego_cfg = config.get("ego_vehicle", {})
        ego = spawn_ego_vehicle(
            world,
            vehicle_filter=ego_cfg.get("blueprint_filter", "vehicle.tesla.model3"),
            spawn_point_index=int(ego_cfg.get("spawn_point_index", 0)),
            logger=logger,
        )
        actors.append(ego)
        if bool(ego_cfg.get("autopilot_enabled", True)):
            ego.set_autopilot(True, tm_port)

        camera = attach_rgb_camera(world, ego, config.get("sensors", {}).get("camera", {}), logger)
        actors.append(camera)
        camera_q = queue.Queue()
        camera.listen(camera_q.put)

        lidar = None
        lidar_q = None
        if args.pipeline == "fusion":
            lidar_cfg = config.get("sensors", {}).get("lidar", {})
            if not bool(lidar_cfg.get("enabled", True)):
                raise RuntimeError("Fusion pipeline requires sensors.lidar.enabled=true in config.")
            lidar = attach_lidar(world, ego, lidar_cfg, logger)
            actors.append(lidar)
            lidar_q = queue.Queue()
            lidar.listen(lidar_q.put)

            cam_cfg = config.get("sensors", {}).get("camera", {})
            intrinsics = build_camera_intrinsics(
                int(cam_cfg.get("image_width", 1280)),
                int(cam_cfg.get("image_height", 720)),
                float(cam_cfg.get("fov", 90)),
            )
            fusion_cfg = config.get("fusion", {})
        else:
            intrinsics = None
            fusion_cfg = {}

        frame_csv_path = logs_dir / f"{run_id}_frame_attack_log.csv"
        frame_jsonl_path = logs_dir / f"{run_id}_frame_attack_log.jsonl"
        frame_csv_fp = frame_csv_path.open("w", newline="", encoding="utf-8")
        frame_writer = csv.DictWriter(
            frame_csv_fp,
            fieldnames=[
                "run_id",
                "pipeline",
                "scenario_name",
                "timestamp",
                "frame_id",
                "camera_attack_type",
                "lidar_attack_type",
                "clean_detection_count",
                "attacked_detection_count",
                "detection_changed",
                "clean_decision",
                "attacked_decision",
                "decision_changed",
                "lidar_injected_points",
                "clean_front_obstacle_points",
                "attacked_front_obstacle_points",
                "rgb_clean_path",
                "rgb_attacked_path",
                "annotated_clean_path",
                "annotated_attacked_path",
                "lidar_clean_path",
                "lidar_attacked_path",
                "lidar_bev_clean_path",
                "lidar_bev_attacked_path",
            ],
        )
        frame_writer.writeheader()
        frame_jsonl_fp = frame_jsonl_path.open("w", encoding="utf-8")

        logger.info("Attack loop started. total_frames=%d warmup=%d save_every_n=%d", total_frames, warmup_frames, save_every_n)

        frames_processed = 0
        detection_changed_count = 0
        decision_changed_count = 0

        for step in range(total_frames + warmup_frames):
            if args.pipeline == "fusion":
                rgb_data, lidar_data = get_synced_rgb_lidar(world, camera_q, lidar_q, timeout_seconds=2.0)
                lidar_points_clean = lidar_measurement_to_array(lidar_data)
            else:
                rgb_data = get_synced_camera_image(world, camera_q, timeout_seconds=2.0)
                lidar_points_clean = None

            if step < warmup_frames:
                continue
            if ((step - warmup_frames) % save_every_n) != 0:
                continue

            frame_id = int(rgb_data.frame)
            ts = float(rgb_data.timestamp)
            rgb_clean = image_to_bgr_array(rgb_data)
            rgb_attacked, cam_meta = apply_camera_attack(rgb_clean, config.get("attacks", {}).get("camera", {}))

            clean_dets = detector.detect(rgb_clean)
            attacked_dets = detector.detect(rgb_attacked)

            lidar_points_attacked = None
            lidar_meta = {"lidar_attack_type": "none", "injected_points": 0}
            if args.pipeline == "fusion":
                lidar_points_attacked, lidar_meta = apply_lidar_attack(
                    lidar_points_clean, config.get("attacks", {}).get("lidar", {}), frame_id=frame_id
                )

                uv_clean, depth_clean = project_lidar_to_image(lidar_points_clean, camera, lidar, intrinsics)
                uv_att, depth_att = project_lidar_to_image(lidar_points_attacked, camera, lidar, intrinsics)
                clean_out = apply_late_fusion(clean_dets, uv_clean, depth_clean, fusion_cfg)
                attacked_out = apply_late_fusion(attacked_dets, uv_att, depth_att, fusion_cfg)
                clean_front_points = count_close_front_points(lidar_points_clean)
                attacked_front_points = count_close_front_points(lidar_points_attacked)
                clean_decision = fusion_decision_from_outputs(clean_out, lidar_points_clean, config.get("decision", {}))
                attacked_decision = fusion_decision_from_outputs(attacked_out, lidar_points_attacked, config.get("decision", {}))
                delta_trigger = int(config.get("decision", {}).get("lidar_attack_delta_trigger", 120))
                if (
                    config.get("attacks", {}).get("lidar", {}).get("enabled", False)
                    and (attacked_front_points - clean_front_points) >= delta_trigger
                ):
                    attacked_decision = "BRAKE_SPOOF_ALERT"
                annotated_clean = draw_fusion_detections(rgb_clean, clean_out, header_text=f"clean decision={clean_decision}")
                annotated_att = draw_fusion_detections(rgb_attacked, attacked_out, header_text=f"attacked decision={attacked_decision}")
                clean_eval = clean_out
                attacked_eval = attacked_out
            else:
                if config.get("attacks", {}).get("lidar", {}).get("enabled", False):
                    logger.info("Frame %d: LiDAR attack configured but skipped for camera_only pipeline.", frame_id)
                clean_decision = camera_decision_from_detections(clean_dets)
                attacked_decision = camera_decision_from_detections(attacked_dets)
                annotated_clean = add_header_text(draw_detections(rgb_clean, clean_dets), f"clean decision={clean_decision}")
                annotated_att = add_header_text(draw_detections(rgb_attacked, attacked_dets), f"attacked decision={attacked_decision}")
                clean_eval = clean_dets
                attacked_eval = attacked_dets
                clean_front_points = 0
                attacked_front_points = 0

            det_changed = int(detection_changed(clean_eval, attacked_eval))
            dec_changed = int(clean_decision != attacked_decision)
            detection_changed_count += det_changed
            decision_changed_count += dec_changed

            rgb_clean_path = rgb_clean_dir / f"frame_{frame_id:06d}.png"
            rgb_attack_path = rgb_attack_dir / f"frame_{frame_id:06d}.png"
            ann_clean_path = ann_clean_dir / f"frame_{frame_id:06d}.png"
            ann_attack_path = ann_attack_dir / f"frame_{frame_id:06d}.png"
            save_bgr_image(rgb_clean, rgb_clean_path)
            save_bgr_image(rgb_attacked, rgb_attack_path)
            save_bgr_image(annotated_clean, ann_clean_path)
            save_bgr_image(annotated_att, ann_attack_path)

            lidar_clean_path = ""
            lidar_attack_path = ""
            lidar_bev_clean_path = ""
            lidar_bev_attack_path = ""
            if args.pipeline == "fusion":
                lidar_clean_path = str(lidar_clean_dir / f"frame_{frame_id:06d}.npy")
                lidar_attack_path = str(lidar_attack_dir / f"frame_{frame_id:06d}.npy")
                np.save(lidar_clean_path, lidar_points_clean)
                np.save(lidar_attack_path, lidar_points_attacked)
                lidar_bev_clean = lidar_bev_clean_dir / f"frame_{frame_id:06d}.png"
                lidar_bev_attacked = lidar_bev_attack_dir / f"frame_{frame_id:06d}.png"
                save_lidar_bev(lidar_points_clean, lidar_bev_clean)
                save_lidar_bev(lidar_points_attacked, lidar_bev_attacked)
                lidar_bev_clean_path = str(lidar_bev_clean)
                lidar_bev_attack_path = str(lidar_bev_attacked)

            row = {
                "run_id": run_id,
                "pipeline": args.pipeline,
                "scenario_name": config.get("scenario", {}).get("name", "default"),
                "timestamp": round(ts, 6),
                "frame_id": frame_id,
                "camera_attack_type": cam_meta.get("camera_attack_type", "none"),
                "lidar_attack_type": lidar_meta.get("lidar_attack_type", "none"),
                "clean_detection_count": len(clean_eval),
                "attacked_detection_count": len(attacked_eval),
                "detection_changed": det_changed,
                "clean_decision": clean_decision,
                "attacked_decision": attacked_decision,
                "decision_changed": dec_changed,
                "lidar_injected_points": int(lidar_meta.get("injected_points", 0)),
                "clean_front_obstacle_points": clean_front_points,
                "attacked_front_obstacle_points": attacked_front_points,
                "rgb_clean_path": str(rgb_clean_path),
                "rgb_attacked_path": str(rgb_attack_path),
                "annotated_clean_path": str(ann_clean_path),
                "annotated_attacked_path": str(ann_attack_path),
                "lidar_clean_path": lidar_clean_path,
                "lidar_attacked_path": lidar_attack_path,
                "lidar_bev_clean_path": lidar_bev_clean_path,
                "lidar_bev_attacked_path": lidar_bev_attack_path,
            }
            frame_writer.writerow(row)
            frame_jsonl_fp.write(json.dumps(row) + "\n")

            frames_processed += 1
            if frames_processed % 10 == 0:
                logger.info(
                    "Heartbeat: processed=%d detection_changed=%d decision_changed=%d",
                    frames_processed,
                    detection_changed_count,
                    decision_changed_count,
                )

        metrics.update(
            {
                "status": "completed",
                "frames_processed": frames_processed,
                "detection_changed_frames": detection_changed_count,
                "decision_changed_frames": decision_changed_count,
                "frame_attack_csv": str(frame_csv_path),
                "frame_attack_jsonl": str(frame_jsonl_path),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage 5 attack run failed: %s", exc)
        metrics["status"] = "failed"
        metrics["error"] = str(exc)
        return_code = 1
    else:
        return_code = 0
    finally:
        if frame_jsonl_fp is not None:
            frame_jsonl_fp.close()
        if frame_csv_fp is not None:
            frame_csv_fp.close()
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        cleanup_actors(actors, logger)
        if client is not None and npc_vehicle_ids:
            cleanup_actor_ids(client, npc_vehicle_ids, logger)
        metrics["duration_seconds"] = round(time.time() - start, 3)
        metrics_path = out_dirs["metrics"] / f"{run_id}_stage5_attack_eval.json"
        save_metrics_json(metrics, metrics_path)
        logger.info("Saved metrics: %s", str(metrics_path))
        logger.info("Run status: %s", metrics.get("status"))

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
