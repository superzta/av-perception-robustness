import argparse
import csv
import json
import logging
import math
import queue
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import carla_utils
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
from utils.decision_utils import camera_decision_from_detections, fusion_decision_from_outputs
from utils.fusion_utils import (
    apply_late_fusion,
    build_camera_intrinsics,
    draw_fusion_detections,
    get_synced_rgb_lidar,
    project_lidar_to_image,
)
from utils.yolo_utils import YoloDetector, draw_detections, save_bgr_image


carla = carla_utils.carla


@dataclass
class EpisodeSpec:
    condition_name: str
    category: str
    episode_index: int
    seed: int
    town: str
    spawn_index: int
    vehicle_count: int
    pedestrian_count: int
    weather: dict
    camera_transform: dict
    event: dict
    attacks: dict


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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


def preflight_simulator(carla_cfg: dict, logger) -> bool:
    host = carla_cfg.get("host", "127.0.0.1")
    port = int(carla_cfg.get("port", 2000))
    timeout = float(carla_cfg.get("preflight_timeout_seconds", 6.0))
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        world = client.get_world()
        map_name = world.get_map().name
        logger.info("Preflight OK: simulator reachable at %s:%d map=%s", host, port, map_name)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Preflight FAILED: simulator not reachable at %s:%d within %.1fs. "
            "Please launch CarlaUE4.exe and wait until world is loaded. Error: %s",
            host,
            port,
            timeout,
            exc,
        )
        return False


def run_town_switch_diagnostic(carla_cfg: dict, towns: List[str], logger) -> Tuple[bool, List[dict]]:
    host = carla_cfg.get("host", "127.0.0.1")
    port = int(carla_cfg.get("port", 2000))
    timeout = float(carla_cfg.get("timeout_seconds", 20.0))
    results = []
    all_ok = True

    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
    except Exception as exc:  # noqa: BLE001
        logger.error("Town diagnostic failed to create CARLA client: %s", exc)
        return False, []

    for town in towns:
        start = time.time()
        status = "ok"
        error = ""
        try:
            world = client.load_world(town)
            _ = world.get_map().name
            # Stronger health check: spawn one temporary ego+camera and wait a tick.
            spawn_points = world.get_map().get_spawn_points()
            if spawn_points:
                bp_lib = world.get_blueprint_library()
                vehicle_bp = random.choice(bp_lib.filter("vehicle.*"))
                ego = world.try_spawn_actor(vehicle_bp, spawn_points[0])
                if ego is not None:
                    camera_bp = bp_lib.find("sensor.camera.rgb")
                    camera_bp.set_attribute("image_size_x", "640")
                    camera_bp.set_attribute("image_size_y", "360")
                    camera_bp.set_attribute("fov", "90")
                    cam = world.spawn_actor(
                        camera_bp,
                        carla.Transform(carla.Location(x=1.5, z=2.2)),
                        attach_to=ego,
                    )
                    q = queue.Queue()
                    cam.listen(q.put)
                    world.tick()
                    _ = q.get(timeout=3.0)
                    cam.stop()
                    cam.destroy()
                    ego.destroy()
                else:
                    status = "failed"
                    error = "Could not spawn temp ego actor during town diagnostic."
            else:
                status = "failed"
                error = "No spawn points in map."
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)
            all_ok = False
        if status != "ok":
            all_ok = False
        elapsed = round(time.time() - start, 3)
        results.append(
            {
                "town": town,
                "status": status,
                "elapsed_seconds": elapsed,
                "error": error,
            }
        )
        if status == "ok":
            logger.info("Town diagnostic: %s OK in %.2fs", town, elapsed)
        else:
            logger.error("Town diagnostic: %s FAILED in %.2fs error=%s", town, elapsed, error)
    return all_ok, results


def sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ensure_output_layout(project_root: Path) -> Dict[str, Path]:
    outputs_root = project_root / "outputs"
    dirs = {
        "episode_logs": outputs_root / "episode_logs",
        "summary_tables": outputs_root / "summary_tables",
        "plots": outputs_root / "plots",
        "representative_screenshots": outputs_root / "representative_screenshots",
        "final_report_assets": outputs_root / "final_report_assets",
        "final_presentation_assets": outputs_root / "final_presentation_assets",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def attach_collision_sensor(world, ego_vehicle, collision_queue: queue.Queue, logger) -> carla.Actor:
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=ego_vehicle)
    sensor.listen(collision_queue.put)
    logger.info("Attached collision sensor id=%s", sensor.id)
    return sensor


def stabilize_world(world, logger, duration_seconds: float, tick_timeout_seconds: float = 2.0) -> None:
    if duration_seconds <= 0:
        return
    start = time.time()
    ticks = 0
    while (time.time() - start) < duration_seconds:
        try:
            world.wait_for_tick(tick_timeout_seconds)
            ticks += 1
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    logger.info("World stabilization complete: duration=%.1fs ticks=%d", duration_seconds, ticks)


def spawn_target_actor(world, spec: EpisodeSpec, ego_spawn_index: int, tm_port: int, logger) -> Optional[carla.Actor]:
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return None
    target_type = spec.event.get("target_type", "vehicle")
    offset = int(spec.event.get("target_spawn_offset", 5))
    start_idx = (ego_spawn_index + offset + (spec.episode_index * 3)) % len(spawn_points)

    if target_type == "pedestrian":
        bps = world.get_blueprint_library().filter("walker.pedestrian.*")
        if not bps:
            return None
        for shift in range(len(spawn_points)):
            idx = (start_idx + shift) % len(spawn_points)
            actor = world.try_spawn_actor(random.choice(bps), spawn_points[idx])
            if actor is not None:
                logger.info("Spawned target pedestrian id=%s spawn_idx=%s", actor.id, idx)
                return actor
        return None

    vehicle_bps = world.get_blueprint_library().filter("vehicle.*")
    for shift in range(len(spawn_points)):
        idx = (start_idx + shift) % len(spawn_points)
        actor = world.try_spawn_actor(random.choice(vehicle_bps), spawn_points[idx])
        if actor is not None:
            actor.set_autopilot(True, tm_port)
            logger.info("Spawned target vehicle id=%s spawn_idx=%s", actor.id, idx)
            return actor
    return None


def spawn_ego_with_fallback(world, vehicle_filter: str, preferred_spawn_index: int, seed: int, logger):
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available for ego spawn.")

    blueprints = world.get_blueprint_library().filter(vehicle_filter)
    if not blueprints:
        blueprints = world.get_blueprint_library().filter("vehicle.*")
    if not blueprints:
        raise RuntimeError("No vehicle blueprints available for ego spawn.")

    rng = random.Random(seed + 17)
    candidate_indices = []
    n = len(spawn_points)
    preferred = int(preferred_spawn_index) % n
    candidate_indices.append(preferred)
    for shift in range(1, n):
        candidate_indices.append((preferred + shift) % n)

    for idx in candidate_indices:
        bp = rng.choice(blueprints)
        actor = world.try_spawn_actor(bp, spawn_points[idx])
        if actor is not None:
            logger.info("Spawned ego with fallback id=%s blueprint=%s spawn_idx=%s", actor.id, bp.id, idx)
            return actor, idx

    raise RuntimeError(f"Failed to spawn ego vehicle after trying {len(candidate_indices)} spawn points.")


def estimate_target_visibility(
    frame_idx: int,
    camera_actor,
    target_actor,
    event_cfg: dict,
    camera_fov_degrees: float,
) -> Tuple[bool, float]:
    if target_actor is None or not target_actor.is_alive:
        return False, math.nan
    start = int(event_cfg.get("window_start_frame", 20))
    end = int(event_cfg.get("window_end_frame", 70))
    if frame_idx < start or frame_idx > end:
        return False, math.nan

    cam_loc = camera_actor.get_transform().location
    tgt_loc = target_actor.get_transform().location
    distance = float(cam_loc.distance(tgt_loc))
    # Episode event window is the primary ground-truth proxy for visibility.
    # Distance is logged for analysis but does not gate visibility, which avoids sparse positives.
    return True, distance


def is_target_detected(detections: List[dict], target_type: str) -> Tuple[bool, float]:
    if target_type == "pedestrian":
        target_classes = {"person"}
    else:
        target_classes = {"car", "truck", "bus", "motorcycle", "bicycle"}
    confs = [float(det.get("fused_confidence", det.get("confidence", 0.0))) for det in detections if det["class_name"] in target_classes]
    if confs:
        return True, max(confs)
    return False, 0.0


def min_front_obstacle_distance(points_xyzi: Optional[np.ndarray]) -> float:
    if points_xyzi is None or len(points_xyzi) == 0:
        return math.nan
    x = points_xyzi[:, 0]
    y = points_xyzi[:, 1]
    z = points_xyzi[:, 2]
    mask = (x > 0.5) & (x < 40.0) & (np.abs(y) < 2.5) & (z > -2.5) & (z < 2.0)
    front = x[mask]
    if front.size == 0:
        return math.nan
    return float(np.min(front))


def add_header(image_bgr: np.ndarray, text: str) -> np.ndarray:
    out = image_bgr.copy()
    try:
        import cv2

        cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    except ImportError:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(out[:, :, ::-1])
        draw = ImageDraw.Draw(pil)
        draw.text((10, 10), text, fill=(255, 255, 255))
        arr = np.array(pil)
        return arr[:, :, ::-1]


def build_episode_id(spec: EpisodeSpec) -> str:
    return f"{sanitize_name(spec.condition_name)}_ep{spec.episode_index:02d}_seed{spec.seed}_{sanitize_name(spec.town)}"


def build_run_id(spec: EpisodeSpec, pipeline: str, mode: str) -> str:
    return f"{build_episode_id(spec)}_{pipeline}_{mode}"


def get_episode_summary_path(out_dirs: Dict[str, Path], spec: EpisodeSpec, pipeline: str, mode: str) -> Path:
    episode_id = build_episode_id(spec)
    return out_dirs["episode_logs"] / spec.condition_name / episode_id / f"{pipeline}_{mode}" / "episode_summary.json"


def run_episode(
    project_root: Path,
    spec: EpisodeSpec,
    pipeline: str,
    mode: str,
    cfg: dict,
    detector: YoloDetector,
    out_dirs: Dict[str, Path],
    force_run: bool,
    episode_ordinal: int,
    total_episodes: int,
    town_switched: bool,
) -> dict:
    episode_id = build_episode_id(spec)
    run_id = build_run_id(spec, pipeline, mode)
    episode_root = out_dirs["episode_logs"] / spec.condition_name / episode_id / f"{pipeline}_{mode}"
    summary_path = get_episode_summary_path(out_dirs, spec, pipeline, mode)
    if summary_path.exists() and not force_run:
        return load_json(summary_path)

    img_raw_dir = episode_root / "rgb_raw"
    img_ann_dir = episode_root / "rgb_annotated"
    lidar_clean_dir = episode_root / "lidar_clean"
    lidar_attack_dir = episode_root / "lidar_attacked"
    lidar_bev_dir = episode_root / "lidar_bev"
    logs_dir = episode_root / "logs"
    for d in [img_raw_dir, img_ann_dir, lidar_clean_dir, lidar_attack_dir, lidar_bev_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = build_logger(logs_dir / f"{run_id}.log", run_id)
    logger.info(
        "Episode start [%d/%d] condition=%s episode=%d seed=%d town=%s pipeline=%s mode=%s",
        episode_ordinal,
        total_episodes,
        spec.condition_name,
        spec.episode_index,
        spec.seed,
        spec.town,
        pipeline,
        mode,
    )

    frame_csv = logs_dir / f"{run_id}_frame_metrics.csv"
    client = world = None
    original_settings = None
    actors = []
    npc_ids = []
    frame_fp = None
    start_ts = time.time()

    try:
        carla_cfg = dict(cfg["carla"])
        carla_cfg["map"] = spec.town
        client, world = connect_to_carla(carla_cfg, logger)
        original_settings = world.get_settings()
        set_world_synchronous(world, enabled=True, fixed_delta_seconds=float(cfg["simulation"]["fixed_delta_seconds"]))
        apply_weather(world, spec.weather, logger)
        if town_switched:
            stabilize_world(
                world,
                logger,
                duration_seconds=float(cfg.get("post_map_switch_stabilize_seconds", 8.0)),
                tick_timeout_seconds=float(cfg.get("post_map_switch_tick_timeout_seconds", 2.0)),
            )

        tm_port = int(carla_cfg.get("traffic_manager_port", 8000))
        ego_cfg = cfg["ego_vehicle"]
        ego, actual_spawn_idx = spawn_ego_with_fallback(
            world,
            vehicle_filter=ego_cfg.get("blueprint_filter", "vehicle.tesla.model3"),
            preferred_spawn_index=spec.spawn_index,
            seed=spec.seed,
            logger=logger,
        )
        actors.append(ego)
        if bool(ego_cfg.get("autopilot_enabled", True)):
            ego.set_autopilot(True, tm_port)

        cam_cfg = json.loads(json.dumps(cfg["sensors"]["camera"]))
        cam_cfg["transform"] = spec.camera_transform
        camera = attach_rgb_camera(world, ego, cam_cfg, logger)
        actors.append(camera)
        camera_q = queue.Queue()
        camera.listen(camera_q.put)

        lidar = None
        lidar_q = None
        intrinsics = None
        if pipeline == "fusion":
            lidar_cfg = cfg["sensors"]["lidar"]
            lidar = attach_lidar(world, ego, lidar_cfg, logger)
            actors.append(lidar)
            lidar_q = queue.Queue()
            lidar.listen(lidar_q.put)
            intrinsics = build_camera_intrinsics(
                int(cam_cfg["image_width"]),
                int(cam_cfg["image_height"]),
                float(cam_cfg["fov"]),
            )

        collision_q = queue.Queue()
        collision_sensor = attach_collision_sensor(world, ego, collision_q, logger)
        actors.append(collision_sensor)

        target_actor = spawn_target_actor(world, spec, actual_spawn_idx, tm_port, logger)
        if target_actor is not None:
            actors.append(target_actor)

        # Spawn background NPCs after ego/target to avoid occupying the intended ego spawn.
        npc_count = max(0, spec.vehicle_count - 1)
        npc_cap = int(cfg.get("npc_spawn_cap", 9999))
        npc_count = min(npc_count, npc_cap)
        npc_ids = spawn_npc_vehicles(
            client,
            world,
            vehicle_count=npc_count,
            traffic_manager_port=tm_port,
            seed=spec.seed,
            logger=logger,
        )

        frame_fp = frame_csv.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "run_id",
            "condition_name",
            "category",
            "episode_index",
            "pipeline",
            "mode",
            "town",
            "seed",
            "timestamp",
            "frame_index",
            "sim_frame",
            "target_visible_gt",
            "target_distance_m",
            "target_detected",
            "target_confidence",
            "detection_count",
            "decision",
            "gt_should_stop",
            "decision_is_stop",
            "tp",
            "fp",
            "tn",
            "fn",
            "front_obstacle_distance_m",
            "collision_flag",
            "rgb_raw_path",
            "rgb_annotated_path",
            "lidar_clean_path",
            "lidar_attacked_path",
            "lidar_bev_path",
        ]
        writer = csv.DictWriter(frame_fp, fieldnames=fieldnames)
        writer.writeheader()

        total_frames = int(cfg["episode_frames"])
        warmup_frames = int(cfg["warmup_frames"])
        save_every_n = int(cfg["save_every_n"])
        heartbeat = int(cfg["heartbeat_interval_frames"])
        step_timeout_seconds = float(cfg.get("step_timeout_seconds", 3.0))
        episode_timeout_seconds = float(cfg.get("episode_timeout_seconds", 420.0))

        tp = fp = tn = fn = 0
        stop_correct = stop_missed = stop_false = 0
        visible_frames = missed_visible_frames = 0
        first_visible_frame = None
        first_detect_frame = None
        min_distance = math.nan
        collision_flag = 0
        frame_rows = 0
        representative_frames = []

        attack_cfg = spec.attacks if mode == "attacked" else {"camera": {"enabled": False}, "lidar": {"enabled": False}}
        logger.info(
            "Loop started frames=%d warmup=%d save_every_n=%d attack_camera=%s attack_lidar=%s",
            total_frames,
            warmup_frames,
            save_every_n,
            attack_cfg.get("camera", {}).get("enabled", False),
            attack_cfg.get("lidar", {}).get("enabled", False),
        )

        for step in range(total_frames + warmup_frames):
            if (time.time() - start_ts) > episode_timeout_seconds:
                raise TimeoutError(
                    f"Episode timeout after {episode_timeout_seconds}s. "
                    f"frames_processed={frame_rows}, condition={spec.condition_name}, pipeline={pipeline}, mode={mode}"
                )
            if pipeline == "fusion":
                rgb_data, lidar_data = get_synced_rgb_lidar(
                    world,
                    camera_q,
                    lidar_q,
                    timeout_seconds=step_timeout_seconds,
                )
                lidar_clean = lidar_measurement_to_array(lidar_data)
            else:
                rgb_data = get_synced_camera_image(world, camera_q, timeout_seconds=step_timeout_seconds)
                lidar_clean = None

            while not collision_q.empty():
                collision_q.get_nowait()
                collision_flag = 1

            if step < warmup_frames:
                continue
            if ((step - warmup_frames) % save_every_n) != 0:
                continue
            frame_index = step - warmup_frames

            rgb_clean = image_to_bgr_array(rgb_data)
            rgb_used, _ = apply_camera_attack(rgb_clean, attack_cfg.get("camera", {}))
            if pipeline == "fusion":
                lidar_used, _ = apply_lidar_attack(lidar_clean, attack_cfg.get("lidar", {}), int(rgb_data.frame))
            else:
                lidar_used = None

            detections = detector.detect(rgb_used)
            if pipeline == "fusion":
                uv, depth = project_lidar_to_image(lidar_used, camera, lidar, intrinsics)
                outputs = apply_late_fusion(detections, uv, depth, cfg["fusion"])
                decision = fusion_decision_from_outputs(outputs, lidar_used, cfg["decision"])
                detection_count = len(outputs)
                target_detected, target_confidence = is_target_detected(outputs, spec.event.get("target_type", "vehicle"))
                annotated = draw_fusion_detections(rgb_used, outputs, header_text=f"{spec.condition_name} {pipeline} {mode}")
            else:
                outputs = detections
                decision = camera_decision_from_detections(outputs)
                detection_count = len(outputs)
                target_detected, target_confidence = is_target_detected(outputs, spec.event.get("target_type", "vehicle"))
                annotated = add_header(draw_detections(rgb_used, outputs), f"{spec.condition_name} {pipeline} {mode}")

            gt_visible, target_distance = estimate_target_visibility(
                frame_idx=frame_index,
                camera_actor=camera,
                target_actor=target_actor,
                event_cfg=spec.event,
                camera_fov_degrees=float(cam_cfg["fov"]),
            )
            gt_positive = int(gt_visible)
            pred_positive = int(target_detected)
            if gt_positive and pred_positive:
                tp += 1
            elif (not gt_positive) and pred_positive:
                fp += 1
            elif (not gt_positive) and (not pred_positive):
                tn += 1
            else:
                fn += 1

            if gt_visible:
                visible_frames += 1
                if first_visible_frame is None:
                    first_visible_frame = frame_index
                if not target_detected:
                    missed_visible_frames += 1
            if target_detected and first_detect_frame is None:
                first_detect_frame = frame_index

            stop_distance = float(spec.event.get("stop_distance_m", 16.0))
            gt_should_stop = bool(gt_visible and not math.isnan(target_distance) and target_distance <= stop_distance)
            decision_is_stop = bool(str(decision).startswith("BRAKE"))
            if gt_should_stop and decision_is_stop:
                stop_correct += 1
            elif gt_should_stop and (not decision_is_stop):
                stop_missed += 1
            elif (not gt_should_stop) and decision_is_stop:
                stop_false += 1

            if pipeline == "fusion":
                front_distance = min_front_obstacle_distance(lidar_used)
            else:
                front_distance = float(target_distance) if gt_visible and not math.isnan(target_distance) else math.nan
            if not math.isnan(front_distance):
                min_distance = front_distance if math.isnan(min_distance) else min(min_distance, front_distance)

            rgb_raw_path = img_raw_dir / f"frame_{int(rgb_data.frame):06d}.png"
            rgb_ann_path = img_ann_dir / f"frame_{int(rgb_data.frame):06d}.png"
            save_bgr_image(rgb_used, rgb_raw_path)
            save_bgr_image(annotated, rgb_ann_path)

            lidar_clean_path = ""
            lidar_attack_path = ""
            lidar_bev_path = ""
            if pipeline == "fusion":
                lidar_clean_path = str(lidar_clean_dir / f"frame_{int(rgb_data.frame):06d}.npy")
                lidar_attack_path = str(lidar_attack_dir / f"frame_{int(rgb_data.frame):06d}.npy")
                np.save(lidar_clean_path, lidar_clean)
                np.save(lidar_attack_path, lidar_used)
                bev_path = lidar_bev_dir / f"frame_{int(rgb_data.frame):06d}.png"
                save_lidar_bev(lidar_used, bev_path)
                lidar_bev_path = str(bev_path)

            row = {
                "run_id": run_id,
                "condition_name": spec.condition_name,
                "category": spec.category,
                "episode_index": spec.episode_index,
                "pipeline": pipeline,
                "mode": mode,
                "town": spec.town,
                "seed": spec.seed,
                "timestamp": round(float(rgb_data.timestamp), 6),
                "frame_index": frame_index,
                "sim_frame": int(rgb_data.frame),
                "target_visible_gt": int(gt_visible),
                "target_distance_m": round(float(target_distance), 4) if not math.isnan(target_distance) else "",
                "target_detected": int(target_detected),
                "target_confidence": round(float(target_confidence), 6),
                "detection_count": detection_count,
                "decision": decision,
                "gt_should_stop": int(gt_should_stop),
                "decision_is_stop": int(decision_is_stop),
                "tp": int(gt_positive and pred_positive),
                "fp": int((not gt_positive) and pred_positive),
                "tn": int((not gt_positive) and (not pred_positive)),
                "fn": int(gt_positive and (not pred_positive)),
                "front_obstacle_distance_m": round(front_distance, 4) if not math.isnan(front_distance) else "",
                "collision_flag": collision_flag,
                "rgb_raw_path": str(rgb_raw_path),
                "rgb_annotated_path": str(rgb_ann_path),
                "lidar_clean_path": lidar_clean_path,
                "lidar_attacked_path": lidar_attack_path,
                "lidar_bev_path": lidar_bev_path,
            }
            writer.writerow(row)
            frame_rows += 1
            representative_frames.append(str(rgb_ann_path))

            if frame_rows % heartbeat == 0:
                logger.info(
                    "Heartbeat frames=%d tp=%d fp=%d fn=%d stop_missed=%d collision=%d",
                    frame_rows,
                    tp,
                    fp,
                    fn,
                    stop_missed,
                    collision_flag,
                )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        visible_miss_rate = missed_visible_frames / visible_frames if visible_frames > 0 else 0.0

        if first_visible_frame is not None and first_detect_frame is not None:
            first_detection_latency = max(0, first_detect_frame - first_visible_frame)
        else:
            first_detection_latency = -1

        gt_stop_frames = stop_correct + stop_missed
        non_stop_frames = max(1, frame_rows - gt_stop_frames)
        correct_stop_rate = stop_correct / gt_stop_frames if gt_stop_frames > 0 else 0.0
        missed_stop_rate = stop_missed / gt_stop_frames if gt_stop_frames > 0 else 0.0
        false_stop_rate = stop_false / non_stop_frames

        rep_image = representative_frames[len(representative_frames) // 2] if representative_frames else ""
        summary = {
            "run_id": run_id,
            "condition_name": spec.condition_name,
            "category": spec.category,
            "episode_index": spec.episode_index,
            "pipeline": pipeline,
            "mode": mode,
            "town": spec.town,
            "seed": spec.seed,
            "status": "completed",
            "frames_processed": frame_rows,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "false_positive_rate": round(fpr, 6),
            "false_negative_rate": round(fnr, 6),
            "first_detection_latency_frames": int(first_detection_latency),
            "visible_window_miss_rate": round(visible_miss_rate, 6),
            "correct_stop_rate": round(correct_stop_rate, 6),
            "missed_stop_rate": round(missed_stop_rate, 6),
            "false_stop_rate": round(false_stop_rate, 6),
            "min_obstacle_distance_m": round(min_distance, 4) if not math.isnan(min_distance) else "",
            "collision_flag": int(collision_flag),
            "weather": spec.weather,
            "vehicle_count": spec.vehicle_count,
            "pedestrian_count": spec.pedestrian_count,
            "frame_metrics_csv": str(frame_csv),
            "representative_screenshot": rep_image,
            "duration_seconds": round(time.time() - start_ts, 3),
        }
        save_json(summary_path, summary)
        logger.info("Episode completed frames=%d precision=%.3f recall=%.3f", frame_rows, precision, recall)
        return summary
    except Exception as exc:  # noqa: BLE001
        logger.exception("Episode failed: %s", exc)
        summary = {
            "run_id": run_id,
            "condition_name": spec.condition_name,
            "category": spec.category,
            "episode_index": spec.episode_index,
            "pipeline": pipeline,
            "mode": mode,
            "town": spec.town,
            "seed": spec.seed,
            "status": "failed",
            "error": str(exc),
            "frame_metrics_csv": str(frame_csv),
            "representative_screenshot": "",
            "duration_seconds": round(time.time() - start_ts, 3),
        }
        save_json(summary_path, summary)
        return summary
    finally:
        if frame_fp is not None:
            frame_fp.close()
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        cleanup_actors(actors, logger)
        if client is not None and npc_ids:
            cleanup_actor_ids(client, npc_ids, logger)


def build_episode_specs(cfg: dict, condition_filter: Optional[List[str]], episodes_override: Optional[int], max_conditions: Optional[int]) -> List[EpisodeSpec]:
    seed_base = int(cfg["seed_base"])
    episodes_per_condition = int(episodes_override if episodes_override is not None else cfg["episodes_per_condition"])
    excluded_towns = {str(t) for t in cfg.get("skip_towns", [])}
    towns = [t for t in cfg["towns"] if str(t) not in excluded_towns]
    if not towns:
        raise ValueError("No available towns after applying skip_towns filter.")
    spawn_pool = cfg["spawn_index_pool"]
    conds = cfg["conditions"]

    if condition_filter:
        allowed = {sanitize_name(c) for c in condition_filter}
        conds = [c for c in conds if sanitize_name(c["name"]) in allowed]
    if max_conditions is not None:
        conds = conds[: max(0, max_conditions)]

    specs = []
    for cond_idx, condition in enumerate(conds):
        condition_name = sanitize_name(condition["name"])
        category = condition["category"]
        for ep in range(episodes_per_condition):
            seed = seed_base + cond_idx * 10000 + ep
            rng = random.Random(seed)
            town = towns[(ep + cond_idx) % len(towns)]
            spawn_index = spawn_pool[(ep * 3 + cond_idx) % len(spawn_pool)]

            traffic_cfg = condition.get("traffic_override", cfg["traffic"])
            vehicle_count = rng.randint(int(traffic_cfg["vehicle_count_min"]), int(traffic_cfg["vehicle_count_max"]))
            ped_cfg = cfg["pedestrians"]
            pedestrian_count = rng.randint(int(ped_cfg["walker_count_min"]), int(ped_cfg["walker_count_max"]))

            specs.append(
                EpisodeSpec(
                    condition_name=condition_name,
                    category=category,
                    episode_index=ep,
                    seed=seed,
                    town=town,
                    spawn_index=spawn_index,
                    vehicle_count=vehicle_count,
                    pedestrian_count=pedestrian_count,
                    weather=condition["weather"],
                    camera_transform=condition["camera_transform"],
                    event=condition["event"],
                    attacks=condition["attacks"],
                )
            )
    if bool(cfg.get("group_episodes_by_town", True)):
        specs.sort(key=lambda s: (s.town, s.condition_name, s.episode_index))
    return specs


def merge_config_for_episode(cfg: dict, spec: EpisodeSpec) -> dict:
    merged = {
        "carla": cfg["carla"],
        "simulation": {
            "fixed_delta_seconds": cfg["simulation"]["fixed_delta_seconds"],
        },
        "episode_frames": int(cfg["episode_frames"]),
        "warmup_frames": int(cfg["warmup_frames"]),
        "save_every_n": int(cfg["save_every_n"]),
        "heartbeat_interval_frames": int(cfg["heartbeat_interval_frames"]),
        "ego_vehicle": cfg["ego_vehicle"],
        "sensors": cfg["sensors"],
        "detector": cfg["detector"],
        "fusion": cfg["fusion"],
        "decision": cfg["decision"],
    }
    return merged


def safe_mean(vals: List[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def safe_std(vals: List[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def aggregate_condition_metrics(episode_rows: List[dict]) -> List[dict]:
    bucket: Dict[Tuple[str, str, str], List[dict]] = {}
    for row in episode_rows:
        if row["status"] != "completed":
            continue
        key = (row["condition_name"], row["category"], row["pipeline"])
        bucket.setdefault(key, []).append(row)

    aggregates = []
    for (condition_name, category, pipeline), rows in sorted(bucket.items()):
        def vals(field):
            return [float(r[field]) for r in rows if str(r.get(field, "")) != ""]

        metric_fields = [
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
            "visible_window_miss_rate",
            "correct_stop_rate",
            "missed_stop_rate",
            "false_stop_rate",
        ]
        agg = {
            "condition_name": condition_name,
            "category": category,
            "pipeline": pipeline,
            "episodes": len(rows),
            "failed_episodes": 0,
            "first_detection_latency_mean": round(safe_mean(vals("first_detection_latency_frames")), 4),
            "first_detection_latency_std": round(safe_std(vals("first_detection_latency_frames")), 4),
            "min_obstacle_distance_mean": round(safe_mean(vals("min_obstacle_distance_m")), 4),
            "collision_rate": round(safe_mean(vals("collision_flag")), 4),
        }
        for field in metric_fields:
            agg[f"{field}_mean"] = round(safe_mean(vals(field)), 6)
            agg[f"{field}_std"] = round(safe_std(vals(field)), 6)
        aggregates.append(agg)
    return aggregates


def read_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def build_attack_pair_table(episode_rows: List[dict]) -> List[dict]:
    index = defaultdict(dict)
    for row in episode_rows:
        key = (row["condition_name"], row["episode_index"], row["pipeline"], row["town"], row["seed"])
        index[key][row["mode"]] = row

    attack_rows = []
    for key, paired in sorted(index.items()):
        if "clean" not in paired or "attacked" not in paired:
            continue
        clean = paired["clean"]
        attacked = paired["attacked"]
        if clean["category"] != "adversarial_attacks":
            continue
        clean_frames = read_csv_rows(Path(clean["frame_metrics_csv"]))
        attacked_frames = read_csv_rows(Path(attacked["frame_metrics_csv"]))
        n = min(len(clean_frames), len(attacked_frames))
        if n == 0:
            continue

        det_changed = 0
        dec_changed = 0
        conf_drop_sum = 0.0
        spoof_false_stop = 0
        attack_missed_stop = 0
        for i in range(n):
            c = clean_frames[i]
            a = attacked_frames[i]
            c_pred = int(c.get("target_detected", 0))
            a_pred = int(a.get("target_detected", 0))
            c_dec = str(c.get("decision", ""))
            a_dec = str(a.get("decision", ""))
            c_conf = float(c.get("target_confidence", 0.0))
            a_conf = float(a.get("target_confidence", 0.0))
            gt_stop = int(c.get("gt_should_stop", 0))

            if c_pred != a_pred:
                det_changed += 1
            if c_dec != a_dec:
                dec_changed += 1
            conf_drop_sum += (c_conf - a_conf)

            if (gt_stop == 0) and (not c_dec.startswith("BRAKE")) and a_dec.startswith("BRAKE"):
                spoof_false_stop += 1
            if (gt_stop == 1) and c_dec.startswith("BRAKE") and (not a_dec.startswith("BRAKE")):
                attack_missed_stop += 1

        attack_rows.append(
            {
                "condition_name": clean["condition_name"],
                "category": clean["category"],
                "episode_index": clean["episode_index"],
                "pipeline": clean["pipeline"],
                "town": clean["town"],
                "seed": clean["seed"],
                "frames_compared": n,
                "detection_change_rate": round(det_changed / n, 6),
                "decision_change_rate_under_attack": round(dec_changed / n, 6),
                "mean_confidence_drop": round(conf_drop_sum / n, 6),
                "attack_false_stop_rate": round(spoof_false_stop / n, 6),
                "attack_missed_stop_rate": round(attack_missed_stop / n, 6),
                "clean_run_id": clean["run_id"],
                "attacked_run_id": attacked["run_id"],
            }
        )
    return attack_rows


def top_failure_cases(episode_rows: List[dict], top_n: int = 20) -> List[dict]:
    completed = [r for r in episode_rows if r.get("status") == "completed"]
    scored = []
    for row in completed:
        fnr = float(row.get("false_negative_rate", 0.0))
        miss = float(row.get("missed_stop_rate", 0.0))
        coll = float(row.get("collision_flag", 0.0))
        vis = float(row.get("visible_window_miss_rate", 0.0))
        score = fnr * 0.45 + miss * 0.35 + vis * 0.15 + coll * 0.05
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for rank, (score, row) in enumerate(scored[:top_n], start=1):
        out.append(
            {
                "rank": rank,
                "failure_score": round(score, 6),
                "run_id": row["run_id"],
                "condition_name": row["condition_name"],
                "category": row["category"],
                "pipeline": row["pipeline"],
                "mode": row["mode"],
                "town": row["town"],
                "seed": row["seed"],
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "false_negative_rate": row.get("false_negative_rate", ""),
                "missed_stop_rate": row.get("missed_stop_rate", ""),
                "false_stop_rate": row.get("false_stop_rate", ""),
                "collision_flag": row.get("collision_flag", ""),
                "representative_screenshot": row.get("representative_screenshot", ""),
            }
        )
    return out


def choose_representative_screenshots(episode_rows: List[dict], out_dir: Path) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    best_by_key = {}
    for row in episode_rows:
        if row.get("status") != "completed":
            continue
        key = (row["condition_name"], row["pipeline"], row["mode"])
        score = float(row.get("false_negative_rate", 0.0)) + float(row.get("missed_stop_rate", 0.0))
        if key not in best_by_key or score > best_by_key[key][0]:
            best_by_key[key] = (score, row)

    for (condition, pipeline, mode), (_, row) in sorted(best_by_key.items()):
        src = Path(row.get("representative_screenshot", ""))
        if not src.exists():
            continue
        dst = out_dir / f"{sanitize_name(condition)}_{pipeline}_{mode}.png"
        shutil.copy2(src, dst)
        index_rows.append(
            {
                "condition_name": condition,
                "pipeline": pipeline,
                "mode": mode,
                "source": str(src),
                "selected_path": str(dst),
            }
        )
    return index_rows


def make_plots(plots_dir: Path, condition_rows: List[dict], attack_rows: List[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required. Install with `pip install matplotlib`.") from exc

    plots_dir.mkdir(parents=True, exist_ok=True)
    color_map = {"camera_only": "#d95f02", "fusion": "#1b9e77"}
    pipelines = ["camera_only", "fusion"]

    # Plot 1: precision/recall with error bars by condition.
    conditions = sorted({r["condition_name"] for r in condition_rows})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=170)
    width = 0.36
    for ax, metric in zip(axes, ["precision", "recall"]):
        for i, pipeline in enumerate(pipelines):
            vals = []
            errs = []
            for c in conditions:
                row = next((r for r in condition_rows if r["condition_name"] == c and r["pipeline"] == pipeline), None)
                vals.append(float(row[f"{metric}_mean"]) if row else 0.0)
                errs.append(float(row[f"{metric}_std"]) if row else 0.0)
            x = [k + (i - 0.5) * width for k in range(len(conditions))]
            ax.bar(x, vals, width=width, color=color_map[pipeline], label=pipeline, yerr=errs, capsize=3)
        ax.set_title(metric.title())
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.suptitle("Camera-only vs Fusion precision/recall by condition (episode mean±std)")
    fig.tight_layout()
    fig.savefig(plots_dir / "full_pipeline_precision_recall_by_condition.png")
    plt.close(fig)

    # Plot 2: missed stop / false stop rates.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=170)
    for ax, metric in zip(axes, ["missed_stop_rate", "false_stop_rate"]):
        for i, pipeline in enumerate(pipelines):
            vals = []
            errs = []
            for c in conditions:
                row = next((r for r in condition_rows if r["condition_name"] == c and r["pipeline"] == pipeline), None)
                vals.append(float(row[f"{metric}_mean"]) if row else 0.0)
                errs.append(float(row[f"{metric}_std"]) if row else 0.0)
            x = [k + (i - 0.5) * width for k in range(len(conditions))]
            ax.bar(x, vals, width=width, color=color_map[pipeline], label=pipeline, yerr=errs, capsize=3)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Rate")
    axes[0].legend()
    fig.suptitle("Per-condition missed stop / false stop rates")
    fig.tight_layout()
    fig.savefig(plots_dir / "full_pipeline_stop_error_rates_by_condition.png")
    plt.close(fig)

    # Plot 3: attack-induced decision change rate.
    if attack_rows:
        attack_conditions = sorted({r["condition_name"] for r in attack_rows})
        fig, ax = plt.subplots(figsize=(10, 5), dpi=170)
        for i, pipeline in enumerate(pipelines):
            vals = []
            for c in attack_conditions:
                subset = [r for r in attack_rows if r["condition_name"] == c and r["pipeline"] == pipeline]
                vals.append(safe_mean([float(s["decision_change_rate_under_attack"]) for s in subset]))
            x = [k + (i - 0.5) * width for k in range(len(attack_conditions))]
            ax.bar(x, vals, width=width, color=color_map[pipeline], label=pipeline)
        ax.set_title("Attack-induced decision change rate")
        ax.set_ylabel("Decision change rate")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(range(len(attack_conditions)))
        ax.set_xticklabels(attack_conditions, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "full_pipeline_attack_decision_change_rate.png")
        plt.close(fig)

    # Plot 4: first-detection latency by condition.
    fig, ax = plt.subplots(figsize=(12, 5), dpi=170)
    for i, pipeline in enumerate(pipelines):
        vals = []
        errs = []
        for c in conditions:
            row = next((r for r in condition_rows if r["condition_name"] == c and r["pipeline"] == pipeline), None)
            vals.append(float(row["first_detection_latency_mean"]) if row else 0.0)
            errs.append(float(row["first_detection_latency_std"]) if row else 0.0)
        x = [k + (i - 0.5) * width for k in range(len(conditions))]
        ax.bar(x, vals, width=width, color=color_map[pipeline], label=pipeline, yerr=errs, capsize=3)
    ax.set_title("First-detection latency by condition")
    ax.set_ylabel("Latency (frames)")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "full_pipeline_first_detection_latency_by_condition.png")
    plt.close(fig)

    # Plot 5: viewpoint side/rear comparison.
    viewpoint_conditions = ["side_view_moving_target", "rear_view_moving_target"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    for i, pipeline in enumerate(pipelines):
        vals = []
        for c in viewpoint_conditions:
            row = next((r for r in condition_rows if r["condition_name"] == c and r["pipeline"] == pipeline), None)
            vals.append(float(row["recall_mean"]) if row else 0.0)
        x = [k + (i - 0.5) * width for k in range(len(viewpoint_conditions))]
        ax.bar(x, vals, width=width, color=color_map[pipeline], label=pipeline)
    ax.set_title("Viewpoint side/rear recall comparison")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(len(viewpoint_conditions)))
    ax.set_xticklabels(viewpoint_conditions, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "full_pipeline_viewpoint_side_rear_comparison.png")
    plt.close(fig)

    # Plot 6: clean vs attacked paired results.
    if attack_rows:
        attack_conditions = sorted({r["condition_name"] for r in attack_rows})
        fig, ax = plt.subplots(figsize=(11, 5), dpi=170)
        for i, pipeline in enumerate(pipelines):
            vals = []
            for c in attack_conditions:
                subset = [r for r in attack_rows if r["condition_name"] == c and r["pipeline"] == pipeline]
                vals.append(safe_mean([float(s["detection_change_rate"]) for s in subset]))
            x = [k + (i - 0.5) * width for k in range(len(attack_conditions))]
            ax.bar(x, vals, width=width, color=color_map[pipeline], label=pipeline)
        ax.set_title("Clean vs attacked paired detection-change rate")
        ax.set_ylabel("Detection change rate")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(range(len(attack_conditions)))
        ax.set_xticklabels(attack_conditions, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "full_pipeline_clean_vs_attacked_paired.png")
        plt.close(fig)


def write_findings_and_outlines(
    report_dir: Path,
    presentation_dir: Path,
    condition_rows: List[dict],
    attack_rows: List[dict],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    presentation_dir.mkdir(parents=True, exist_ok=True)

    # Key findings markdown.
    lines = ["# Automated Episode-Based Evaluation Summary", ""]
    lines.append("## Key Findings")
    by_cat = defaultdict(list)
    for row in condition_rows:
        by_cat[row["category"]].append(row)
    for category in sorted(by_cat.keys()):
        cams = [r for r in by_cat[category] if r["pipeline"] == "camera_only"]
        fus = [r for r in by_cat[category] if r["pipeline"] == "fusion"]
        cam_recall = safe_mean([float(r["recall_mean"]) for r in cams])
        fus_recall = safe_mean([float(r["recall_mean"]) for r in fus])
        cam_miss = safe_mean([float(r["missed_stop_rate_mean"]) for r in cams])
        fus_miss = safe_mean([float(r["missed_stop_rate_mean"]) for r in fus])
        lines.append(
            f"- **{category}**: recall camera={cam_recall:.3f}, fusion={fus_recall:.3f}; "
            f"missed-stop camera={cam_miss:.3f}, fusion={fus_miss:.3f}."
        )
    if attack_rows:
        cam_dec = safe_mean([float(r["decision_change_rate_under_attack"]) for r in attack_rows if r["pipeline"] == "camera_only"])
        fus_dec = safe_mean([float(r["decision_change_rate_under_attack"]) for r in attack_rows if r["pipeline"] == "fusion"])
        lines.append(
            f"- **Attack sensitivity**: decision-change rate under attack camera={cam_dec:.3f}, fusion={fus_dec:.3f}."
        )
    lines.extend(
        [
            "",
            "## Notes on Approximations",
            "- Target visibility uses an event window plus camera-FOV/distance checks from actor transforms.",
            "- Episode-level precision/recall/FPR/FNR are computed against this visibility-ground-truth approximation.",
            "- Minimum obstacle distance uses front LiDAR points for fusion and target-distance proxy for camera-only.",
        ]
    )
    (report_dir / "main_findings.md").write_text("\n".join(lines), encoding="utf-8")

    # Presentation outline.
    presentation_lines = [
        "# Final Presentation Outline",
        "",
        "## 1. Motivation",
        "- Why robust perception under weather + attacks matters for AV safety.",
        "",
        "## 2. Hypothesis",
        "- Camera+LiDAR fusion is more robust than camera-only in degraded and adversarial settings.",
        "",
        "## 3. System Setup",
        "- CARLA packaged environment, synchronized sensors, episode-based evaluation design.",
        "",
        "## 4. Experiment Design",
        "- 10+ independent episodes per condition, matched camera/fusion seeds and parameters.",
        "- Condition families: normal, adverse weather, viewpoint, adversarial attacks.",
        "",
        "## 5. Main Results",
        "- Use plots from `outputs/plots/full_pipeline_*.png`.",
        "",
        "## 6. Failure Cases",
        "- Show representative screenshots from `outputs/representative_screenshots`.",
        "",
        "## 7. Conclusion",
        "- Summary of robustness differences and key design implications.",
    ]
    (presentation_dir / "presentation_outline.md").write_text("\n".join(presentation_lines), encoding="utf-8")

    report_outline_lines = [
        "# Final Report Outline",
        "",
        "## Abstract",
        "- One-paragraph summary of findings and contributions.",
        "",
        "## 1. Introduction and Motivation",
        "",
        "## 2. Methodology",
        "- 2.1 Pipelines (camera-only and fusion).",
        "- 2.2 Episode-based scenario design and independence strategy.",
        "- 2.3 Adversarial attack implementations.",
        "",
        "## 3. Metrics",
        "- Perception: precision/recall/FPR/FNR, first-detection latency, visible-window miss rate.",
        "- Decision: correct/missed/false stop, attack-induced decision change, min obstacle distance, collision flag.",
        "",
        "## 4. Results",
        "- Figure placement: `full_pipeline_precision_recall_by_condition.png`.",
        "- Figure placement: `full_pipeline_stop_error_rates_by_condition.png`.",
        "- Figure placement: `full_pipeline_attack_decision_change_rate.png`.",
        "- Figure placement: `full_pipeline_first_detection_latency_by_condition.png`.",
        "- Figure placement: `full_pipeline_viewpoint_side_rear_comparison.png`.",
        "- Figure placement: `full_pipeline_clean_vs_attacked_paired.png`.",
        "- Table placement: `episode_level_raw_metrics.csv`.",
        "- Table placement: `condition_level_aggregate_metrics.csv`.",
        "- Table placement: `attack_summary_table.csv`.",
        "- Table placement: `top_failure_cases.csv`.",
        "",
        "## 5. Qualitative Analysis",
        "- Representative screenshots and selected failure narratives.",
        "",
        "## 6. Limitations",
        "- Approximate visibility-ground-truth and simulator transfer constraints.",
        "",
        "## 7. Conclusion and Future Work",
    ]
    (report_dir / "report_outline.md").write_text("\n".join(report_outline_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full episode-based camera vs fusion evaluation pipeline.")
    parser.add_argument("--config", type=str, default="configs/full_pipeline_config.json")
    parser.add_argument("--episodes-per-condition", type=int, default=0, help="Override episodes per condition.")
    parser.add_argument("--max-conditions", type=int, default=0, help="Optional limit for quick runs.")
    parser.add_argument("--condition-filter", type=str, default="", help="Comma-separated condition names.")
    parser.add_argument("--skip-towns", type=str, default="", help="Comma-separated towns to skip for this run.")
    parser.add_argument("--town-switch-test", action="store_true", help="Only test map switching across towns, then exit.")
    parser.add_argument("--force", action="store_true", help="Re-run episodes even if summaries already exist.")
    parser.add_argument(
        "--resume-from-progress",
        action="store_true",
        help="Resume from existing per-episode summaries instead of re-running completed episodes.",
    )
    args = parser.parse_args()
    force_run = bool(args.force)
    if args.resume_from_progress and args.force:
        force_run = False
        print("[run_full_pipeline] --resume-from-progress overrides --force; reusing completed episode summaries.")

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return 1
    cfg = load_json(cfg_path)
    if args.skip_towns.strip():
        cfg["skip_towns"] = [s.strip() for s in args.skip_towns.split(",") if s.strip()]
    max_episode_retries = int(cfg.get("max_episode_retries", 1))
    max_consecutive_failures = int(cfg.get("max_consecutive_failures", 8))
    auto_skip_unhealthy_towns = bool(cfg.get("auto_skip_unhealthy_towns", True))

    out_dirs = ensure_output_layout(project_root)
    master_log = build_logger(out_dirs["episode_logs"] / "run_full_pipeline.log", "run_full_pipeline")
    master_log.info("Starting full pipeline config=%s", str(cfg_path))
    if not preflight_simulator(cfg["carla"], master_log):
        return 2
    if args.town_switch_test:
        towns = [t for t in cfg["towns"] if str(t) not in set(cfg.get("skip_towns", []))]
        ok, diag_rows = run_town_switch_diagnostic(cfg["carla"], towns, master_log)
        diag_csv = out_dirs["summary_tables"] / "town_switch_diagnostic.csv"
        write_csv(diag_csv, diag_rows, ["town", "status", "elapsed_seconds", "error"])
        if ok:
            master_log.info("Town switch diagnostic passed. Saved: %s", str(diag_csv))
            return 0
        master_log.error("Town switch diagnostic found failures. Saved: %s", str(diag_csv))
        return 3

    if auto_skip_unhealthy_towns:
        diag_towns = [t for t in cfg["towns"] if str(t) not in set(cfg.get("skip_towns", []))]
        _, diag_rows = run_town_switch_diagnostic(cfg["carla"], diag_towns, master_log)
        healthy = [r["town"] for r in diag_rows if r["status"] == "ok"]
        unhealthy = [r["town"] for r in diag_rows if r["status"] != "ok"]
        if unhealthy:
            master_log.warning("Auto-skipping unhealthy towns: %s", ", ".join(unhealthy))
        if healthy:
            cfg["towns"] = healthy
            master_log.info("Using healthy towns for this run: %s", ", ".join(healthy))
        else:
            master_log.error("No healthy towns available after diagnostic check.")
            return 4

    condition_filter = [sanitize_name(x.strip()) for x in args.condition_filter.split(",") if x.strip()] if args.condition_filter else None
    episodes_override = args.episodes_per_condition if args.episodes_per_condition > 0 else None
    max_conditions = args.max_conditions if args.max_conditions > 0 else None

    specs = build_episode_specs(cfg, condition_filter, episodes_override, max_conditions)
    if not specs:
        master_log.error("No episode specs resolved. Check condition filter and config.")
        return 1
    master_log.info("Resolved %d episode specs.", len(specs))
    total_episode_runs = 0
    for spec in specs:
        condition_cfg = {sanitize_name(c["name"]): c for c in cfg["conditions"]}[spec.condition_name]
        total_episode_runs += 2  # camera clean + fusion clean
        if bool(condition_cfg.get("run_paired_attack", False)):
            total_episode_runs += 2  # camera attacked + fusion attacked
    master_log.info("Planned episode runs (including paired attacks): %d", total_episode_runs)
    existing_completed = 0
    resume_completed_rows = {}
    if args.resume_from_progress:
        for spec in specs:
            condition_cfg = {sanitize_name(c["name"]): c for c in cfg["conditions"]}[spec.condition_name]
            for pipeline in ["camera_only", "fusion"]:
                clean_summary = get_episode_summary_path(out_dirs, spec, pipeline, "clean")
                if clean_summary.exists():
                    try:
                        clean_row = load_json(clean_summary)
                        if clean_row.get("status") == "completed":
                            existing_completed += 1
                            resume_completed_rows[build_run_id(spec, pipeline, "clean")] = clean_row
                    except Exception:
                        pass
                if bool(condition_cfg.get("run_paired_attack", False)):
                    attacked_summary = get_episode_summary_path(out_dirs, spec, pipeline, "attacked")
                    if attacked_summary.exists():
                        try:
                            attacked_row = load_json(attacked_summary)
                            if attacked_row.get("status") == "completed":
                                existing_completed += 1
                                resume_completed_rows[build_run_id(spec, pipeline, "attacked")] = attacked_row
                        except Exception:
                            pass
        master_log.info(
            "Resume mode enabled: found %d completed runs and %d remaining.",
            existing_completed,
            max(0, total_episode_runs - existing_completed),
        )

    detector = YoloDetector(cfg["detector"], master_log)
    episode_rows = []

    # Track per condition whether we should run attacked pairs.
    condition_cfg_by_name = {sanitize_name(c["name"]): c for c in cfg["conditions"]}
    completed_runs = existing_completed if args.resume_from_progress else 0
    failed_runs = 0
    consecutive_failures = 0
    pipeline_start = time.time()
    last_town_run = None

    for spec in specs:
        merged_cfg = merge_config_for_episode(cfg, spec)
        condition_cfg = condition_cfg_by_name[spec.condition_name]
        run_paired_attack = bool(condition_cfg.get("run_paired_attack", False))

        for pipeline in ["camera_only", "fusion"]:
            town_switched_now = (last_town_run is None) or (last_town_run != spec.town)
            clean_run_id = build_run_id(spec, pipeline, "clean")
            clean_resumed = False
            if args.resume_from_progress and clean_run_id in resume_completed_rows and not force_run:
                episode_rows.append(resume_completed_rows[clean_run_id])
                last_town_run = spec.town
                clean_resumed = True
                if run_paired_attack:
                    attacked_run_id = build_run_id(spec, pipeline, "attacked")
                    if attacked_run_id in resume_completed_rows:
                        episode_rows.append(resume_completed_rows[attacked_run_id])
                        last_town_run = spec.town
                        continue
                else:
                    continue

            if not clean_resumed:
                clean = None
                for attempt in range(max_episode_retries + 1):
                    clean = run_episode(
                        project_root=project_root,
                        spec=spec,
                        pipeline=pipeline,
                        mode="clean",
                        cfg=merged_cfg,
                        detector=detector,
                        out_dirs=out_dirs,
                        force_run=force_run,
                        episode_ordinal=completed_runs + failed_runs + 1,
                        total_episodes=total_episode_runs,
                        town_switched=town_switched_now,
                    )
                    if clean.get("status") == "completed":
                        break
                    if attempt < max_episode_retries:
                        master_log.warning(
                            "Retrying clean episode run_id=%s attempt=%d/%d",
                            clean.get("run_id"),
                            attempt + 1,
                            max_episode_retries,
                        )
                episode_rows.append(clean)
                last_town_run = spec.town
                if clean.get("status") == "completed":
                    completed_runs += 1
                    consecutive_failures = 0
                else:
                    failed_runs += 1
                    consecutive_failures += 1

                processed = completed_runs + failed_runs
                elapsed = time.time() - pipeline_start
                avg_per_run = (elapsed / processed) if processed > 0 else 0.0
                remaining = max(0, total_episode_runs - processed)
                eta_min = (avg_per_run * remaining) / 60.0 if avg_per_run > 0 else 0.0
                master_log.info(
                    "Progress %d/%d completed=%d failed=%d avg=%.1fs ETA=%.1fmin",
                    processed,
                    total_episode_runs,
                    completed_runs,
                    failed_runs,
                    avg_per_run,
                    eta_min,
                )
                if consecutive_failures >= max_consecutive_failures:
                    master_log.error(
                        "Aborting early after %d consecutive failed episode runs. "
                        "Likely simulator instability/freeze; restart CARLA and rerun.",
                        consecutive_failures,
                    )
                    break

            if run_paired_attack:
                town_switched_now = False  # same spec/town immediately after clean run
                attacked_run_id = build_run_id(spec, pipeline, "attacked")
                if args.resume_from_progress and attacked_run_id in resume_completed_rows and not force_run:
                    episode_rows.append(resume_completed_rows[attacked_run_id])
                    last_town_run = spec.town
                    continue
                attacked = None
                for attempt in range(max_episode_retries + 1):
                    attacked = run_episode(
                        project_root=project_root,
                        spec=spec,
                        pipeline=pipeline,
                        mode="attacked",
                        cfg=merged_cfg,
                        detector=detector,
                        out_dirs=out_dirs,
                        force_run=force_run,
                        episode_ordinal=completed_runs + failed_runs + 1,
                        total_episodes=total_episode_runs,
                        town_switched=town_switched_now,
                    )
                    if attacked.get("status") == "completed":
                        break
                    if attempt < max_episode_retries:
                        master_log.warning(
                            "Retrying attacked episode run_id=%s attempt=%d/%d",
                            attacked.get("run_id"),
                            attempt + 1,
                            max_episode_retries,
                        )
                episode_rows.append(attacked)
                if attacked.get("status") == "completed":
                    completed_runs += 1
                    consecutive_failures = 0
                else:
                    failed_runs += 1
                    consecutive_failures += 1

                processed = completed_runs + failed_runs
                elapsed = time.time() - pipeline_start
                avg_per_run = (elapsed / processed) if processed > 0 else 0.0
                remaining = max(0, total_episode_runs - processed)
                eta_min = (avg_per_run * remaining) / 60.0 if avg_per_run > 0 else 0.0
                master_log.info(
                    "Progress %d/%d completed=%d failed=%d avg=%.1fs ETA=%.1fmin",
                    processed,
                    total_episode_runs,
                    completed_runs,
                    failed_runs,
                    avg_per_run,
                    eta_min,
                )
                if consecutive_failures >= max_consecutive_failures:
                    master_log.error(
                        "Aborting early after %d consecutive failed episode runs. "
                        "Likely simulator instability/freeze; restart CARLA and rerun.",
                        consecutive_failures,
                    )
                    break
        if consecutive_failures >= max_consecutive_failures:
            break

    # Save episode-level raw table.
    episode_fieldnames = [
        "run_id",
        "condition_name",
        "category",
        "episode_index",
        "pipeline",
        "mode",
        "town",
        "seed",
        "status",
        "frames_processed",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
        "first_detection_latency_frames",
        "visible_window_miss_rate",
        "correct_stop_rate",
        "missed_stop_rate",
        "false_stop_rate",
        "min_obstacle_distance_m",
        "collision_flag",
        "vehicle_count",
        "pedestrian_count",
        "frame_metrics_csv",
        "representative_screenshot",
        "duration_seconds",
    ]
    write_csv(out_dirs["summary_tables"] / "episode_level_raw_metrics.csv", episode_rows, episode_fieldnames)

    # Aggregates and attack pair table.
    condition_rows = aggregate_condition_metrics(episode_rows)
    write_csv(
        out_dirs["summary_tables"] / "condition_level_aggregate_metrics.csv",
        condition_rows,
        [
            "condition_name",
            "category",
            "pipeline",
            "episodes",
            "failed_episodes",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "false_positive_rate_mean",
            "false_positive_rate_std",
            "false_negative_rate_mean",
            "false_negative_rate_std",
            "first_detection_latency_mean",
            "first_detection_latency_std",
            "visible_window_miss_rate_mean",
            "visible_window_miss_rate_std",
            "correct_stop_rate_mean",
            "correct_stop_rate_std",
            "missed_stop_rate_mean",
            "missed_stop_rate_std",
            "false_stop_rate_mean",
            "false_stop_rate_std",
            "min_obstacle_distance_mean",
            "collision_rate",
        ],
    )

    attack_rows = build_attack_pair_table(episode_rows)
    write_csv(
        out_dirs["summary_tables"] / "attack_summary_table.csv",
        attack_rows,
        [
            "condition_name",
            "category",
            "episode_index",
            "pipeline",
            "town",
            "seed",
            "frames_compared",
            "detection_change_rate",
            "decision_change_rate_under_attack",
            "mean_confidence_drop",
            "attack_false_stop_rate",
            "attack_missed_stop_rate",
            "clean_run_id",
            "attacked_run_id",
        ],
    )

    failures = top_failure_cases(episode_rows, top_n=20)
    write_csv(
        out_dirs["summary_tables"] / "top_failure_cases.csv",
        failures,
        [
            "rank",
            "failure_score",
            "run_id",
            "condition_name",
            "category",
            "pipeline",
            "mode",
            "town",
            "seed",
            "precision",
            "recall",
            "false_negative_rate",
            "missed_stop_rate",
            "false_stop_rate",
            "collision_flag",
            "representative_screenshot",
        ],
    )

    # Plots + screenshots + markdown assets.
    make_plots(out_dirs["plots"], condition_rows, attack_rows)
    shot_index = choose_representative_screenshots(episode_rows, out_dirs["representative_screenshots"])
    write_csv(
        out_dirs["summary_tables"] / "representative_screenshots_index.csv",
        shot_index,
        ["condition_name", "pipeline", "mode", "source", "selected_path"],
    )

    write_findings_and_outlines(
        report_dir=out_dirs["final_report_assets"],
        presentation_dir=out_dirs["final_presentation_assets"],
        condition_rows=condition_rows,
        attack_rows=attack_rows,
    )

    # Also copy key files into final assets folders for convenience.
    for filename in [
        "condition_level_aggregate_metrics.csv",
        "attack_summary_table.csv",
        "top_failure_cases.csv",
    ]:
        src = out_dirs["summary_tables"] / filename
        if src.exists():
            shutil.copy2(src, out_dirs["final_report_assets"] / filename)
    for plot_file in out_dirs["plots"].glob("full_pipeline_*.png"):
        shutil.copy2(plot_file, out_dirs["final_report_assets"] / plot_file.name)
        shutil.copy2(plot_file, out_dirs["final_presentation_assets"] / plot_file.name)

    findings_src = out_dirs["final_report_assets"] / "main_findings.md"
    if findings_src.exists():
        shutil.copy2(findings_src, out_dirs["final_presentation_assets"] / "main_findings.md")

    master_log.info("Full pipeline completed.")
    master_log.info("Episode raw metrics: %s", str(out_dirs["summary_tables"] / "episode_level_raw_metrics.csv"))
    master_log.info("Condition aggregates: %s", str(out_dirs["summary_tables"] / "condition_level_aggregate_metrics.csv"))
    master_log.info("Attack summary: %s", str(out_dirs["summary_tables"] / "attack_summary_table.csv"))
    master_log.info("Top failures: %s", str(out_dirs["summary_tables"] / "top_failure_cases.csv"))
    master_log.info("Plots directory: %s", str(out_dirs["plots"]))
    master_log.info("Representative screenshots: %s", str(out_dirs["representative_screenshots"]))
    master_log.info("Final report assets: %s", str(out_dirs["final_report_assets"]))
    master_log.info("Final presentation assets: %s", str(out_dirs["final_presentation_assets"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
