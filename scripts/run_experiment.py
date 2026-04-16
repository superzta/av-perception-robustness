import argparse
import json
import logging
import sys
import time
from pathlib import Path

from utils.carla_utils import (
    apply_weather,
    capture_single_image,
    cleanup_actors,
    connect_to_carla,
    spawn_camera_on_vehicle,
)
from utils.io_utils import (
    annotate_image,
    append_metrics_csv,
    ensure_directories,
    make_run_id,
    save_metrics_json,
)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("experiment_runner")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one CARLA perception experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_template.json",
        help="Path to JSON experiment config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    config = load_config(config_path)
    run_id = make_run_id(config.get("experiment_name", "experiment"))

    output_dirs = ensure_directories(
        {
            "logs": "outputs/logs",
            "images": "outputs/images",
            "metrics": "outputs/metrics",
            "plots": "outputs/plots",
            "tables": "outputs/tables",
            "videos": "outputs/videos",
        }
    )

    log_path = output_dirs["logs"] / f"{run_id}.log"
    logger = build_logger(log_path)
    logger.info("Starting run_id=%s using config=%s", run_id, str(config_path))

    metrics = {
        "run_id": run_id,
        "experiment_name": config.get("experiment_name", "unknown"),
        "map": config.get("carla", {}).get("map", "Town03"),
        "traffic_level": config.get("traffic", {}).get("vehicle_count", 0),
        "pedestrian_level": config.get("pedestrians", {}).get("walker_count", 0),
        "attack_settings": config.get("attacks", {}),
        "status": "started",
    }

    actors = []
    start_time = time.time()

    try:
        client, world = connect_to_carla(config["carla"], logger)
        apply_weather(world, config.get("weather", {}), logger)

        sensor_cfg = config.get("sensors", {}).get("camera", {})
        if sensor_cfg.get("enabled", True):
            vehicle, camera = spawn_camera_on_vehicle(world, sensor_cfg, logger)
            actors.extend([camera, vehicle])

            image_filename = output_dirs["images"] / f"{run_id}_camera.png"
            camera_image = capture_single_image(world, camera, timeout_seconds=5.0)
            camera_image.save_to_disk(str(image_filename))

            annotated_filename = output_dirs["images"] / f"{run_id}_camera_annotated.png"
            annotate_image(
                image_path=image_filename,
                output_path=annotated_filename,
                text_lines=[
                    f"run_id: {run_id}",
                    f"map: {metrics['map']}",
                    f"weather: {config.get('weather', {}).get('preset', 'custom')}",
                ],
                logger=logger,
            )
            metrics["image_path"] = str(annotated_filename)

        metrics["status"] = "completed"

    except Exception as exc:  # noqa: BLE001
        logger.exception("Experiment failed: %s", exc)
        metrics["status"] = "failed"
        metrics["error"] = str(exc)
        return_code = 1
    else:
        return_code = 0
    finally:
        cleanup_actors(actors, logger)
        metrics["duration_seconds"] = round(time.time() - start_time, 3)
        metrics_json = output_dirs["metrics"] / f"{run_id}.json"
        metrics_csv = output_dirs["metrics"] / "runs.csv"
        save_metrics_json(metrics, metrics_json)
        append_metrics_csv(metrics, metrics_csv)
        logger.info("Saved metrics to %s", str(metrics_json))
        logger.info("Run finished with status=%s", metrics["status"])

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
