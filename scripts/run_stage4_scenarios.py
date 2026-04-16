import argparse
import csv
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def run_pipeline(script_path: Path, config_path: Path, run_id: str, project_root: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--run-id",
        run_id,
    ]
    return subprocess.run(cmd, cwd=str(project_root), check=False, capture_output=True, text=True)


def run_pipeline_with_retries(
    script_path: Path,
    config_path: Path,
    run_id: str,
    project_root: Path,
    max_retries: int,
):
    attempt = 0
    last_proc = None
    while attempt <= max_retries:
        last_proc = run_pipeline(script_path, config_path, run_id, project_root)
        if last_proc.returncode == 0:
            return last_proc
        attempt += 1
        if attempt <= max_retries:
            print(
                f"Retrying {script_path.name} for run_id={run_id}. "
                f"attempt={attempt}/{max_retries}"
            )
    return last_proc


def find_representative_image(directory: Path) -> Path:
    images = sorted(directory.glob("*.png"))
    if not images:
        return None
    return images[len(images) // 2]


def copy_representative_image(source_image: Path, target_path: Path) -> str:
    if source_image is None or not source_image.exists():
        return ""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, target_path)
    return str(target_path)


def parse_float(val, default=0.0):
    try:
        return float(val)
    except Exception:  # noqa: BLE001
        return float(default)


def summarize_frame_metrics(frame_metrics_csv: Path) -> dict:
    summary = {
        "frames": 0,
        "mean_detection_count": 0.0,
        "missed_detection_frames": 0,
        "unstable_classification_frames": 0,
    }
    if frame_metrics_csv is None:
        return summary
    try:
        if str(frame_metrics_csv).strip() in ("", "."):
            return summary
    except Exception:  # noqa: BLE001
        return summary
    if not frame_metrics_csv.exists() or not frame_metrics_csv.is_file():
        return summary

    total_detections = 0.0
    with frame_metrics_csv.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            summary["frames"] += 1
            if "detection_count" in row:
                total_detections += parse_float(row.get("detection_count", 0.0))
            else:
                total_detections += parse_float(row.get("fused_detection_count", 0.0))
            summary["missed_detection_frames"] += int(parse_float(row.get("missed_detection_flag", 0), 0))
            summary["unstable_classification_frames"] += int(
                parse_float(row.get("unstable_classification_flag", 0), 0)
            )
    if summary["frames"] > 0:
        summary["mean_detection_count"] = total_detections / summary["frames"]
    return summary


def extract_weather_string(weather_cfg: dict) -> str:
    if "preset" in weather_cfg:
        return str(weather_cfg["preset"])
    return json.dumps(weather_cfg, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4 robustness sweep for camera-only and fusion baselines."
    )
    parser.add_argument(
        "--scenario-config",
        type=str,
        default="configs/stage4_scenarios.json",
        help="Path to stage4 scenario definition file.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scenario_cfg_path = Path(args.scenario_config).resolve()
    if not scenario_cfg_path.exists():
        print(f"Scenario config not found: {scenario_cfg_path}")
        return 1
    scenario_cfg = load_json(scenario_cfg_path)
    max_retries = int(scenario_cfg.get("pipeline_retry_count", 1))

    base_camera_cfg = load_json((project_root / scenario_cfg["base_camera_config"]).resolve())
    base_fusion_cfg = load_json((project_root / scenario_cfg["base_fusion_config"]).resolve())
    global_overrides = scenario_cfg.get("global_overrides", {})

    report_root = project_root / "outputs" / "stage4_report"
    generated_cfg_root = report_root / "generated_configs"
    screenshots_root = report_root / "representative_screenshots"
    report_root.mkdir(parents=True, exist_ok=True)
    generated_cfg_root.mkdir(parents=True, exist_ok=True)
    screenshots_root.mkdir(parents=True, exist_ok=True)

    camera_script = project_root / "scripts" / "run_camera_baseline.py"
    fusion_script = project_root / "scripts" / "run_fusion_baseline.py"
    summary_csv = report_root / "summary.csv"

    summary_rows = []
    for scenario in scenario_cfg.get("scenarios", []):
        scenario_name = sanitize_name(scenario["name"])
        print(f"\n[Stage4] Running scenario: {scenario_name}")

        camera_cfg = deepcopy(base_camera_cfg)
        fusion_cfg = deepcopy(base_fusion_cfg)
        deep_update(camera_cfg, deepcopy(global_overrides))
        deep_update(fusion_cfg, deepcopy(global_overrides))

        camera_cfg["scenario"] = {
            "name": scenario_name,
            "stability_jaccard_threshold": global_overrides.get("scenario", {}).get(
                "stability_jaccard_threshold", 0.3
            ),
            "notes": scenario.get("notes", ""),
        }
        fusion_cfg["scenario"] = deepcopy(camera_cfg["scenario"])

        if "weather" in scenario:
            camera_cfg["weather"] = deepcopy(scenario["weather"])
            fusion_cfg["weather"] = deepcopy(scenario["weather"])
        if "traffic" in scenario:
            camera_cfg["traffic"] = deepcopy(scenario["traffic"])
            fusion_cfg["traffic"] = deepcopy(scenario["traffic"])
        if "pedestrians" in scenario:
            camera_cfg["pedestrians"] = deepcopy(scenario["pedestrians"])
            fusion_cfg["pedestrians"] = deepcopy(scenario["pedestrians"])
        if "camera_transform" in scenario:
            camera_cfg["sensors"]["camera"]["transform"] = deepcopy(scenario["camera_transform"])
            fusion_cfg["sensors"]["camera"]["transform"] = deepcopy(scenario["camera_transform"])

        camera_cfg["experiment_name"] = f"stage4_camera_{scenario_name}"
        fusion_cfg["experiment_name"] = f"stage4_fusion_{scenario_name}"

        camera_cfg_path = generated_cfg_root / f"{scenario_name}_camera.json"
        fusion_cfg_path = generated_cfg_root / f"{scenario_name}_fusion.json"
        save_json(camera_cfg_path, camera_cfg)
        save_json(fusion_cfg_path, fusion_cfg)

        camera_run_id = f"stage4_camera_{scenario_name}"
        fusion_run_id = f"stage4_fusion_{scenario_name}"

        camera_proc = run_pipeline_with_retries(
            camera_script,
            camera_cfg_path,
            camera_run_id,
            project_root,
            max_retries=max_retries,
        )
        print(camera_proc.stdout)
        if camera_proc.returncode != 0:
            print(camera_proc.stderr)

        fusion_proc = run_pipeline_with_retries(
            fusion_script,
            fusion_cfg_path,
            fusion_run_id,
            project_root,
            max_retries=max_retries,
        )
        print(fusion_proc.stdout)
        if fusion_proc.returncode != 0:
            print(fusion_proc.stderr)

        camera_metrics_path = project_root / "outputs" / "metrics" / f"{camera_run_id}_stage2_camera_baseline.json"
        fusion_metrics_path = project_root / "outputs" / "metrics" / f"{fusion_run_id}_stage3_fusion_baseline.json"
        camera_metrics = load_json(camera_metrics_path) if camera_metrics_path.exists() else {}
        fusion_metrics = load_json(fusion_metrics_path) if fusion_metrics_path.exists() else {}

        camera_frame_metrics_value = camera_metrics.get("frame_metrics_csv", "")
        fusion_frame_metrics_value = fusion_metrics.get("frame_metrics_csv", "")
        camera_frame_metrics = Path(camera_frame_metrics_value) if camera_frame_metrics_value else None
        fusion_frame_metrics = Path(fusion_frame_metrics_value) if fusion_frame_metrics_value else None
        camera_frame_summary = summarize_frame_metrics(camera_frame_metrics)
        fusion_frame_summary = summarize_frame_metrics(fusion_frame_metrics)

        camera_ann_dir = Path(camera_metrics.get("annotated_frames_dir", ""))
        fusion_ann_dir = Path(fusion_metrics.get("output_root", "")) / "fusion_annotated"
        camera_rep = find_representative_image(camera_ann_dir) if camera_ann_dir.exists() else None
        fusion_rep = find_representative_image(fusion_ann_dir) if fusion_ann_dir.exists() else None
        camera_rep_out = copy_representative_image(
            camera_rep,
            screenshots_root / f"{scenario_name}_camera.png",
        )
        fusion_rep_out = copy_representative_image(
            fusion_rep,
            screenshots_root / f"{scenario_name}_fusion.png",
        )

        row_base = {
            "scenario_name": scenario_name,
            "weather_conditions": extract_weather_string(scenario.get("weather", {})),
            "vehicle_count": int(scenario.get("traffic", {}).get("vehicle_count", 0)),
            "pedestrian_count": int(scenario.get("pedestrians", {}).get("walker_count", 0)),
        }
        summary_rows.append(
            {
                **row_base,
                "pipeline": "camera_only",
                "status": camera_metrics.get("status", "missing"),
                "frames": camera_frame_summary["frames"],
                "mean_detection_count": round(camera_frame_summary["mean_detection_count"], 4),
                "missed_detection_frames": camera_frame_summary["missed_detection_frames"],
                "unstable_classification_frames": camera_frame_summary["unstable_classification_frames"],
                "frame_metrics_csv": str(camera_frame_metrics) if camera_frame_metrics is not None else "",
                "representative_screenshot": camera_rep_out,
            }
        )
        summary_rows.append(
            {
                **row_base,
                "pipeline": "camera_lidar_fusion",
                "status": fusion_metrics.get("status", "missing"),
                "frames": fusion_frame_summary["frames"],
                "mean_detection_count": round(fusion_frame_summary["mean_detection_count"], 4),
                "missed_detection_frames": fusion_frame_summary["missed_detection_frames"],
                "unstable_classification_frames": fusion_frame_summary["unstable_classification_frames"],
                "frame_metrics_csv": str(fusion_frame_metrics) if fusion_frame_metrics is not None else "",
                "representative_screenshot": fusion_rep_out,
            }
        )

    fieldnames = [
        "scenario_name",
        "pipeline",
        "status",
        "weather_conditions",
        "vehicle_count",
        "pedestrian_count",
        "frames",
        "mean_detection_count",
        "missed_detection_frames",
        "unstable_classification_frames",
        "frame_metrics_csv",
        "representative_screenshot",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\nStage 4 summary saved: {summary_csv}")
    print(f"Representative screenshots: {screenshots_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
