# Robust Perception for Autonomous Vehicles

Camera-only vs sensor-fusion perception robustness experiments in CARLA under adversarial and environmental stress.

## Project Structure

- `scripts/` Python experiment entrypoints and utilities
- `configs/` JSON config templates
- `outputs/` Logs, metrics, images, plots, tables, and videos from runs
- `models/` Model checkpoints and inference assets
- `docs/` Notes, report drafts, and presentation material

## Quick Start

1. Start CARLA server separately (already packaged at `D:\18662_Project\carla`).
2. Activate your virtual environment:
   - PowerShell: `D:\18662_Project\venv\Scripts\Activate.ps1`
3. Install dependencies in the venv:
   - `pip install pillow numpy`
4. Run Stage 1 smoke test:
   - `python scripts\run_experiment.py --config configs\experiment_template.json`
5. Run Stage 2 camera-only baseline (YOLO):
   - `pip install ultralytics opencv-python`
   - `python scripts\run_camera_baseline.py --config configs\stage2_camera_daylight.json`
6. Run Stage 2 post-processing:
   - `pip install matplotlib` (optional, only needed for plots)
   - `python scripts\postprocess_stage2.py`

Outputs are written to:
- `outputs\logs`
- `outputs\images`
- `outputs\metrics`

For Stage 2, frame outputs are organized as:
- `outputs\images\raw\<run_id>\`
- `outputs\images\annotated\<run_id>\`
- `outputs\logs\<run_id>_detections.csv`
- `outputs\logs\<run_id>_detections.jsonl`
- `outputs\tables\<run_id>_counts_by_class.csv`
- `outputs\tables\<run_id>_counts_by_frame.csv`
- `outputs\plots\<run_id>_class_counts.png`

YOLO weight path is set to:
- `models\weights\yolo\yolov8n.pt`

## Config Fields

The runner reads one JSON file with:
- `carla`: host/port/timeout/map
- `weather`: preset or custom weather values
- `sensors`: camera and lidar settings
- `traffic`: vehicle density controls
- `pedestrians`: pedestrian density controls
- `attacks`: attack toggles and parameters

## 6-Stage Execution Guide

### 1) Environment freeze and repo structure
- Freeze environment versions (`pip freeze > requirements.txt`).
- Keep this folder layout stable for reproducibility.
- Confirm CARLA server launch and Python client connectivity.

### 2) Baseline camera-only pipeline
- Use `configs\stage2_camera_daylight.json` for repeatable daylight baseline.
- Run `python scripts\run_camera_baseline.py --config configs\stage2_camera_daylight.json`.
- This script connects CARLA, spawns ego + front camera, captures synchronized frames, runs YOLO, saves raw and annotated images, and writes per-frame detection logs.
- Run `python scripts\postprocess_stage2.py` to generate summary tables and a quick class-count plot.

### 3) Multimodal data capture and basic fusion
- Enable lidar in config and add your fusion model code under `scripts/`.
- Capture synchronized camera + LiDAR data.
- Compare camera-only vs fusion on identical seeds/scenarios.

### 4) Scenario engine and robustness
- Add scripted weather/time/occlusion variations.
- Sweep traffic and pedestrian levels via config files.
- Store per-scenario metrics and aggregate summaries.

### 5) Adversarial attack implementation
- Enable and parameterize attack settings in `attacks`.
- Add patch, spoofing, and blinding logic modules incrementally.
- Track both perception failures and downstream driving impact.

### 6) Evaluation, visualization, report/presentation
- Generate comparative plots/tables in `outputs\plots` and `outputs\tables`.
- Export scenario videos to `outputs\videos`.
- Summarize findings in `docs\` for report and slides.

## Notes

- Current runner is intentionally minimal and reproducible.
- It connects to CARLA, applies map/weather, spawns one ego camera, captures one frame, and saves run metrics.
- Traffic, pedestrian simulation, and attack behavior are represented in config and logging first, then expanded in later stages.
