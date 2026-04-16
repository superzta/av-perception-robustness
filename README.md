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
3. Install minimal Python dependencies in the venv:
   - `pip install pillow`
4. Run one experiment:
   - `python scripts\run_experiment.py --config configs\experiment_template.json`

Outputs are written to:
- `outputs\logs`
- `outputs\images`
- `outputs\metrics`

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
- Set `sensors.camera.enabled=true`, `sensors.lidar.enabled=false`.
- Run baseline scenarios with clear weather first.
- Save detection and behavior metrics to `outputs\metrics`.

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
