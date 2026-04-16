# Robust Perception for Autonomous Vehicles

Comparative study of camera-only and camera+LiDAR fusion perception pipelines in CARLA under environmental degradation and adversarial perturbations.

This repository is designed to be reproducible, presentation-ready, and practical for course research: one command can run the full episode-based workflow and generate figures, tables, logs, and report assets.

## Research Scope

This project evaluates two perception stacks:

- `camera_only`: RGB + YOLO-based object detection
- `fusion`: RGB + LiDAR late-fusion baseline (LiDAR confirmation of visual detections)

Across four stress families:

- normal visibility
- adverse weather and low visibility
- viewpoint variation (front/side/rear/occlusion-style setups)
- adversarial attacks (camera glare/patch-style effects and LiDAR spoofing)

## Repository Layout

- `scripts/` experiment runners and utilities
- `configs/` reproducible JSON experiment configs
- `models/` local model weights and checkpoints (git-ignored)
- `outputs/` all runtime artifacts (logs, images, tables, plots, assets)
- `docs/` report notes and supporting documentation

## Environment

- OS: Windows 11
- Simulator: packaged CARLA at `D:\18662_Project\carla`
- Python venv: `D:\18662_Project\venv`

Activate environment:

- PowerShell: `D:\18662_Project\venv\Scripts\Activate.ps1`

Install core dependencies:

- `pip install numpy pillow matplotlib opencv-python ultralytics`

YOLO default weights location:

- `models\weights\yolo\yolov8n.pt`

## Quick Start

1. Start CARLA server separately.
2. Activate the virtual environment.
3. Run a baseline pipeline or run the full workflow.

Camera baseline:

- `python scripts\run_camera_baseline.py --config configs\stage2_camera_daylight.json`

Fusion baseline:

- `python scripts\run_fusion_baseline.py --config configs\stage3_fusion_daylight.json`

Full episode-based pipeline (recommended):

- `python scripts\run_full_pipeline.py`

Resume interrupted run:

- `python scripts\run_full_pipeline.py --resume-from-progress`

Fast smoke test:

- `python scripts\run_full_pipeline.py --episodes-per-condition 1 --max-conditions 2`

Town-switch diagnostic only:

- `python scripts\run_full_pipeline.py --town-switch-test`

## Stage-by-Stage Commands

1) Environment and connectivity smoke test:

- `python scripts\run_experiment.py --config configs\experiment_template.json`

2) Camera-only baseline + postprocess:

- `python scripts\run_camera_baseline.py --config configs\stage2_camera_daylight.json`
- `python scripts\postprocess_stage2.py`

3) RGB+LiDAR fusion baseline:

- `python scripts\run_fusion_baseline.py --config configs\stage3_fusion_daylight.json`

4) Controlled robustness scenarios:

- `python scripts\run_stage4_scenarios.py --scenario-config configs\stage4_scenarios.json`

5) Adversarial attack evaluation:

- Camera attack example: `python scripts\run_attack_evaluation.py --pipeline camera_only --config configs\stage5_camera_attack.json`
- LiDAR spoof example: `python scripts\run_attack_evaluation.py --pipeline fusion --config configs\stage5_lidar_spoof_attack.json`

6) Evaluation/report assets:

- `python scripts\generate_stage6_assets.py`

## Full-Pipeline Outputs

The automated workflow writes final artifacts to:

- `outputs\episode_logs\`
- `outputs\summary_tables\`
- `outputs\plots\`
- `outputs\representative_screenshots\`
- `outputs\final_report_assets\`
- `outputs\final_presentation_assets\`

Key generated files include:

- `outputs\summary_tables\episode_level_raw_metrics.csv`
- `outputs\summary_tables\condition_level_aggregate_metrics.csv`
- `outputs\summary_tables\attack_summary_table.csv`
- `outputs\summary_tables\top_failure_cases.csv`
- `outputs\plots\full_pipeline_*.png`
- `outputs\final_report_assets\main_findings.md`
- `outputs\final_presentation_assets\presentation_outline.md`

## Configuration Overview

Main config fields (see `configs\full_pipeline_config.json`):

- `carla`: server host/port/timeouts
- `towns`, `skip_towns`: town selection and exclusions
- `conditions`: scenario families and attack pair settings
- `episodes_per_condition`, `seed_base`: experiment scale and reproducibility
- `sensors`: camera and LiDAR parameters
- `traffic`, `pedestrians`: scene complexity controls
- `attacks`: camera and LiDAR perturbation settings

## Reproducibility and Reliability Features

- deterministic episode specification via seeded config
- per-episode summaries for resume support
- simulator preflight checks before long runs
- optional town health diagnostics
- map-switch stabilization delay
- episode retries and consecutive-failure stop guard
- heartbeat logging during long-running episodes

## Notes for GitHub Portfolio

This repository intentionally prioritizes:

- clear experiment design over overly complex abstractions
- practical attack/fusion baselines suitable for academic projects
- reproducible output structure for report and slide generation

If you are reviewing this project, start with:

1. `configs\full_pipeline_config.json`
2. `scripts\run_full_pipeline.py`
3. `outputs\summary_tables\` and `outputs\plots\` after a run
