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
7. Run Stage 3 camera+LiDAR fusion baseline:
   - `python scripts\run_fusion_baseline.py --config configs\stage3_fusion_daylight.json`
8. Run Stage 4 controlled robustness sweep (camera-only + fusion):
   - `python scripts\run_stage4_scenarios.py --scenario-config configs\stage4_scenarios.json`
9. Run Stage 5 adversarial attack evaluation (camera-only or fusion):
   - Camera attack example: `python scripts\run_attack_evaluation.py --pipeline camera_only --config configs\stage5_camera_attack.json`
   - LiDAR spoofing example: `python scripts\run_attack_evaluation.py --pipeline fusion --config configs\stage5_lidar_spoof_attack.json`
10. Run Stage 6 evaluation + report asset generation:
   - `python scripts\generate_stage6_assets.py`
11. Run full automated episode-based workflow (recommended final command):
   - `python scripts\run_full_pipeline.py`
   - Optional quick smoke test: `python scripts\run_full_pipeline.py --episodes-per-condition 1 --max-conditions 2`
   - Resume after crash/interruption without restarting completed episodes: `python scripts\run_full_pipeline.py --resume-from-progress`
   - The script now performs simulator preflight and will fail fast if CARLA is not reachable.
   - You can skip unstable maps: `python scripts\run_full_pipeline.py --skip-towns Town04`
   - You can run map-switch diagnostics only: `python scripts\run_full_pipeline.py --town-switch-test`
   - The pipeline now adds a short stabilization phase after town changes to reduce simulator freeze risk.
   - The pipeline can auto-skip unhealthy towns after a real streaming/sensor health check (`auto_skip_unhealthy_towns` in config).

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

For Stage 3, synchronized fusion outputs are organized as:
- `outputs\fusion_baseline\<run_id>\rgb_raw\frame_<id>.png`
- `outputs\fusion_baseline\<run_id>\lidar_points\frame_<id>.npy`
- `outputs\fusion_baseline\<run_id>\fusion_annotated\frame_<id>.png`
- `outputs\fusion_baseline\<run_id>\logs\<run_id>_fusion_detections.csv`
- `outputs\fusion_baseline\<run_id>\logs\<run_id>_fusion_detections.jsonl`

For Stage 4, report-ready comparison outputs are organized as:
- `outputs\stage4_report\summary.csv`
- `outputs\stage4_report\representative_screenshots\<scenario>_camera.png`
- `outputs\stage4_report\representative_screenshots\<scenario>_fusion.png`
- `outputs\stage4_report\generated_configs\`

For Stage 5 attack outputs are organized as:
- `outputs\attacks\<pipeline>\<run_id>\rgb_clean\`
- `outputs\attacks\<pipeline>\<run_id>\rgb_attacked\`
- `outputs\attacks\<pipeline>\<run_id>\annotated_clean\`
- `outputs\attacks\<pipeline>\<run_id>\annotated_attacked\`
- `outputs\attacks\<pipeline>\<run_id>\lidar_clean\` and `lidar_attacked\` (fusion)
- `outputs\attacks\<pipeline>\<run_id>\lidar_bev_clean\` and `lidar_bev_attacked\` (fusion)
- `outputs\attacks\<pipeline>\<run_id>\logs\<run_id>_frame_attack_log.csv`
- `outputs\attacks\<pipeline>\<run_id>\logs\<run_id>.log`

For Stage 6 evaluation/report assets are organized as:
- `outputs\tables\stage6_scenario_metrics.csv`
- `outputs\tables\stage6_attack_metrics.csv`
- `outputs\tables\stage6_condition_metrics.csv`
- `outputs\tables\stage6_presentation_outline.md`
- `outputs\tables\stage6_report_outline.md`
- `outputs\plots\stage6_mean_detection_by_scenario.png`
- `outputs\plots\stage6_precision_recall_by_condition.png`
- `outputs\plots\stage6_attack_decision_change_rate.png`
- `outputs\plots\stage6_representative_screenshots\`

For the automated episode-based workflow, outputs are organized as:
- `outputs\episode_logs\`
- `outputs\summary_tables\`
- `outputs\plots\`
- `outputs\representative_screenshots\`
- `outputs\final_report_assets\`
- `outputs\final_presentation_assets\`

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
- Use `python scripts\run_fusion_baseline.py --config configs\stage3_fusion_daylight.json`.
- This baseline uses practical late fusion (YOLO + LiDAR confirmation in each RGB box) for stable Windows execution.
- For direct comparison, keep the same map/weather/simulation/traffic/seed values between `stage2` and `stage3` configs.

### 4) Scenario engine and robustness
- Run `python scripts\run_stage4_scenarios.py --scenario-config configs\stage4_scenarios.json`.
- Scenarios include clear daylight, dusk/night, fog, rain, side view, rear view, and a partial-occlusion approximation.
- The script runs both pipelines using matched scenario overrides (same spawn logic, traffic, weather, and camera viewpoint per scenario).
- It records frame-level failure flags (`missed_detection_flag`, `unstable_classification_flag`) and writes an aggregated side-by-side summary CSV.

### 5) Adversarial attack implementation
- Use `python scripts\run_attack_evaluation.py --pipeline <camera_only|fusion> --config <stage5_config.json>`.
- Camera attacks support practical glare/overexposure and patch-style perturbation in the RGB processing path.
- LiDAR attacks support spoofing via phantom point cluster injection in the LiDAR processing path.
- The runner writes before/after artifacts, per-frame change flags (`detection_changed`, `decision_changed`), obstacle-point counters, and heartbeat logs every 10 processed frames.

### 6) Evaluation, visualization, report/presentation
- Generate comparative plots/tables in `outputs\plots` and `outputs\tables`.
- Export scenario videos to `outputs\videos`.
- Summarize findings in `docs\` for report and slides.

## Notes

- Current runner is intentionally minimal and reproducible.
- It connects to CARLA, applies map/weather, spawns one ego camera, captures one frame, and saves run metrics.
- Traffic, pedestrian simulation, and attack behavior are represented in config and logging first, then expanded in later stages.
