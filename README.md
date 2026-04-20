# Robust Perception for Autonomous Vehicles

Comparative study of camera-only and camera+LiDAR fusion perception pipelines in CARLA under environmental degradation and adversarial perturbations.

This repository is designed to be reproducible and presentation-ready: one command runs the full episode-based workflow and generates tables, plots, screenshots, and report assets.

## Research Scope

This project evaluates two perception stacks:

- `camera_only`: RGB + YOLO-based object detection
- `fusion`: RGB + LiDAR PointPainting-style fusion (semantic painting + 3D point confirmation)

Across four stress families:

- normal visibility
- adverse weather and low visibility
- viewpoint variation (front/side/rear/occlusion-style setups)
- adversarial attacks (camera glare/patch-style effects and LiDAR spoofing)

The ego vehicle is driven in **closed loop** directly from the perception stack:
throttle, brake, and steering are computed every frame from detection outputs and
(for fusion) LiDAR-derived front-obstacle distance, and applied to CARLA via
`VehicleControl`. There is no offline signal in the control path, so attacks on the
camera or LiDAR can change the actual trajectory of the car, not only the logged
decision label.

## Repository Layout

- `scripts/` experiment runners and utilities
- `configs/` reproducible JSON experiment configs
- `models/` local model weights (git-ignored)
- `outputs/` all runtime artifacts (logs, images, tables, plots, assets)
- `README.md`, `requirements.txt`, `.gitignore`

## Setup

### 1) Clone the repository

```powershell
git clone https://github.com/superzta/av-perception-robustness.git
cd av-perception-robustness
```

### 2) Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3) Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 4) Install CARLA (packaged release)

This project expects a packaged CARLA install (not a source build). Download CARLA from the official release page:

- [CARLA Releases](https://github.com/carla-simulator/carla/releases)

Recommended local path:

- `av-perception-robustness\carla\`

The folder is already git-ignored, so it will not be committed.

### 5) Start CARLA server

From your CARLA folder:

```powershell
.\CarlaUE4.exe
```

## Quick Start

Run one baseline, or run the full study workflow.

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

PointPainting settings in `fusion`:

- `mode`: `pointpainting_semantic_fusion`
- `require_pointpainting`: fail fast if torch/torchvision are unavailable
- `segmentation_model`: currently `deeplabv3_resnet50`
- `segmentation_input_size`: long-side resize for the segmentation input (e.g. `512`). Smaller is faster.
- `segmentation_half_precision`: run the segmentation backbone in FP16 on CUDA
- `semantic_match_min_ratio`, `semantic_match_min_score`: semantic confirmation thresholds

## Closed-Loop Autonomous Driving (Perception-in-the-Loop)

Each episode runs a real closed-loop AV stack rather than a purely offline evaluation.
The controller lives in `scripts\utils\control_utils.py` and is wired into
`scripts\run_full_pipeline.py`.

Architecture:

- Perception (camera-only or camera+LiDAR PointPainting fusion) produces per-frame
  detections and, for fusion, a minimum front-obstacle distance from LiDAR.
- `PerceptionDrivingController` converts that perception output into a
  `carla.VehicleControl(throttle, brake, steer)` command every synchronous tick.
- CARLA autopilot is disabled for the ego; the controller's output is the only
  actuation signal.

Longitudinal control (throttle and brake) is driven entirely by perception:

- Emergency brake when the symbolic decision is `BRAKE`, or when the measured
  front-obstacle distance is at or below `brake_distance_m`.
- Soft slow-down (reduced throttle, light brake) when the decision is `SLOW_DOWN`
  or the front-obstacle distance is at or below `slow_distance_m`.
- Otherwise a P-controller tracks `target_speed_kmh`.

Lateral control (steering) follows the CARLA map's lane waypoints with a small
look-ahead pure-pursuit style law. This mirrors how real AV stacks separate
planning (route/lane following) from perception (when to stop and how hard to brake).

Controller configuration lives under `ego_vehicle.controller` in
`configs\full_pipeline_config.json`:

```json
"ego_vehicle": {
  "blueprint_filter": "vehicle.tesla.model3",
  "autopilot_enabled": false,
  "controller": {
    "mode": "perception_closed_loop",
    "target_speed_kmh": 25.0,
    "lookahead_m": 6.0,
    "max_steer": 0.7,
    "steer_kp": 0.9,
    "throttle_kp": 0.5,
    "max_throttle": 0.6,
    "brake_distance_m": 12.0,
    "slow_distance_m": 22.0,
    "slow_throttle": 0.2,
    "stop_speed_threshold_mps": 0.3,
    "emergency_brake": 1.0
  }
}
```

Because the controller is closed-loop on perception, the pipeline also logs the
ego vehicle's **real** behavior, not only the symbolic decision:

- frame-level columns: `ego_speed_mps`, `ego_throttle`, `ego_brake`, `ego_steer`,
  `control_reason`, `actual_stop_ok`, `actual_stop_missed`, `actual_stop_false`
- episode-level fields: `control_mode`, `real_correct_stop_rate`,
  `real_missed_stop_rate`, `real_false_stop_rate`, `mean_ego_speed_mps`,
  `min_ego_speed_mps`, `mean_throttle`, `mean_brake`, plus the existing
  `collision_flag` and `min_obstacle_distance_m`.

These real-behavior metrics are aggregated per condition, so final plots and
tables differentiate camera-only vs. fusion (and clean vs. attacked) by what the
car physically did under perception control, in addition to detection accuracy.

To fall back to CARLA autopilot for debugging, set
`ego_vehicle.controller.mode` to any value other than `perception_closed_loop`
and `ego_vehicle.autopilot_enabled` to `true`.

## Performance Tuning

All compute- and I/O-heavy components expose knobs so that large episode sweeps
stay tractable. Defaults are chosen for fast runs on a machine with an NVIDIA
GPU.

GPU / compute:

- `detector.device`, `detector.require_cuda`: force YOLO onto a specific CUDA
  device and fail fast if CUDA is missing.
- `detector.imgsz`: inference resolution passed to Ultralytics (e.g. `640`).
- `detector.half_precision`: run YOLO in FP16 on CUDA.
- `fusion.segmentation_device`, `fusion.segmentation_half_precision`: GPU/FP16
  selection for the DeepLabV3 segmentation backbone.
- `fusion.segmentation_input_size`: long-side resize for segmentation input.
  `512` is roughly 3-4x faster than full 1280x720 with negligible impact on
  semantic point painting.

Top-level I/O knobs in `configs\full_pipeline_config.json`:

- `save_every_n`: save every Nth processed frame (RGB images and frame CSV row).
- `save_every_n_lidar`: save raw LiDAR `.npy` every Nth frame (defaults to
  `save_every_n`).
- `save_every_n_lidar_bev`: save the LiDAR bird's-eye visualization every Nth
  frame.
- `save_lidar_clean_when_no_attack`: when LiDAR attacks are disabled for a run,
  skip saving the redundant clean LiDAR copy (default `false`).
- `image_format`: `"jpg"` (default) or `"png"`. JPEG is dramatically faster and
  smaller with no loss of detection fidelity.
- `jpeg_quality`, `png_compression_level`: codec speed/quality tradeoffs.
- `representative_frame_cap`: upper bound on the number of annotated frames
  tracked for representative-screenshot selection.

Simulator-level knobs:

- `post_map_switch_stabilize_seconds`: time to let CARLA settle after a map
  reload. Set lower on very stable setups.
- `npc_spawn_cap`: upper bound on NPC vehicles; lower values reduce per-tick
  physics cost.
- `group_episodes_by_town`: process all episodes of a town before switching
  maps, minimizing the number of expensive map reloads.

## Reproducibility and Reliability Features

- deterministic episode specification via seeded config
- per-episode summaries for resume support
- simulator preflight checks before long runs
- optional town health diagnostics
- map-switch stabilization delay
- episode retries and consecutive-failure stop guard
- heartbeat logging during long-running episodes

## GitHub Notes

- `carla/`, `models/`, `outputs/`, and `venv/` are git-ignored.
- No `data/` folder is required for this project setup.
- No `docs/` folder is required to run experiments; report assets are generated under `outputs/`.

If you are reviewing this project, start with:

1. `configs\full_pipeline_config.json`
2. `scripts\run_full_pipeline.py`
3. `outputs\summary_tables\` and `outputs\plots\` after a run
