import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:  # noqa: BLE001
        return int(default)


def compute_rates(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return precision, recall, fpr, fnr


def load_stage4_summary(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(row)
    return rows


def load_frame_counts(frame_metrics_csv: Path, pipeline: str):
    counts = []
    if not frame_metrics_csv.exists():
        return counts
    with frame_metrics_csv.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if pipeline == "camera_only":
                counts.append(safe_int(row.get("detection_count", 0)))
            else:
                counts.append(safe_int(row.get("fused_detection_count", 0)))
    return counts


def classify_scenario_group(name: str) -> str:
    if name == "clear_daylight_front":
        return "normal"
    if name in {"dusk_night_front", "fog_front", "rain_front"}:
        return "adverse_weather_low_visibility"
    return "viewpoint_variation"


def compute_stage4_metrics(stage4_rows):
    by_scenario = defaultdict(dict)
    for row in stage4_rows:
        by_scenario[row["scenario_name"]][row["pipeline"]] = row

    scenario_metrics = []
    for scenario_name, data in by_scenario.items():
        cam = data.get("camera_only")
        fus = data.get("camera_lidar_fusion")
        if not cam or not fus:
            continue
        cam_csv = Path(cam.get("frame_metrics_csv", ""))
        fus_csv = Path(fus.get("frame_metrics_csv", ""))
        cam_counts = load_frame_counts(cam_csv, "camera_only")
        fus_counts = load_frame_counts(fus_csv, "camera_lidar_fusion")
        n = min(len(cam_counts), len(fus_counts))
        if n == 0:
            continue

        for pipeline_name in ["camera_only", "camera_lidar_fusion"]:
            tp = fp = tn = fn = 0
            for i in range(n):
                gt_positive = int((cam_counts[i] > 0) or (fus_counts[i] > 0))
                pred_positive = int(cam_counts[i] > 0) if pipeline_name == "camera_only" else int(fus_counts[i] > 0)
                if pred_positive and gt_positive:
                    tp += 1
                elif pred_positive and not gt_positive:
                    fp += 1
                elif (not pred_positive) and (not gt_positive):
                    tn += 1
                else:
                    fn += 1

            precision, recall, fpr, fnr = compute_rates(tp, fp, tn, fn)
            ref = cam if pipeline_name == "camera_only" else fus
            scenario_metrics.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_group": classify_scenario_group(scenario_name),
                    "pipeline": pipeline_name,
                    "frames": n,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "false_positive_rate": round(fpr, 4),
                    "false_negative_rate": round(fnr, 4),
                    "mean_detection_count": safe_float(ref.get("mean_detection_count", 0.0)),
                    "missed_detection_frames": safe_int(ref.get("missed_detection_frames", 0)),
                    "unstable_classification_frames": safe_int(ref.get("unstable_classification_frames", 0)),
                    "representative_screenshot": ref.get("representative_screenshot", ""),
                }
            )
    return scenario_metrics


def find_stage5_metric_files(metrics_dir: Path):
    return sorted(metrics_dir.glob("*_stage5_attack_eval.json"))


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_stage5_metrics(stage5_metric_files):
    attack_rows = []
    for metric_path in stage5_metric_files:
        m = read_json(metric_path)
        if m.get("status") != "completed":
            continue
        frame_csv = Path(m.get("frame_attack_csv", ""))
        if not frame_csv.exists():
            continue
        tp = fp = tn = fn = 0
        correct_stop = missed_stop = false_stop = 0
        frames = 0
        det_changed = 0
        decision_changed = 0
        with frame_csv.open("r", newline="", encoding="utf-8") as fp_csv:
            reader = csv.DictReader(fp_csv)
            for row in reader:
                frames += 1
                clean_pos = int(safe_int(row.get("clean_detection_count", 0)) > 0)
                attacked_pos = int(safe_int(row.get("attacked_detection_count", 0)) > 0)
                if attacked_pos and clean_pos:
                    tp += 1
                elif attacked_pos and not clean_pos:
                    fp += 1
                elif (not attacked_pos) and (not clean_pos):
                    tn += 1
                else:
                    fn += 1

                clean_decision = row.get("clean_decision", "")
                attacked_decision = row.get("attacked_decision", "")
                clean_brake = clean_decision.startswith("BRAKE")
                attacked_brake = attacked_decision.startswith("BRAKE")
                if clean_brake and attacked_brake:
                    correct_stop += 1
                elif clean_brake and (not attacked_brake):
                    missed_stop += 1
                elif (not clean_brake) and attacked_brake:
                    false_stop += 1

                det_changed += safe_int(row.get("detection_changed", 0))
                decision_changed += safe_int(row.get("decision_changed", 0))

        precision, recall, fpr, fnr = compute_rates(tp, fp, tn, fn)
        attack_rows.append(
            {
                "run_id": m.get("run_id", metric_path.stem),
                "pipeline": m.get("pipeline", "unknown"),
                "attack_experiment": m.get("experiment_name", "unknown"),
                "scenario_name": m.get("scenario_name", "unknown"),
                "frames": frames,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
                "correct_stop": correct_stop,
                "missed_stop": missed_stop,
                "false_stop": false_stop,
                "detection_changed_frames": det_changed,
                "decision_changed_frames": decision_changed,
                "frame_attack_csv": str(frame_csv),
                "output_root": m.get("output_root", ""),
            }
        )
    return attack_rows


def aggregate_condition_metrics(stage4_metrics, stage5_metrics):
    bucket = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "frames": 0, "pipeline": ""})
    for row in stage4_metrics:
        group = row["scenario_group"]
        key = (group, row["pipeline"])
        bucket[key]["tp"] += row["tp"]
        bucket[key]["fp"] += row["fp"]
        bucket[key]["tn"] += row["tn"]
        bucket[key]["fn"] += row["fn"]
        bucket[key]["frames"] += row["frames"]
        bucket[key]["pipeline"] = row["pipeline"]

    for row in stage5_metrics:
        key = ("adversarial_attacks", row["pipeline"])
        bucket[key]["tp"] += row["tp"]
        bucket[key]["fp"] += row["fp"]
        bucket[key]["tn"] += row["tn"]
        bucket[key]["fn"] += row["fn"]
        bucket[key]["frames"] += row["frames"]
        bucket[key]["pipeline"] = row["pipeline"]

    out = []
    for (group, pipeline), vals in sorted(bucket.items()):
        precision, recall, fpr, fnr = compute_rates(vals["tp"], vals["fp"], vals["tn"], vals["fn"])
        out.append(
            {
                "condition_group": group,
                "pipeline": pipeline,
                "frames": vals["frames"],
                "tp": vals["tp"],
                "fp": vals["fp"],
                "tn": vals["tn"],
                "fn": vals["fn"],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4),
            }
        )
    return out


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_representative_images(stage4_rows, stage5_rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for row in stage4_rows:
        src = Path(row.get("representative_screenshot", ""))
        if src.exists():
            dst = out_dir / f"stage4_{row['scenario_name']}_{row['pipeline']}.png"
            shutil.copy2(src, dst)
            copied.append(str(dst))

    for row in stage5_rows:
        output_root = Path(row.get("output_root", ""))
        attacked_dir = output_root / "annotated_attacked"
        clean_dir = output_root / "annotated_clean"
        attacked_images = sorted(attacked_dir.glob("*.png"))
        clean_images = sorted(clean_dir.glob("*.png"))
        if clean_images:
            src = clean_images[len(clean_images) // 2]
            dst = out_dir / f"stage5_{row['run_id']}_clean.png"
            shutil.copy2(src, dst)
            copied.append(str(dst))
        if attacked_images:
            src = attacked_images[len(attacked_images) // 2]
            dst = out_dir / f"stage5_{row['run_id']}_attacked.png"
            shutil.copy2(src, dst)
            copied.append(str(dst))
    return copied


def make_plots(plots_dir: Path, stage4_metrics, condition_metrics, stage5_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for Stage 6 plots. Install with `pip install matplotlib`.") from exc

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: detection counts by scenario.
    scenarios = sorted({r["scenario_name"] for r in stage4_metrics})
    pipelines = ["camera_only", "camera_lidar_fusion"]
    pipeline_color = {"camera_only": "#e76f51", "camera_lidar_fusion": "#2a9d8f"}
    x = range(len(scenarios))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 5), dpi=170)
    for idx, pipeline in enumerate(pipelines):
        vals = []
        for sc in scenarios:
            match = next((r for r in stage4_metrics if r["scenario_name"] == sc and r["pipeline"] == pipeline), None)
            vals.append(match["mean_detection_count"] if match else 0.0)
        offset = [i + (idx - 0.5) * width for i in x]
        ax.bar(offset, vals, width=width, color=pipeline_color[pipeline], label=pipeline.replace("_", " "))
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios, rotation=28, ha="right")
    ax.set_ylabel("Mean detections per frame")
    ax.set_title("Camera-only vs Fusion across Stage 4 scenarios")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_mean_detection_by_scenario.png")
    plt.close(fig)

    # Plot 2: precision/recall by condition group.
    condition_groups = ["normal", "adverse_weather_low_visibility", "viewpoint_variation", "adversarial_attacks"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=170)
    for ax_idx, metric in enumerate(["precision", "recall"]):
        ax = axes[ax_idx]
        for idx, pipeline in enumerate(pipelines):
            vals = []
            for g in condition_groups:
                row = next((r for r in condition_metrics if r["condition_group"] == g and r["pipeline"] == pipeline), None)
                vals.append(row[metric] if row else 0.0)
            offset = [i + (idx - 0.5) * width for i in range(len(condition_groups))]
            ax.bar(offset, vals, width=width, color=pipeline_color[pipeline], label=pipeline.replace("_", " "))
        ax.set_xticks(list(range(len(condition_groups))))
        ax.set_xticklabels(condition_groups, rotation=22, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(metric.title())
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.suptitle("Precision/Recall comparison by condition group")
    fig.tight_layout()
    fig.savefig(plots_dir / "stage6_precision_recall_by_condition.png")
    plt.close(fig)

    # Plot 3: attack impact rates.
    if stage5_rows:
        attacks = sorted({r["attack_experiment"] for r in stage5_rows})
        fig, ax = plt.subplots(figsize=(10, 5), dpi=170)
        bar_w = 0.35
        for idx, pipeline in enumerate(pipelines):
            vals = []
            labels = []
            for attack in attacks:
                subset = [r for r in stage5_rows if r["pipeline"] == pipeline and r["attack_experiment"] == attack]
                if not subset:
                    vals.append(0.0)
                else:
                    total_frames = sum(r["frames"] for r in subset)
                    changed = sum(r["decision_changed_frames"] for r in subset)
                    vals.append(changed / total_frames if total_frames > 0 else 0.0)
                labels.append(attack)
            offset = [i + (idx - 0.5) * bar_w for i in range(len(attacks))]
            ax.bar(offset, vals, width=bar_w, color=pipeline_color[pipeline], label=pipeline.replace("_", " "))
        ax.set_xticks(list(range(len(attacks))))
        ax.set_xticklabels(attacks, rotation=20, ha="right")
        ax.set_ylabel("Decision change rate")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Adversarial attack impact on downstream decisions")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "stage6_attack_decision_change_rate.png")
        plt.close(fig)


def write_outlines(tables_dir: Path):
    presentation_outline = tables_dir / "stage6_presentation_outline.md"
    report_outline = tables_dir / "stage6_report_outline.md"

    presentation_outline.write_text(
        "\n".join(
            [
                "# Stage 6 Presentation Outline",
                "",
                "## 1) Motivation",
                "- Why camera-only vs fusion robustness matters for AV safety and cost.",
                "- Threat model: weather degradation and adversarial sensor attacks.",
                "",
                "## 2) Hypothesis",
                "- Fusion is expected to be more robust under degradation and single-modality attacks.",
                "",
                "## 3) System Setup",
                "- CARLA environment, maps, weather controls, and synchronized sensors.",
                "- Camera-only baseline (YOLO) and camera+LiDAR late-fusion baseline.",
                "",
                "## 4) Experiment Design",
                "- Stage 4 controlled scenarios: clear, dusk/night, fog, rain, viewpoint shifts.",
                "- Stage 5 attacks: camera glare/patch and LiDAR phantom point spoofing.",
                "",
                "## 5) Main Results",
                "- Show `stage6_mean_detection_by_scenario.png` and `stage6_precision_recall_by_condition.png`.",
                "- Highlight condition groups where camera-only degrades most.",
                "",
                "## 6) Failure Cases",
                "- Representative screenshots from `stage6_representative_screenshots`.",
                "- Show examples of detection changes and decision changes under attacks.",
                "",
                "## 7) Conclusion",
                "- Summarize robustness differences and design implications.",
                "- Next steps: stronger attack realism and better calibration/ground-truthing.",
            ]
        ),
        encoding="utf-8",
    )

    report_outline.write_text(
        "\n".join(
            [
                "# Stage 6 Report Outline",
                "",
                "## Abstract",
                "- One-paragraph summary of objectives, methods, and key findings.",
                "",
                "## 1. Introduction",
                "- Problem statement and research gap.",
                "",
                "## 2. Methodology",
                "- 2.1 Simulation environment and data collection setup.",
                "- 2.2 Camera-only and fusion baselines.",
                "- 2.3 Stage 4 controlled robustness scenarios.",
                "- 2.4 Stage 5 adversarial attack implementation.",
                "",
                "## 3. Evaluation Metrics",
                "- Precision/recall/FPR/FNR and decision metrics (correct stop, missed stop, false stop).",
                "- Note proxy-ground-truth assumption used for simulator logs.",
                "",
                "## 4. Results",
                "- 4.1 Normal vs adverse weather/visibility.",
                "  - Figure: `stage6_mean_detection_by_scenario.png`.",
                "  - Table: `stage6_scenario_metrics.csv`.",
                "- 4.2 Condition-group metric comparison.",
                "  - Figure: `stage6_precision_recall_by_condition.png`.",
                "  - Table: `stage6_condition_metrics.csv`.",
                "- 4.3 Adversarial attack impact.",
                "  - Figure: `stage6_attack_decision_change_rate.png`.",
                "  - Table: `stage6_attack_metrics.csv`.",
                "",
                "## 5. Qualitative Failure Analysis",
                "- Place representative screenshots from `stage6_representative_screenshots/`.",
                "- Discuss failure modes for camera-only and fusion separately.",
                "",
                "## 6. Limitations",
                "- Proxy labels and simulator realism constraints.",
                "",
                "## 7. Conclusion and Future Work",
                "- Final takeaways and next experimental improvements.",
            ]
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Stage 6 evaluation and reporting asset generator.")
    parser.add_argument("--stage4-summary", type=str, default="outputs/stage4_report/summary.csv")
    parser.add_argument("--metrics-dir", type=str, default="outputs/metrics")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots")
    parser.add_argument("--tables-dir", type=str, default="outputs/tables")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    stage4_summary_path = (project_root / args.stage4_summary).resolve()
    metrics_dir = (project_root / args.metrics_dir).resolve()
    plots_dir = (project_root / args.plots_dir).resolve()
    tables_dir = (project_root / args.tables_dir).resolve()

    stage4_rows = load_stage4_summary(stage4_summary_path)
    stage4_metrics = compute_stage4_metrics(stage4_rows)
    stage5_metric_files = find_stage5_metric_files(metrics_dir)
    stage5_metrics = compute_stage5_metrics(stage5_metric_files)
    condition_metrics = aggregate_condition_metrics(stage4_metrics, stage5_metrics)

    write_csv(
        tables_dir / "stage6_scenario_metrics.csv",
        stage4_metrics,
        [
            "scenario_name",
            "scenario_group",
            "pipeline",
            "frames",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
            "mean_detection_count",
            "missed_detection_frames",
            "unstable_classification_frames",
            "representative_screenshot",
        ],
    )
    write_csv(
        tables_dir / "stage6_attack_metrics.csv",
        stage5_metrics,
        [
            "run_id",
            "pipeline",
            "attack_experiment",
            "scenario_name",
            "frames",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
            "correct_stop",
            "missed_stop",
            "false_stop",
            "detection_changed_frames",
            "decision_changed_frames",
            "frame_attack_csv",
            "output_root",
        ],
    )
    write_csv(
        tables_dir / "stage6_condition_metrics.csv",
        condition_metrics,
        [
            "condition_group",
            "pipeline",
            "frames",
            "tp",
            "fp",
            "tn",
            "fn",
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
        ],
    )

    copied_images = copy_representative_images(stage4_rows, stage5_metrics, plots_dir / "stage6_representative_screenshots")
    write_csv(
        tables_dir / "stage6_representative_screenshots_index.csv",
        [{"image_path": p} for p in copied_images],
        ["image_path"],
    )

    make_plots(plots_dir, stage4_metrics, condition_metrics, stage5_metrics)
    write_outlines(tables_dir)

    print(f"Stage 6 scenario metrics: {tables_dir / 'stage6_scenario_metrics.csv'}")
    print(f"Stage 6 attack metrics: {tables_dir / 'stage6_attack_metrics.csv'}")
    print(f"Stage 6 condition metrics: {tables_dir / 'stage6_condition_metrics.csv'}")
    print(f"Stage 6 plots saved under: {plots_dir}")
    print(f"Representative screenshots: {plots_dir / 'stage6_representative_screenshots'}")
    print(f"Presentation/report outlines: {tables_dir}")


if __name__ == "__main__":
    main()
