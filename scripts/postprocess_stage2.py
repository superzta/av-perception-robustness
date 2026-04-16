import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def find_latest_detection_csv(logs_dir: Path) -> Path:
    candidates = sorted(logs_dir.glob("*_detections.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No detection CSV found in {logs_dir}")
    return candidates[0]


def parse_run_id(csv_path: Path) -> str:
    name = csv_path.name
    suffix = "_detections.csv"
    return name[: -len(suffix)] if name.endswith(suffix) else csv_path.stem


def write_class_table(output_path: Path, class_counts: Counter) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["class_name", "count"])
        writer.writeheader()
        for class_name, count in class_counts.most_common():
            writer.writerow({"class_name": class_name, "count": count})


def write_frame_table(output_path: Path, frame_counts: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["frame_id", "detection_count"])
        writer.writeheader()
        for frame_id in sorted(frame_counts.keys()):
            writer.writerow({"frame_id": frame_id, "detection_count": frame_counts[frame_id]})


def make_quick_plot(output_path: Path, class_counts: Counter) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = []
    values = []
    for class_name, count in class_counts.most_common(12):
        labels.append(class_name)
        values.append(count)

    if not labels:
        labels = ["no_detections"]
        values = [0]

    plt.figure(figsize=(10, 4))
    bars = plt.bar(labels, values)
    plt.title("Stage 2 Detection Count by Class")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process Stage 2 detection logs into tables and quick plot.")
    parser.add_argument("--detections_csv", type=str, default="", help="Path to a *_detections.csv file.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / "outputs" / "logs"
    tables_dir = project_root / "outputs" / "tables"
    plots_dir = project_root / "outputs" / "plots"

    detections_csv = Path(args.detections_csv).resolve() if args.detections_csv else find_latest_detection_csv(logs_dir)
    if not detections_csv.exists():
        raise FileNotFoundError(f"Detection CSV not found: {detections_csv}")

    run_id = parse_run_id(detections_csv)
    class_counts = Counter()
    frame_counts = defaultdict(int)
    total_rows = 0

    with detections_csv.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            total_rows += 1
            frame_id = int(row["frame_id"])
            cls = row["class_name"]
            if cls != "__none__":
                class_counts[cls] += 1
                frame_counts[frame_id] += 1
            else:
                frame_counts[frame_id] += 0

    class_table_path = tables_dir / f"{run_id}_counts_by_class.csv"
    frame_table_path = tables_dir / f"{run_id}_counts_by_frame.csv"
    plot_path = plots_dir / f"{run_id}_class_counts.png"

    write_class_table(class_table_path, class_counts)
    write_frame_table(frame_table_path, frame_counts)
    plot_generated = make_quick_plot(plot_path, class_counts)

    print(f"Detections source: {detections_csv}")
    print(f"Total rows: {total_rows}")
    print(f"Saved: {class_table_path}")
    print(f"Saved: {frame_table_path}")
    if plot_generated:
        print(f"Saved: {plot_path}")
    else:
        print("Plot skipped: matplotlib not installed. Install with `pip install matplotlib`.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
