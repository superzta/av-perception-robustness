import csv
import json
from datetime import datetime
from pathlib import Path


def make_run_id(experiment_name: str) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in experiment_name)
    return f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def ensure_directories(mapping: dict) -> dict:
    out = {}
    for key, raw_path in mapping.items():
        path = Path(raw_path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        out[key] = path
    return out


def save_metrics_json(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def append_metrics_csv(metrics: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        key: (json.dumps(value) if isinstance(value, (dict, list)) else value)
        for key, value in metrics.items()
    }

    existing_rows = []
    existing_fieldnames = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)

    fieldnames = sorted(set(existing_fieldnames).union(normalized.keys()))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)
        writer.writerow(normalized)


def annotate_image(image_path: Path, output_path: Path, text_lines, logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow not found, saving original image without annotations.")
        output_path.write_bytes(image_path.read_bytes())
        return

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    line_height = 18
    margin = 12
    box_height = margin * 2 + line_height * max(1, len(text_lines))
    draw.rectangle([(0, 0), (image.width, box_height)], fill=(0, 0, 0))

    y = margin
    for line in text_lines:
        draw.text((margin, y), str(line), fill=(255, 255, 255), font=font)
        y += line_height

    image.save(output_path)
