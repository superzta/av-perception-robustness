"""Regenerate summary tables + paper-quality plots from existing episode summaries.

This does NOT re-run any CARLA experiment. It walks
``outputs/episode_logs/*/<run_id>/episode_summary.json`` and rebuilds all
summary tables and plots into ``outputs/summary_tables`` and ``outputs/plots``.

Usage:
    python scripts/regenerate_paper_plots.py
    python scripts/regenerate_paper_plots.py --outputs-root outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_full_pipeline import (  # type: ignore  # noqa: E402
    aggregate_condition_metrics,
    build_attack_pair_table,
    build_category_level_table,
    choose_representative_screenshots,
    make_paper_plots,
    make_plots,
    top_failure_cases,
    write_csv,
    write_findings_and_outlines,
)


def _load_episode_rows(episode_logs_root: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in episode_logs_root.rglob("episode_summary.json"):
        try:
            with summary_path.open("r", encoding="utf-8") as fp:
                rows.append(json.load(fp))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="outputs",
                    help="Root containing episode_logs, summary_tables, plots, ...")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("regen")

    outputs_root = Path(args.outputs_root).resolve()
    episode_logs = outputs_root / "episode_logs"
    summary_tables = outputs_root / "summary_tables"
    plots_dir = outputs_root / "plots"
    rep_shots = outputs_root / "representative_screenshots"
    final_report = outputs_root / "final_report_assets"
    final_pres = outputs_root / "final_presentation_assets"
    for d in (summary_tables, plots_dir, rep_shots, final_report, final_pres):
        d.mkdir(parents=True, exist_ok=True)

    episode_rows = _load_episode_rows(episode_logs)
    logger.info("Loaded %d episode summaries from %s", len(episode_rows), episode_logs)
    if not episode_rows:
        logger.error("No episode summaries found.")
        return 1

    completed = [r for r in episode_rows if r.get("status") == "completed"]
    logger.info("%d completed (valid-visibility) episodes will be used for metrics.",
                len(completed))

    condition_rows = aggregate_condition_metrics(episode_rows)
    attack_rows = build_attack_pair_table(episode_rows)
    logger.info("Conditions: %d  attack pairs: %d", len(condition_rows), len(attack_rows))

    # Episode-level raw table (flatten to CSV-friendly).
    if episode_rows:
        fields = sorted({k for r in episode_rows for k in r.keys() if not isinstance(r[k], dict)})
        flat = []
        for r in episode_rows:
            flat.append({k: (r[k] if (k in r and not isinstance(r[k], (dict, list))) else "")
                         for k in fields})
        write_csv(summary_tables / "episode_level_raw_metrics.csv", flat, fields)

    if condition_rows:
        write_csv(summary_tables / "condition_level_aggregate_metrics.csv",
                  condition_rows, list(condition_rows[0].keys()))

    if attack_rows:
        write_csv(summary_tables / "attack_summary_table.csv",
                  attack_rows, list(attack_rows[0].keys()))

    failures = top_failure_cases(episode_rows)
    if failures:
        write_csv(summary_tables / "top_failure_cases.csv", failures, list(failures[0].keys()))

    # Category-level paper table.
    cat_rows = build_category_level_table(episode_rows)
    if cat_rows:
        fields = list(cat_rows[0].keys())
        for r in cat_rows:
            for f in fields:
                r.setdefault(f, "")
        write_csv(summary_tables / "category_level_pipeline_comparison.csv", cat_rows, fields)

    # Plots.
    logger.info("Rendering standard plots...")
    make_plots(plots_dir, condition_rows, attack_rows)
    logger.info("Rendering paper-level plots...")
    make_paper_plots(plots_dir, episode_rows=episode_rows,
                     condition_rows=condition_rows, attack_rows=attack_rows, logger=logger)

    # Representative screenshots index (best-effort).
    try:
        shot_index = choose_representative_screenshots(episode_rows, rep_shots)
        if shot_index:
            write_csv(summary_tables / "representative_screenshots_index.csv",
                      shot_index, list(shot_index[0].keys()))
    except Exception as exc:  # noqa: BLE001
        logger.warning("representative_screenshots failed: %s", exc)

    # Findings markdown.
    try:
        write_findings_and_outlines(
            report_dir=final_report,
            presentation_dir=final_pres,
            condition_rows=condition_rows,
            attack_rows=attack_rows,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("write_findings_and_outlines failed: %s", exc)

    # Mirror to final assets.
    import shutil
    for fname in ("condition_level_aggregate_metrics.csv", "attack_summary_table.csv",
                  "top_failure_cases.csv", "category_level_pipeline_comparison.csv"):
        src = summary_tables / fname
        if src.exists():
            shutil.copy2(src, final_report / fname)
    for png in plots_dir.glob("full_pipeline_*.png"):
        shutil.copy2(png, final_report / png.name)
        shutil.copy2(png, final_pres / png.name)

    logger.info("Done. Summary tables: %s", summary_tables)
    logger.info("Done. Plots: %s", plots_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
