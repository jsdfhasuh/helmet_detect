#!/usr/bin/env python3
"""Run real-model regression checks against privacy-cropped camera samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from helmet_detect.config import load_config
from helmet_detect.fixtures import load_manifest_and_materialize
from helmet_detect.pipeline import HelmetDetectionPipeline
from helmet_detect.render import annotate_frame
from helmet_detect.sample import build_video_from_images
from helmet_detect.video import process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="testdata/camera_exit/manifest.json",
    )
    parser.add_argument("--output", default="artifacts/regression")
    return parser.parse_args()


def load_pipeline(
    config_path: Path,
    cache: dict[Path, HelmetDetectionPipeline],
) -> HelmetDetectionPipeline:
    key = config_path.resolve()
    if key not in cache:
        cache[key] = HelmetDetectionPipeline(load_config(key))
    return cache[key]


def run_image_cases(
    manifest: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    cache: dict[Path, HelmetDetectionPipeline],
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for regression tests") from exc

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    image_output = output_dir / "images"
    image_output.mkdir(parents=True, exist_ok=True)

    for case in manifest["image_cases"]:
        case_id = str(case["id"])
        config_path = (manifest_path.parent / case["config"]).resolve()
        image_path = (manifest_path.parent / case["image"]).resolve()
        pipeline = load_pipeline(config_path, cache)
        frame = cv2.imread(str(image_path))
        if frame is None:
            failures.append(f"{case_id}: cannot read {image_path}")
            continue

        result = pipeline.detect_frame(frame)
        score = result.fused_no_helmet_score
        expected_alarm = bool(case["expected_alarm"])
        passed = result.alarm == expected_alarm
        reasons: list[str] = []
        if result.alarm != expected_alarm:
            reasons.append(f"alarm={result.alarm}, expected={expected_alarm}")
        if "min_score" in case and score < float(case["min_score"]):
            passed = False
            reasons.append(f"score={score:.4f} < min_score={float(case['min_score']):.4f}")
        if "max_score" in case and score > float(case["max_score"]):
            passed = False
            reasons.append(f"score={score:.4f} > max_score={float(case['max_score']):.4f}")
        required_model = case.get("required_model")
        if required_model:
            required_score = result.best_no_helmet.get(str(required_model), 0.0)
            required_min = float(case.get("required_model_min_score", 0.0))
            if required_score < required_min:
                passed = False
                reasons.append(
                    f"{required_model}={required_score:.4f} < required={required_min:.4f}"
                )

        annotated = annotate_frame(
            frame,
            result,
            roi=pipeline.config.roi,
            gate=pipeline.config.gate,
        )
        annotated_path = image_output / f"{case_id}.jpg"
        if not cv2.imwrite(str(annotated_path), annotated):
            passed = False
            reasons.append(f"failed to write {annotated_path}")

        row = {
            "id": case_id,
            "expected_alarm": expected_alarm,
            "actual_alarm": result.alarm,
            "fused_score": round(score, 6),
            "best_no_helmet": result.best_no_helmet,
            "best_helmet": result.best_helmet,
            "passed": passed,
            "reason": "; ".join(reasons),
            "annotated": str(annotated_path),
        }
        rows.append(row)
        (image_output / f"{case_id}.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not passed:
            failures.append(f"{case_id}: {row['reason'] or 'regression failed'}")

    return rows, failures


def run_video_case(
    manifest: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    cache: dict[Path, HelmetDetectionPipeline],
) -> tuple[dict[str, Any], list[str]]:
    case = manifest["video_case"]
    config_path = (manifest_path.parent / case["config"]).resolve()
    video_output = output_dir / "video"
    video_output.mkdir(parents=True, exist_ok=True)
    if "frames" in case:
        frame_paths = [manifest_path.parent / item for item in case["frames"]]
        source_path = build_video_from_images(
            frame_paths,
            video_output / "input.avi",
            fps=float(case.get("fps", 6.0)),
            repeat_each=int(case.get("repeat_each", 1)),
        )
    else:
        source_path = (manifest_path.parent / case["source"]).resolve()
    pipeline = load_pipeline(config_path, cache)
    summary = process_video(
        pipeline,
        source_path,
        video_output / "annotated.mp4",
        video_output / "frames.jsonl",
        sample_fps=float(case.get("sample_fps", 0)),
    )
    data = summary.to_dict()
    failures: list[str] = []
    min_alarm_frames = int(case.get("min_alarm_frames", 1))
    min_events = int(case.get("min_events", 1))
    min_score = float(case.get("min_score", 0.0))
    if summary.alarm_frames < min_alarm_frames:
        failures.append(
            f"video: alarm_frames={summary.alarm_frames} < {min_alarm_frames}"
        )
    if summary.events < min_events:
        failures.append(f"video: events={summary.events} < {min_events}")
    if summary.maximum_no_helmet_score < min_score:
        failures.append(
            "video: maximum_no_helmet_score="
            f"{summary.maximum_no_helmet_score:.4f} < {min_score:.4f}"
        )
    data["passed"] = not failures
    return data, failures


def write_reports(
    output_dir: Path,
    image_rows: list[dict[str, Any]],
    video_row: dict[str, Any],
    failures: list[str],
) -> None:
    report = {
        "passed": not failures,
        "image_cases": image_rows,
        "video_case": video_row,
        "failures": failures,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "image_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "expected_alarm",
                "actual_alarm",
                "fused_score",
                "passed",
                "reason",
            ],
        )
        writer.writeheader()
        for row in image_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    lines = [
        "# Helmet detection regression",
        "",
        f"**Result:** {'PASS' if not failures else 'FAIL'}",
        "",
        "## Image cases",
        "",
        "| Case | Expected | Actual | Fused no-helmet score | Result |",
        "|---|---:|---:|---:|---|",
    ]
    for row in image_rows:
        lines.append(
            f"| {row['id']} | {row['expected_alarm']} | {row['actual_alarm']} | "
            f"{row['fused_score']:.4f} | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Video case",
            "",
            f"- Processed frames: {video_row['processed_frames']}",
            f"- Alarm frames: {video_row['alarm_frames']}",
            f"- Temporal events: {video_row['events']}",
            f"- Maximum no-helmet score: {video_row['maximum_no_helmet_score']:.4f}",
        ]
    )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path, manifest = load_manifest_and_materialize(args.manifest)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, HelmetDetectionPipeline] = {}

    image_rows, failures = run_image_cases(
        manifest,
        manifest_path,
        output_dir,
        cache,
    )
    video_row, video_failures = run_video_case(
        manifest,
        manifest_path,
        output_dir,
        cache,
    )
    failures.extend(video_failures)
    write_reports(output_dir, image_rows, video_row, failures)

    print((output_dir / "summary.md").read_text(encoding="utf-8"))
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
