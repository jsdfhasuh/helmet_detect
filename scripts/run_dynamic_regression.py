#!/usr/bin/env python3
"""Run full-frame, position-invariant dynamic V2 regression with real models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from helmet_detect.dynamic_config import load_dynamic_config
from helmet_detect.dynamic_pipeline import DynamicHelmetDetectionPipeline
from helmet_detect.dynamic_render import annotate_dynamic_frame
from helmet_detect.dynamic_types import HelmetState
from helmet_detect.dynamic_video import process_dynamic_video
from helmet_detect.fixtures import load_manifest_and_materialize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="testdata/dynamic_camera/manifest.json")
    parser.add_argument("--config", default="configs/dynamic_full_frame.json")
    parser.add_argument("--output", default="artifacts/dynamic-regression")
    return parser.parse_args()


def build_canvas(
    patch: np.ndarray,
    *,
    width: int,
    height: int,
    patch_x: int,
    patch_y: int,
    background: int,
) -> np.ndarray:
    if patch_x < 0 or patch_y < 0:
        raise ValueError("patch position cannot be negative")
    if patch_x + patch.shape[1] > width or patch_y + patch.shape[0] > height:
        raise ValueError("patch does not fit inside canvas")
    canvas = np.full((height, width, 3), int(background), dtype=np.uint8)
    canvas[
        patch_y : patch_y + patch.shape[0],
        patch_x : patch_x + patch.shape[1],
    ] = patch
    return canvas


def write_lossless_video(frames: list[np.ndarray], output: Path, fps: float) -> None:
    import cv2

    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"FFV1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create regression video: {output}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def run_image_cases(
    pipeline: DynamicHelmetDetectionPipeline,
    manifest: dict[str, Any],
    patch: np.ndarray,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    import cv2

    canvas_config = manifest["canvas"]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for case in manifest["image_cases"]:
        frame = build_canvas(
            patch,
            width=int(canvas_config["width"]),
            height=int(canvas_config["height"]),
            patch_x=int(case["patch_x"]),
            patch_y=int(canvas_config["patch_y"]),
            background=int(canvas_config.get("background", 114)),
        )
        result = pipeline.detect_frame(
            frame,
            timestamp_seconds=0.0,
            persist_tracks=False,
            apply_temporal=False,
        )
        no_helmet_people = [
            item for item in result.persons if item.state is HelmetState.NO_HELMET
        ]
        target = max(no_helmet_people, key=lambda item: item.no_helmet_score, default=None)
        actual_person_confidence = target.person.confidence if target is not None else 0.0
        actual_no_helmet_score = target.no_helmet_score if target is not None else 0.0
        centre_x = target.person.box.centre[0] if target is not None else -1.0
        minimum_person_confidence = float(case["minimum_person_confidence"])
        minimum_no_helmet_score = float(case["minimum_no_helmet_score"])
        expected_min_x = int(case["patch_x"]) + 170
        expected_max_x = int(case["patch_x"]) + 330
        passed = (
            target is not None
            and actual_person_confidence >= minimum_person_confidence
            and actual_no_helmet_score >= minimum_no_helmet_score
            and expected_min_x <= centre_x <= expected_max_x
        )
        reasons: list[str] = []
        if target is None:
            reasons.append("no dynamic no-helmet person was produced")
        if actual_person_confidence < minimum_person_confidence:
            reasons.append(
                f"person_confidence={actual_person_confidence:.4f} "
                f"< {minimum_person_confidence:.4f}"
            )
        if actual_no_helmet_score < minimum_no_helmet_score:
            reasons.append(
                f"no_helmet_score={actual_no_helmet_score:.4f} "
                f"< {minimum_no_helmet_score:.4f}"
            )
        if not expected_min_x <= centre_x <= expected_max_x:
            reasons.append(
                f"person_centre_x={centre_x:.1f} not in "
                f"[{expected_min_x}, {expected_max_x}]"
            )

        annotated = annotate_dynamic_frame(
            frame,
            result,
            config=pipeline.config,
            show_contexts=True,
        )
        image_path = images_dir / f"{case['id']}.jpg"
        cv2.imwrite(str(image_path), annotated)
        json_path = images_dir / f"{case['id']}.json"
        json_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        row = {
            "id": str(case["id"]),
            "patch_x": int(case["patch_x"]),
            "person_confidence": round(actual_person_confidence, 6),
            "no_helmet_score": round(actual_no_helmet_score, 6),
            "person_centre_x": round(centre_x, 3),
            "track_id": target.track_id if target is not None else None,
            "passed": passed,
            "reason": "; ".join(reasons),
        }
        rows.append(row)
        if not passed:
            failures.append(f"{case['id']}: {row['reason']}")
    return rows, failures


def run_video_case(
    pipeline: DynamicHelmetDetectionPipeline,
    manifest: dict[str, Any],
    patch: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    canvas_config = manifest["canvas"]
    case = manifest["video_case"]
    frames = [
        build_canvas(
            patch,
            width=int(canvas_config["width"]),
            height=int(canvas_config["height"]),
            patch_x=int(case["patch_x"]) + index,
            patch_y=int(canvas_config["patch_y"]),
            background=int(canvas_config.get("background", 114)),
        )
        for index in range(int(case["frames"]))
    ]
    video_dir = output_dir / "video"
    input_path = video_dir / "input.avi"
    write_lossless_video(frames, input_path, float(case["fps"]))
    summary = process_dynamic_video(
        pipeline,
        input_path,
        video_dir / "annotated.mp4",
        video_dir / "frames.jsonl",
        sample_fps=float(case["fps"]),
        show_contexts=True,
    )
    failures: list[str] = []
    if summary.no_helmet_observation_frames < int(case["minimum_no_helmet_frames"]):
        failures.append(
            "video: no_helmet_observation_frames="
            f"{summary.no_helmet_observation_frames} "
            f"< {int(case['minimum_no_helmet_frames'])}"
        )
    if summary.events < int(case["minimum_events"]):
        failures.append(f"video: events={summary.events} < {int(case['minimum_events'])}")
    if summary.maximum_no_helmet_score < float(case["minimum_maximum_score"]):
        failures.append(
            "video: maximum_no_helmet_score="
            f"{summary.maximum_no_helmet_score:.4f} "
            f"< {float(case['minimum_maximum_score']):.4f}"
        )
    data = summary.to_dict()
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
        "models": {
            "scene": "official COCO YOLO11n + ByteTrack",
            "helmet": "iam-tsr/yolov8n-helmet-detection",
        },
        "image_cases": image_rows,
        "video_case": video_row,
        "failures": failures,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "image_results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)

    lines = [
        "# Dynamic V2 helmet regression",
        "",
        f"**Result:** {'PASS' if not failures else 'FAIL'}",
        "",
        "- Scene model: official COCO YOLO11n",
        "- Tracker: ByteTrack",
        "- Per-person helmet model: iam-tsr YOLOv8n",
        "- Fixed ROI/gate: disabled",
        "",
        "## Position-invariance image cases",
        "",
        "| Case | Patch X | Person confidence | No-helmet score | Person centre X | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in image_rows:
        lines.append(
            f"| {row['id']} | {row['patch_x']} | {row['person_confidence']:.4f} | "
            f"{row['no_helmet_score']:.4f} | {row['person_centre_x']:.1f} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Per-track video case",
            "",
            f"- Processed frames: {video_row['processed_frames']}",
            f"- Frames with people: {video_row['frames_with_people']}",
            "- No-helmet observation frames: "
            f"{video_row['no_helmet_observation_frames']}",
            f"- Alarm events: {video_row['events']}",
            f"- Unique tracks: {video_row['unique_tracks']}",
            "- Maximum no-helmet score: "
            f"{video_row['maximum_no_helmet_score']:.4f}",
        ]
    )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_path, manifest = load_manifest_and_materialize(args.manifest)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    patch_path = manifest_path.parent / "images/target_patch.webp"
    patch = cv2.imread(str(patch_path))
    if patch is None:
        raise RuntimeError(f"Cannot read materialized patch: {patch_path}")

    pipeline = DynamicHelmetDetectionPipeline(load_dynamic_config(args.config))
    image_rows, failures = run_image_cases(pipeline, manifest, patch, output_dir)
    video_row, video_failures = run_video_case(
        pipeline,
        manifest,
        patch,
        output_dir,
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
