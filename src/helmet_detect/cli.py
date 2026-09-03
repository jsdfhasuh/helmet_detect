"""Command-line interface for legacy and dynamic helmet detection pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .dynamic_config import config_pipeline, load_dynamic_config
from .dynamic_pipeline import DynamicHelmetDetectionPipeline
from .dynamic_render import annotate_dynamic_frame
from .dynamic_video import process_dynamic_video
from .pipeline import HelmetDetectionPipeline
from .render import annotate_frame
from .video import process_video


def _parse_models(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helmet-detect",
        description=(
            "Detect motorcycle/e-bike riders without safety helmets. "
            "Dynamic V2 first tracks people globally and then runs helmet inference "
            "on person-centred context crops."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="Process one image")
    image_parser.add_argument("--config", required=True)
    image_parser.add_argument("--source", required=True)
    image_parser.add_argument("--output", required=True)
    image_parser.add_argument("--json", dest="json_path", required=True)
    image_parser.add_argument("--models", help="Legacy: comma-separated model names")
    image_parser.add_argument("--show-contexts", action="store_true")
    image_parser.add_argument("--fail-on-no-alarm", action="store_true")

    video_parser = subparsers.add_parser("video", help="Process a video")
    video_parser.add_argument("--config", required=True)
    video_parser.add_argument("--source", required=True)
    video_parser.add_argument("--output", required=True)
    video_parser.add_argument("--records", required=True)
    video_parser.add_argument("--summary", required=True)
    video_parser.add_argument("--sample-fps", type=float, default=5.0)
    video_parser.add_argument("--models", help="Legacy: comma-separated model names")
    video_parser.add_argument("--show-contexts", action="store_true")
    video_parser.add_argument("--fail-on-no-alarm", action="store_true")
    return parser


def _is_dynamic(config_path: str) -> bool:
    return config_pipeline(config_path) in {"dynamic", "dynamic_v2"}


def run_image(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "OpenCV is required. Install with: pip install -e '.[runtime]'"
        ) from exc

    frame = cv2.imread(args.source)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {args.source}")
    output_path = Path(args.output)
    json_path = Path(args.json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_dynamic(args.config):
        if args.models:
            raise ValueError("--models is only supported by legacy configurations")
        config = load_dynamic_config(args.config)
        pipeline = DynamicHelmetDetectionPipeline(config)
        result = pipeline.detect_frame(
            frame,
            timestamp_seconds=0.0,
            persist_tracks=False,
            apply_temporal=False,
        )
        annotated = annotate_dynamic_frame(
            frame,
            result,
            config=config,
            show_contexts=args.show_contexts,
        )
        data = result.to_dict()
        alarm = result.alarm
    else:
        config = load_config(args.config)
        pipeline = HelmetDetectionPipeline(
            config,
            enabled_models=_parse_models(args.models),
        )
        result = pipeline.detect_frame(frame)
        annotated = annotate_frame(frame, result, roi=config.roi, gate=config.gate)
        data = result.to_dict()
        alarm = result.alarm

    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Cannot write image: {output_path}")
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(data, ensure_ascii=False))
    return 2 if args.fail_on_no_alarm and not alarm else 0


def run_video(args: argparse.Namespace) -> int:
    if _is_dynamic(args.config):
        if args.models:
            raise ValueError("--models is only supported by legacy configurations")
        config = load_dynamic_config(args.config)
        pipeline = DynamicHelmetDetectionPipeline(config)
        summary = process_dynamic_video(
            pipeline,
            args.source,
            args.output,
            args.records,
            sample_fps=args.sample_fps,
            show_contexts=args.show_contexts,
        )
        success = summary.events > 0
    else:
        config = load_config(args.config)
        pipeline = HelmetDetectionPipeline(
            config,
            enabled_models=_parse_models(args.models),
        )
        summary = process_video(
            pipeline,
            args.source,
            args.output,
            args.records,
            sample_fps=args.sample_fps,
        )
        success = summary.events > 0

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False))
    return 2 if args.fail_on_no_alarm and not success else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "image":
        return run_image(args)
    if args.command == "video":
        return run_video(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
