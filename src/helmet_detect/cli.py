"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
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
        description="Detect motorcycle/e-bike riders without safety helmets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="Process one image")
    image_parser.add_argument("--config", required=True)
    image_parser.add_argument("--source", required=True)
    image_parser.add_argument("--output", required=True)
    image_parser.add_argument("--json", dest="json_path", required=True)
    image_parser.add_argument("--models", help="Comma-separated configured model names")
    image_parser.add_argument("--fail-on-no-alarm", action="store_true")

    video_parser = subparsers.add_parser("video", help="Process a video")
    video_parser.add_argument("--config", required=True)
    video_parser.add_argument("--source", required=True)
    video_parser.add_argument("--output", required=True)
    video_parser.add_argument("--records", required=True)
    video_parser.add_argument("--summary", required=True)
    video_parser.add_argument("--sample-fps", type=float, default=5.0)
    video_parser.add_argument("--models", help="Comma-separated configured model names")
    video_parser.add_argument("--fail-on-no-alarm", action="store_true")
    return parser


def run_image(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("OpenCV is required. Install with: pip install -e '.[runtime]'") from exc

    config = load_config(args.config)
    pipeline = HelmetDetectionPipeline(config, enabled_models=_parse_models(args.models))
    frame = cv2.imread(args.source)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {args.source}")
    result = pipeline.detect_frame(frame)

    output_path = Path(args.output)
    json_path = Path(args.json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = annotate_frame(
        frame,
        result,
        roi=config.roi,
        gate=config.gate,
    )
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Cannot write image: {output_path}")
    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False))
    return 2 if args.fail_on_no_alarm and not result.alarm else 0


def run_video(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pipeline = HelmetDetectionPipeline(config, enabled_models=_parse_models(args.models))
    summary = process_video(
        pipeline,
        args.source,
        args.output,
        args.records,
        sample_fps=args.sample_fps,
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False))
    return 2 if args.fail_on_no_alarm and summary.events == 0 else 0


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
