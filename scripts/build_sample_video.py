#!/usr/bin/env python3
"""Build the repository's compact real-camera sample video."""

from __future__ import annotations

import argparse

from helmet_detect.fixtures import load_manifest_and_materialize
from helmet_detect.sample import build_video_from_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="testdata/camera_exit/manifest.json",
    )
    parser.add_argument("--output", default="artifacts/sample/no_helmet_event.avi")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path, manifest = load_manifest_and_materialize(args.manifest)
    case = manifest["video_case"]
    frame_paths = [manifest_path.parent / item for item in case["frames"]]
    output = build_video_from_images(
        frame_paths,
        args.output,
        fps=float(case.get("fps", 6.0)),
        repeat_each=int(case.get("repeat_each", 1)),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
