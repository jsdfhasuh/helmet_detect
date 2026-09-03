#!/usr/bin/env python3
"""Download and verify the PT weights used by the dynamic V2 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelSource:
    key: str
    url: str
    filename: str
    sha256: str
    role: str
    license_note: str


MODEL_SOURCES = {
    "scene_yolo11n": ModelSource(
        key="scene_yolo11n",
        url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        filename="yolo11n.pt",
        sha256="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        role="COCO person/bicycle/motorcycle detection and ByteTrack input",
        license_note="Ultralytics model/runtime licensing must be reviewed for deployment.",
    ),
    "helmet_yolov8n": ModelSource(
        key="helmet_yolov8n",
        url=(
            "https://huggingface.co/iam-tsr/yolov8n-helmet-detection/resolve/main/"
            "best.pt?download=true"
        ),
        filename="helmet_yolov8n.pt",
        sha256="c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d",
        role="Per-person With Helmet / Without Helmet detection",
        license_note=(
            "The model card currently marks the repository MIT; "
            "verify training data rights."
        ),
    ),
    "rider_yolo11m_optional": ModelSource(
        key="rider_yolo11m_optional",
        url=(
            "https://huggingface.co/nnsohamnn/helmet-detection-yolo11/resolve/main/"
            "yolov11m%28100epochs%29.pt?download=true"
        ),
        filename="rider_yolo11m.pt",
        sha256="5769c45395a73739cb35cf28ed16e5d0acc3a0c90e1fac588fdb8a7403b3a930",
        role="Optional comparison model; not enabled by the V2 default configuration",
        license_note=(
            "The model card currently marks the repository MIT; "
            "verify training data rights."
        ),
    ),
}
DEFAULT_MODELS = ("scene_yolo11n", "helmet_yolov8n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source: ModelSource, destination: Path, *, force: bool) -> None:
    if destination.is_file() and not force:
        actual = sha256_file(destination)
        if actual == source.sha256:
            print(f"Using verified model: {destination}")
            return
        print(f"Checksum mismatch for cached {destination}; downloading again")

    destination.parent.mkdir(parents=True, exist_ok=True)
    request_headers = {
        "User-Agent": "helmet-detect/0.2 (+https://github.com/jsdfhasuh/helmet_detect)"
    }
    last_error: Exception | None = None
    for attempt in range(1, 6):
        temporary_path: Path | None = None
        try:
            request = urllib.request.Request(source.url, headers=request_headers)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                with urllib.request.urlopen(request, timeout=180) as response:
                    shutil.copyfileobj(response, temporary, length=1024 * 1024)
            actual = sha256_file(temporary_path)
            if actual != source.sha256:
                raise RuntimeError(
                    f"SHA256 mismatch for {source.key}: expected {source.sha256}, got {actual}"
                )
            temporary_path.replace(destination)
            print(
                f"Downloaded: {destination} "
                f"({destination.stat().st_size / 1024 / 1024:.1f} MiB)"
            )
            return
        except Exception as exc:  # noqa: BLE001 - bounded download retry
            last_error = exc
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            print(f"Download attempt {attempt}/5 failed: {exc}", file=sys.stderr)
            time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"Failed to download {source.key}") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated keys: {','.join(MODEL_SOURCES)}",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(MODEL_SOURCES))
    if unknown:
        raise SystemExit(f"Unknown model key(s): {', '.join(unknown)}")

    metadata: dict[str, object] = {}
    checksum_lines: list[str] = []
    for key in selected:
        source = MODEL_SOURCES[key]
        destination = output_dir / source.filename
        download(source, destination, force=args.force)
        checksum_lines.append(f"{source.sha256}  {source.filename}")
        metadata[key] = asdict(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dynamic_model_sources.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "DYNAMIC_SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
