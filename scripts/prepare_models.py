#!/usr/bin/env python3
"""Download the public PT weights and export fixed 960x960 ONNX files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelSource:
    key: str
    url: str
    pt_name: str
    onnx_name: str
    sha256: str


MODEL_SOURCES = {
    "helmet_yolov8n": ModelSource(
        key="helmet_yolov8n",
        url=(
            "https://huggingface.co/iam-tsr/yolov8n-helmet-detection/resolve/main/"
            "best.pt?download=true"
        ),
        pt_name="helmet_yolov8n.pt",
        onnx_name="helmet_yolov8n_960.onnx",
        sha256="c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d",
    ),
    "rider_yolo11m": ModelSource(
        key="rider_yolo11m",
        url=(
            "https://huggingface.co/nnsohamnn/helmet-detection-yolo11/resolve/main/"
            "yolov11m%28100epochs%29.pt?download=true"
        ),
        pt_name="rider_yolo11m.pt",
        onnx_name="rider_yolo11m_960.onnx",
        sha256="5769c45395a73739cb35cf28ed16e5d0acc3a0c90e1fac588fdb8a7403b3a930",
    ),
}


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
            print(f"Using cached PT model: {destination}")
            return
        print(f"Checksum mismatch for cached {destination}; downloading again")

    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "helmet-detect/0.1 (+https://github.com/jsdfhasuh/helmet_detect)"}
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(source.url, headers=headers)
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
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SHA256 mismatch for {source.key}: expected {source.sha256}, got {actual}"
                )
            temporary_path.replace(destination)
            print(f"Downloaded: {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MiB)")
            return
        except Exception as exc:  # noqa: BLE001 - retry boundary
            last_error = exc
            print(f"Download attempt {attempt}/5 failed: {exc}", file=sys.stderr)
            time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"Failed to download {source.key}") from last_error


def export_onnx(source: ModelSource, pt_path: Path, output_path: Path, *, force: bool) -> None:
    if output_path.is_file() and not force:
        print(f"Using cached ONNX model: {output_path}")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics export dependencies are missing. "
            "Install with: pip install -e '.[export]'"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pt_path))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=960,
            opset=12,
            simplify=True,
            dynamic=False,
            batch=1,
            device="cpu",
        )
    )
    if exported.resolve() != output_path.resolve():
        shutil.move(str(exported), output_path)
    print(f"Exported: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MiB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models", help="Destination model directory")
    parser.add_argument(
        "--models",
        default=",".join(MODEL_SOURCES),
        help=f"Comma-separated keys: {','.join(MODEL_SOURCES)}",
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(MODEL_SOURCES))
    if unknown:
        raise SystemExit(f"Unknown model key(s): {', '.join(unknown)}")

    metadata: dict[str, dict[str, str | int]] = {}
    checksums: list[str] = []
    for key in selected:
        source = MODEL_SOURCES[key]
        pt_path = output_dir / source.pt_name
        onnx_path = output_dir / source.onnx_name
        download(source, pt_path, force=args.force)
        if not args.download_only:
            export_onnx(source, pt_path, onnx_path, force=args.force)
            checksums.append(f"{sha256_file(onnx_path)}  {onnx_path.name}")
        metadata[key] = {
            "source_url": source.url,
            "pt_file": source.pt_name,
            "pt_sha256": source.sha256,
            "onnx_file": source.onnx_name,
            "input_size": 960,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_sources.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if checksums:
        (output_dir / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
