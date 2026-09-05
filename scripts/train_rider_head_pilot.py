#!/usr/bin/env python3
"""Train an opt-in rider/head prototype on a PRIVATE local YOLO dataset.

Nothing is uploaded. Fitting diagnostics are not independent accuracy evidence.
The final epoch, not a checkpoint chosen on the evaluation video, is delivered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

NAMES = {0: "rider", 1: "helmet_head", 2: "no_helmet_head"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Explicitly allow train=val for a fitting experiment",
    )
    args = parser.parse_args()
    if not args.data.is_file() or not args.base_weights.is_file():
        parser.error("Dataset YAML and base weights must already exist locally")
    if args.output.exists():
        parser.error("Choose a new output directory; do not overwrite another experiment")
    if args.epochs <= 0 or args.cpu_threads <= 0:
        parser.error("epochs and cpu-threads must be positive")

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO, settings

    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = data.get("names", {})
    names = dict(enumerate(names)) if isinstance(names, list) else names
    if names != NAMES:
        parser.error(f"Expected class contract {NAMES}, not {names}")
    if not args.fit_only and data.get("train") == data.get("val"):
        parser.error("train=val is only allowed with --fit-only; never report it as validation")
    settings.update({"sync": False})
    torch.set_num_threads(args.cpu_threads)
    model = YOLO(str(args.base_weights.resolve()))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=640,
        batch=8,
        workers=0,
        device=args.device,
        freeze=10,
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.1,
        warmup_epochs=3,
        nbs=8,
        seed=731,
        deterministic=True,
        amp=False,
        plots=False,
        val=False,
        save=True,
        patience=0,
        project=str(args.output.parent.resolve()),
        name=args.output.name,
        exist_ok=False,
        cache=False,
        mosaic=0.5,
        close_mosaic=10,
        fliplr=0.5,
        scale=0.35,
        translate=0.15,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=5,
    )
    output = Path(model.trainer.save_dir)
    final = output / "weights" / "last.pt"
    report = {
        "status": "EXPERIMENTAL_NOT_PRODUCTION_ACCEPTED",
        "selected_checkpoint": "last.pt",
        "fit_only": args.fit_only,
        "base_sha256": digest(args.base_weights),
        "data_yaml_sha256": digest(args.data),
        "weight_sha256": digest(final),
        "classes": NAMES,
        "epochs": args.epochs,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "note": "Check event/person/date separation independently. YAML path inequality "
        "alone does not establish an independent validation split.",
    }
    (output / "pilot_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
