"""Materialize compact base64-encoded regression fixtures."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA256 digest for ``payload``."""

    return hashlib.sha256(payload).hexdigest()


def materialize_base64_file(
    encoded_path: str | Path,
    output_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Path:
    """Decode one base64 text file atomically and optionally verify its SHA256."""

    source = Path(encoded_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Encoded fixture not found: {source}")

    compact = "".join(source.read_text(encoding="ascii").split())
    try:
        payload = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 fixture: {source}") from exc

    actual_sha256 = sha256_bytes(payload)
    if expected_sha256 and actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"Fixture SHA256 mismatch for {source}: "
            f"expected {expected_sha256.lower()}, got {actual_sha256}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination


def materialize_manifest_fixtures(
    manifest: dict[str, Any],
    manifest_path: str | Path,
) -> list[Path]:
    """Materialize every fixture declared by a regression manifest."""

    root = Path(manifest_path).resolve().parent
    outputs: list[Path] = []
    for output_relative, specification in manifest.get("fixtures", {}).items():
        if not isinstance(specification, dict):
            raise ValueError(f"Fixture specification must be an object: {output_relative}")
        encoded_relative = specification.get("base64")
        if not encoded_relative:
            raise ValueError(f"Fixture is missing its base64 path: {output_relative}")
        outputs.append(
            materialize_base64_file(
                root / str(encoded_relative),
                root / str(output_relative),
                expected_sha256=(
                    str(specification["sha256"]) if specification.get("sha256") else None
                ),
            )
        )
    return outputs


def load_manifest_and_materialize(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Read a JSON manifest and materialize all declared fixtures."""

    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    materialize_manifest_fixtures(manifest, manifest_path)
    return manifest_path, manifest
