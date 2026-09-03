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
    encoded_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    output_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> Path:
    """Decode one or more base64 text parts and optionally verify SHA256."""

    raw_sources = (
        list(encoded_path)
        if isinstance(encoded_path, (list, tuple))
        else [encoded_path]
    )
    sources = [Path(item) for item in raw_sources]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Encoded fixture not found: {source}")
    destination = Path(output_path)
    compact = "".join(
        "".join(source.read_text(encoding="ascii").split()) for source in sources
    )
    try:
        payload = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as exc:
        names = ", ".join(str(source) for source in sources)
        raise ValueError(f"Invalid base64 fixture: {names}") from exc

    actual_sha256 = sha256_bytes(payload)
    if expected_sha256 and actual_sha256 != expected_sha256.lower():
        raise ValueError(
            f"Fixture SHA256 mismatch for {destination}: "
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
        encoded_parts = specification.get("base64_parts")
        if encoded_relative and encoded_parts:
            raise ValueError(
                f"Fixture cannot define both base64 and base64_parts: {output_relative}"
            )
        if encoded_parts:
            if not isinstance(encoded_parts, list) or not encoded_parts:
                raise ValueError(
                    f"Fixture base64_parts must be a non-empty list: {output_relative}"
                )
            encoded_sources: str | list[Path] = [
                root / str(item) for item in encoded_parts
            ]
        elif encoded_relative:
            encoded_sources = str(root / str(encoded_relative))
        else:
            raise ValueError(f"Fixture is missing its base64 path: {output_relative}")
        outputs.append(
            materialize_base64_file(
                encoded_sources,
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
