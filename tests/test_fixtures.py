from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from helmet_detect.fixtures import materialize_base64_file


def test_materialize_base64_file_decodes_and_verifies(tmp_path: Path) -> None:
    payload = b"helmet-fixture"
    encoded = tmp_path / "fixture.bin.b64"
    encoded.write_text(base64.b64encode(payload).decode("ascii"), encoding="ascii")
    output = tmp_path / "fixture.bin"

    result = materialize_base64_file(
        encoded,
        output,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert result == output
    assert output.read_bytes() == payload


def test_materialize_base64_file_rejects_checksum_mismatch(tmp_path: Path) -> None:
    encoded = tmp_path / "fixture.bin.b64"
    encoded.write_text(base64.b64encode(b"wrong").decode("ascii"), encoding="ascii")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        materialize_base64_file(encoded, tmp_path / "fixture.bin", expected_sha256="0" * 64)
