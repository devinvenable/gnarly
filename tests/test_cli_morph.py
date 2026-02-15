from pathlib import Path

import pytest

from gnarly.cli import parse_args


def test_cli_parses_morph_flags(tmp_path: Path):
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"fake")

    config = parse_args(
        [
            str(input_path),
            "-o",
            "out.mp4",
            "--morph",
            "--morph-style",
            "cyberpunk",
            "--creativity",
            "0.8",
        ]
    )

    assert config.morph.enabled is True
    assert config.morph.style == "cyberpunk"
    assert config.morph.creativity == 0.8


def test_cli_rejects_invalid_creativity(tmp_path: Path):
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"fake")

    with pytest.raises(SystemExit):
        parse_args(
            [
                str(input_path),
                "-o",
                "out.mp4",
                "--creativity",
                "1.5",
            ]
        )
