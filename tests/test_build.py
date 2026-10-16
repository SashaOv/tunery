import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "build", Path(__file__).parent.parent / "build.py"
)
assert SPEC and SPEC.loader
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


def test_gate_passes_when_every_check_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert build.gate([["json.tool", "--help"], ["json.tool", "--help"]]) == 0
    assert capsys.readouterr().out.count("$ json.tool --help") == 2


def test_gate_stops_at_first_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = str(tmp_path / "missing.json")

    returncode = build.gate([["json.tool", missing], ["json.tool", "--help"]])

    out = capsys.readouterr().out
    assert returncode != 0
    assert f"gate failed: json.tool {missing} exited with {returncode}" in out
    assert "$ json.tool --help" not in out
