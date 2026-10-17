"""Chart `path:` prefers candidates whose library path matches (LIB8.5)."""

from pathlib import Path

import pytest
from helpers import create_pdfs, write_layout

from tunery.render import render

# Names copied from the JBM Vault, plus `All of Me/All of Me-Piano.pdf`,
# which path order and the top source pick when nothing else is asked.
FILES = [
    "All of Me/All of Me-Piano.pdf",
    "Tunes for Michelle/All of Me (Vocal)-Piano.pdf",
    "Tunes for Michelle/All of Me (Vocal)-Piano.v2.pdf",
    "Vocals/All Of Me/All of Me (Vocal)-Piano.pdf",
    "Vocals/All Of Me/All of Me (Vocal)-Piano.v2.pdf",
]


def render_all_of_me(root: Path, chart: dict, capsys: pytest.CaptureFixture[str]) -> str:
    create_pdfs(root / "Vault", FILES)
    (root / "tunery.yaml").write_text("- library: Vault\n")
    layout = write_layout(root / "show.yaml", [chart])
    capsys.readouterr()
    render(layout, root / "show.pdf", part=["Piano"], verbose=1)
    return capsys.readouterr().out


def found_line(out: str) -> str:
    lines = [line for line in out.splitlines() if line.startswith('found    "All of Me"')]
    assert lines == [lines[0]]
    return lines[0]


def test_path_vocals_prefers_that_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = render_all_of_me(tmp_path, {"title": "All of Me", "path": "vocals"}, capsys)

    found = found_line(out)
    assert "Vocals" in found and "All Of Me" in found
    assert found.endswith("(All of Me (Vocal)-Piano)")
    assert "no candidate under" not in out


def test_paths_list_matches_any_fragment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = render_all_of_me(
        tmp_path,
        {"title": "All of Me", "paths": ["Missing", "Vocals"]},
        capsys,
    )

    found = found_line(out)
    assert "Vocals" in found
    assert found.endswith("(All of Me (Vocal)-Piano)")
    assert "no candidate under" not in out


def test_path_glob_prefers_tunes_for_michelle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = render_all_of_me(
        tmp_path, {"title": "All of Me", "path": "Tunes*"}, capsys
    )

    found = found_line(out)
    assert "Tunes for Michelle" in found
    assert found.endswith("(All of Me (Vocal)-Piano)")


def test_unmatched_path_warns_and_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = render_all_of_me(tmp_path, {"title": "All of Me", "path": "Nowhere"}, capsys)

    assert 'no candidate under "Nowhere" for "All of Me"' in out
    assert found_line(out).endswith("(All of Me-Piano)")


def test_path_and_tag_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = render_all_of_me(
        tmp_path,
        {"title": "All of Me", "path": "Vocals", "tag": "v2"},
        capsys,
    )

    found = found_line(out)
    assert "Vocals" in found
    assert found.endswith("(All of Me (Vocal)-Piano.v2)")
    assert "no candidate under" not in out
    assert "no v2 version" not in out
