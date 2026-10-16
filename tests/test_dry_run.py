"""`render --test TITLE`: resolve titles, print -vv results, write no PDF."""

from pathlib import Path

import pytest
from helpers import create_pdfs, write_layout

from tunery.render import render


def make_show(root: Path) -> Path:
    create_pdfs(
        root / "SheetMusic",
        [
            "A/A - Piano.pdf",
            "B/B - Piano.pdf",
            "C/C - Piano.pdf",
        ],
    )
    (root / "tunery.yaml").write_text("- library: SheetMusic\n")
    return write_layout(
        root / "show.yaml", [{"title": "A"}, {"title": "B"}, {"title": "C"}]
    )


def test_no_file_written(tmp_path: Path) -> None:
    layout = make_show(tmp_path)

    render(layout, tmp_path / "show.pdf", test=["A"])

    assert not (tmp_path / "show.pdf").exists()
    assert list(tmp_path.glob("show*.pdf")) == []


def test_repeated_test_prints_only_named_titles(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = make_show(tmp_path)

    render(layout, tmp_path / "show.pdf", test=["A", "B"])

    out = capsys.readouterr().out
    assert '"A"' in out
    assert '"B"' in out
    assert '"C"' not in out


def test_dot_and_empty_mean_every_title(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = make_show(tmp_path)

    render(layout, tmp_path / "show.pdf", test=["."])
    out = capsys.readouterr().out
    assert '"A"' in out and '"B"' in out and '"C"' in out

    render(layout, tmp_path / "show.pdf", test=[""])
    out = capsys.readouterr().out
    assert '"A"' in out and '"B"' in out and '"C"' in out


def test_unknown_title_is_an_error(tmp_path: Path) -> None:
    layout = make_show(tmp_path)

    with pytest.raises(ValueError, match="Z"):
        render(layout, tmp_path / "show.pdf", test=["Z"])


def test_works_with_several_part_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = make_show(tmp_path)

    render(layout, tmp_path / "show.pdf", part=["Piano", "Bass"], test=["A"])

    out = capsys.readouterr().out
    assert "candidate" in out
    assert list(tmp_path.glob("show*.pdf")) == []
