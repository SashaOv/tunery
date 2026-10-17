"""Concert = Lead Sheet = untagged = Rhythm chart, one rank (LIB6.4)."""

import json
from pathlib import Path

import pytest
from helpers import create_pdfs, write_layout

from tunery.render import render


def test_funk_concert_chart_is_rhythm_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(
        tmp_path / "SheetMusic",
        [
            "Funk/Funk - Rhythm Section.pdf",
            "Funk/Funk - Tenor.pdf",
        ],
    )
    (tmp_path / "tunery.yaml").write_text("- library: SheetMusic\n")
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Funk"}])

    render(layout, tmp_path / "show.pdf", verbose=1)

    assert '"Funk" Concert: Funk - Rhythm Section.pdf' in capsys.readouterr().out


def test_recado_takes_rhythm_before_real_book(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(tmp_path / "SheetMusic", ["Recado/Recado - Rhythm.pdf"])
    create_pdfs(tmp_path / "books", ["RealBook.pdf"])
    (tmp_path / "book.json").write_text(
        json.dumps([{"title": "Recado", "page": 1}])
    )
    (tmp_path / "tunery.yaml").write_text(
        "- library: book.json\n  source: books/RealBook.pdf\n- library: SheetMusic\n"
    )
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Recado"}])

    render(layout, tmp_path / "show.pdf", verbose=1)

    assert '"Recado" Concert: Recado - Rhythm.pdf' in capsys.readouterr().out


def test_i_cant_help_it_concert_chart_is_piano(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(
        tmp_path / "SheetMusic",
        [
            "I Cant Help It/I Cant Help It - Piano.pdf",
            "I Cant Help It/I Cant Help It - Bass.pdf",
            "I Cant Help It/I Cant Help It - Drums.pdf",
        ],
    )
    (tmp_path / "tunery.yaml").write_text("- library: SheetMusic\n")
    layout = write_layout(tmp_path / "show.yaml", [{"title": "I Cant Help It"}])

    render(layout, tmp_path / "show.pdf", verbose=1)

    assert '"I Cant Help It" Concert: I Cant Help It - Piano.pdf' in capsys.readouterr().out


def test_cape_verdean_piano_part_is_rhythm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(
        tmp_path / "SheetMusic",
        [
            "Cape Verdean/Cape Verdean - rhythm.pdf",
            "Cape Verdean/Cape Verdean - Concert.pdf",
        ],
    )
    (tmp_path / "tunery.yaml").write_text("- library: SheetMusic\n")
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Cape Verdean"}])

    render(layout, tmp_path / "show.pdf", part=["Piano"], verbose=1)

    out = capsys.readouterr().out
    assert '"Cape Verdean" Piano: Cape Verdean - rhythm.pdf' in out
    assert "Cape Verdean - Concert.pdf" not in out


def test_concert_falls_back_to_melody_before_piano_and_vocal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(
        tmp_path / "SheetMusic",
        [
            "Tune/Tune - Piano.pdf",
            "Tune/Tune - Melody.pdf",
            "Tune/Tune - Vocal.pdf",
        ],
    )
    (tmp_path / "tunery.yaml").write_text("- library: SheetMusic\n")
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Tune"}])

    render(layout, tmp_path / "show.pdf", verbose=1)

    out = capsys.readouterr().out
    assert '"Tune" Concert: Tune - Melody.pdf' in out
    assert "Tune: Concert → Melody" in out


def test_girl_from_ipanema_takes_voice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(tmp_path / "SheetMusic", ["Girl/Girl-Voice.pdf"])
    (tmp_path / "tunery.yaml").write_text("- library: SheetMusic\n")
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Girl"}])

    render(layout, tmp_path / "show.pdf", verbose=0)

    out = capsys.readouterr().out
    assert '"Girl" Concert: Girl-Voice.pdf' not in out
    assert "Girl: Concert → Vocal" in out


def test_flute_no_longer_substitutes_for_concert(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(tmp_path / "SheetMusic", ["Tune/Tune - Flute.pdf"])
    (tmp_path / "tunery.yaml").write_text("- library: SheetMusic\n")
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Tune"}])

    render(layout, tmp_path / "show.pdf")

    out = capsys.readouterr().out
    assert 'missing Concert for "Tune"' in out
    assert 'not found "Tune"' in out
