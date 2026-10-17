from pathlib import Path

import pytest
from helpers import create_pdfs, outline_titles, write_layout

from tunery.render import render

SHEET_MUSIC = [
    "Poinciana/Poinciana.pdf",
    "Poinciana/Poinciana.Bb.Emaj.pdf",
    "_Petes/Poinciana/Poinciana - concert.pdf",
    "_Petes/Poinciana/Poinciana - bass.pdf",
    "Creepin/Creepin.pdf",
    "Creepin/Creepin.iRealPro.pdf",
    "Take Five/Take Five.pdf",
    "Take Five/Take Five.Bb.pdf",
    "Bb Parts/Take Five Bb Tenor.pdf",
    "I Cant Help It/I Cant Help It.pdf",
    "_Petes/I Can't Help It/I Can't Help It - Tenor.pdf",
    "_Petes/I Can't Help It/I Can't Help It - Bass.pdf",
]


def render_title(
    root: Path, title: str, part: str, capsys: pytest.CaptureFixture[str]
) -> tuple[str, list[tuple[str, list[str]]]]:
    create_pdfs(root / "SheetMusic", SHEET_MUSIC)
    (root / "tunery.yaml").write_text("- library: SheetMusic\n")
    layout = write_layout(root / "gig.yaml", [{"title": title}])
    capsys.readouterr()
    render(layout, root / "book.pdf", part=[part], verbose=1)
    label = "-".join(p.strip() for p in part.split(","))
    book = root / ("book.pdf" if label == "Concert" else f"book.{label}.pdf")
    return capsys.readouterr().out, outline_titles(book)


def test_top_source_supplies_substituted_parts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, tree = render_title(tmp_path, "Poinciana", "Concert,Trumpet,Rhythm", capsys)

    # _Petes has Concert and Bass at their own step, so its concert chart wins;
    # Trumpet's best rank (Bb) exists only in SheetMusic/Poinciana.
    assert tree == [("Poinciana", ["Concert", "Bb", "Bass"])]
    assert '"Poinciana" Concert: Poinciana - concert.pdf' in out
    assert '"Poinciana" Trumpet: Poinciana.Bb.Emaj.pdf' in out
    assert out.count('warning "Poinciana"') == 1
    assert "charts from 2 folders: Poinciana, _Petes/Poinciana" in out


def test_noise_words_lose_and_warn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, tree = render_title(tmp_path, "Creepin", "Concert", capsys)

    assert tree == [("Creepin", [])]
    assert '"Creepin" Concert: Creepin.pdf' in out
    assert "Concert: Creepin/Creepin.pdf (also Creepin/Creepin.iRealPro.pdf)" in out


def test_key_group_ties_with_instrument(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, _ = render_title(tmp_path, "Take Five", "Tenor Sax", capsys)

    assert '"Take Five" Tenor Sax: Take Five Bb Tenor.pdf' in out
    assert (
        "Tenor Sax: Bb Parts/Take Five Bb Tenor.pdf (also Take Five/Take Five.Bb.pdf)"
        in out
    )
    assert out.count("warning") == 1


def test_stray_source_is_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, tree = render_title(tmp_path, "I Can't Help It", "Concert,Tenor Sax", capsys)

    assert tree == [("I Can't Help It", ["Concert", "Tenor Sax"])]
    assert "charts from 2 folders: I Cant Help It, _Petes/I Can't Help It" in out
    assert out.count("warning") == 1


def test_part_itself_beats_its_key_group_at_a_rank(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(
        tmp_path / "SheetMusic",
        [
            "Blues/Blues.pdf",
            "Blues/Blues.Bb.pdf",
            "_Petes/Blues/Blues - tenor (horn 2).pdf",
        ],
    )
    (tmp_path / "tunery.yaml").write_text("- library: SheetMusic\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Blues"}])

    render(layout, tmp_path / "book.pdf", part=["Concert,Tenor Sax"], verbose=1)

    assert '"Blues" Tenor Sax: Blues - tenor (horn 2).pdf' in capsys.readouterr().out


def test_fewer_substitution_steps_win_within_a_rank(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Alto Sax 2 -> [Alto Sax 2], [Alto Sax, Eb]: Alto Sax is one substitution,
    # Eb two; path order alone would pick the Eb chart in folder A.
    create_pdfs(tmp_path / "SheetMusic", ["A/Tune - Eb.pdf", "B/Tune - Alto Sax.pdf"])
    (tmp_path / "tunery.yaml").write_text("- library: SheetMusic\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Tune"}])

    render(layout, tmp_path / "book.pdf", part=["Alto Sax 2"], verbose=1)

    assert '"Tune" Alto Sax 2: Tune - Alto Sax.pdf' in capsys.readouterr().out
