from pathlib import Path

import pytest
from helpers import create_pdf, create_pdfs, write_layout

from tunery.parts import match_candidate, parse_score_key
from tunery.render import render


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("C", ("C", None)),
        ("c", ("C", None)),
        ("Bb", ("Bb", None)),
        ("F#", ("F#", None)),
        ("Gb", ("Gb", None)),
        ("F-sharp", ("F#", None)),
        *[(f"G{mode}", ("G", "minor")) for mode in ("m", "mi", "min", "minor")],
        *[(f"G {mode}", ("G", "minor")) for mode in ("m", "mi", "min", "minor")],
        *[(f"G{mode}", ("G", "major")) for mode in ("ma", "maj", "major")],
        *[(f"G {mode}", ("G", "major")) for mode in ("ma", "maj", "major")],
        ("Bbm", ("Bb", "minor")),
        ("Bb maj", ("Bb", "major")),
        ("Bm", ("B", "minor")),
        ("Em", ("E", "minor")),
        ("Ema", ("E", "major")),
        ("f# min", ("F#", "minor")),
        (" Dm ", ("D", "minor")),
    ],
)
def test_score_key_notation(spec: str, expected: tuple[str, str | None]) -> None:
    assert parse_score_key(spec) == expected


@pytest.mark.parametrize("spec", ["H", "", "Cx", "G mm", "E#", "minor", "Cb"])
def test_invalid_score_key_names_the_value(spec: str) -> None:
    with pytest.raises(ValueError, match="Invalid key"):
        parse_score_key(spec)


@pytest.mark.parametrize(
    ("stem", "part", "key", "mode"),
    [
        ("So Nice (Summer Samba)(C)", "Concert", "C", None),
        ("So Nice (Summer Samba) [C]", "Concert", "C", None),
        ("So Nice (Summer Samba) (Gm)", "Concert", "G", "minor"),
        ("So Nice (Summer Samba) [D min]", "Concert", "D", "minor"),
        ("So Nice (Summer Samba) (F# maj)", "Concert", "F#", "major"),
        # Bracketed Bb/Eb are score keys; unbracketed ones are instrument groups.
        ("So Nice (Summer Samba) (Eb)", "Concert", "Eb", None),
        ("So Nice (Summer Samba) [Bb]", "Concert", "Bb", None),
        ("So Nice (Summer Samba).Eb", "Eb", None, None),
        ("So Nice (Summer Samba) - Eb", "Eb", None, None),
        ("So Nice (Summer Samba) Bb", "Bb", None, None),
        ("So Nice (Summer Samba) - Eb (C)", "Eb", "C", None),
        ("So Nice (Summer Samba) - Alto Sax (F)", "Alto Sax", "F", None),
        # Existing notations keep their meaning.
        ("So Nice (Summer Samba) - Concert F", "Concert", "F", None),
        ("So Nice (Summer Samba) (in F)", "Concert", "F", None),
        ("So Nice (Summer Samba) - Concert A Minor", "Concert", "A", "minor"),
        # Brackets that are not a note name stay what they were.
        ("So Nice (Summer Samba) (Vocal)", "Vocal", None, None),
    ],
)
def test_bracketed_note_is_the_score_key(
    stem: str, part: str, key: str | None, mode: str | None
) -> None:
    candidate = match_candidate("So Nice (Summer Samba)", stem)

    assert candidate is not None
    assert (candidate.part, candidate.score_key, candidate.score_mode) == (
        part,
        key,
        mode,
    )
    assert candidate.free_tags == []


def test_explicit_key_beats_bracketed_key() -> None:
    candidate = match_candidate("Tune", "Tune - Concert F (C)")

    assert candidate is not None
    assert candidate.score_key == "F"


def test_folder_key_carries_mode() -> None:
    candidate = match_candidate("Summer Time", "Summertime - Score", "Summer Time - Concert A Minor")

    assert candidate is not None
    assert (candidate.score_key, candidate.score_mode) == ("A", "minor")


CHARTS = [
    "Tune/Tune (G).pdf",
    "Tune/Tune (Gm).pdf",
    "Tune/Tune (A).pdf",
    "Tune/Tune - Concert Bb Minor.pdf",
]


def pick(
    tmp_path: Path, key: str, capsys: pytest.CaptureFixture[str]
) -> tuple[str, str]:
    """Chart chosen for `key`, and the output."""
    create_pdfs(tmp_path / "Vault", CHARTS)
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Tune", "key": key}])
    capsys.readouterr()
    render(layout, tmp_path / "book.pdf", verbose=1)
    out = capsys.readouterr().out
    return next(line for line in out.splitlines() if line.startswith('  "Tune"')), out


@pytest.mark.parametrize(
    ("key", "chart"),
    [
        ("A", "Tune (A).pdf"),
        ("a", "Tune (A).pdf"),
        ("Bbm", "Tune - Concert Bb Minor.pdf"),
        ("Bb min", "Tune - Concert Bb Minor.pdf"),
        ("Gm", "Tune (Gm).pdf"),
        ("G minor", "Tune (Gm).pdf"),
        ("G maj", "Tune (G).pdf"),
        ("G major", "Tune (G).pdf"),
        ("Gma", "Tune (G).pdf"),
        ("Am", None),
        ("Bb maj", None),
        ("F", None),
    ],
)
def test_chart_key_matches_stated_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str, chart: str | None
) -> None:
    line, out = pick(tmp_path, key, capsys)

    if chart is None:
        assert 'version of "Tune"' in out
    else:
        assert line.endswith(f": {chart}")
        assert "version of" not in out


def test_chart_key_without_mode_matches_any_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(tmp_path / "Vault", ["Tune/Tune (Gm).pdf"])
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Tune", "key": "G"}])
    capsys.readouterr()

    render(layout, tmp_path / "book.pdf", verbose=1)

    out = capsys.readouterr().out
    assert "version of" not in out
    assert 'found    "Tune"' in out


def test_unmatched_minor_key_is_warned_with_its_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = pick(tmp_path, "Am", capsys)

    assert 'no A minor version of "Tune"' in out


def test_summer_samba_beside_the_layout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdf(tmp_path / "Combo3" / "So Nice (Summer Samba)(C).pdf")
    create_pdf(tmp_path / "Books" / "Real Book.pdf", 3)
    (tmp_path / "book.json").write_text('[{"title": "So Nice (Summer Samba)", "page": 2}]')
    (tmp_path / "tunery.yaml").write_text(
        "- library: book.json\n  source: Books/Real Book.pdf\n"
    )
    layout = write_layout(
        tmp_path / "Combo3" / "gig.yaml",
        [{"title": "So Nice (Summer Samba)", "key": "C"}],
    )
    capsys.readouterr()

    render(layout, tmp_path / "book.pdf", verbose=1)

    out = capsys.readouterr().out
    assert '"So Nice (Summer Samba)" Concert: So Nice (Summer Samba)(C).pdf' in out
    assert "version of" not in out
