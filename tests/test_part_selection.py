from pathlib import Path

import json
import pikepdf
import pytest
import yaml

from tunery.render import render


def create_pdf(path: Path, page_count: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pikepdf.Pdf.new()
    for _ in range(page_count):
        pdf.add_blank_page(page_size=(100, 100))
    pdf.save(path)
    pdf.close()
    return path


def write_layout(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(list(records), sort_keys=False))
    return path


def make_vault(root: Path):
    v = root / "Vault"
    create_pdf(v / "Aint Misbehavin - Concert F" / "Aint Misbehavin-Piano.pdf", 2)
    create_pdf(v / "Aint Misbehavin - Concert Bb" / "Aint Misbehavin-Piano.pdf", 3)
    create_pdf(v / "Tunes for Michelle" / "Aint Misbehavin-Piano.pdf", 1)
    create_pdf(v / "Mo Better Blues" / "Mo Better Blues-Piano.pdf", 1)
    create_pdf(v / "Mo Better Blues" / "Mo Better Blues-Piano_(1).pdf", 1)
    create_pdf(v / "All of Me" / "All of Me-Piano.pdf", 1)
    create_pdf(v / "Tunes for Michelle" / "All of Me-Piano.pdf", 2)
    create_pdf(v / "Girl From Ipanema" / "Girl From Ipanema-Concert.pdf", 1)
    create_pdf(v / "Tunes for Michelle" / "Girl From Ipanema-Piano.pdf", 1)
    create_pdf(v / "Mack the Knife" / "Mack the Knife-Score-Piano.pdf", 2)
    create_pdf(v / "Mack the Knife" / "Mack the Knife-Guitar.pdf", 3)
    create_pdf(v / "Mack the Knife" / "Mack the Knife-Bass.pdf", 1)
    create_pdf(v / "Straighten Up" / "Straighten Up-Piano.pdf", 1)
    create_pdf(v / "Straighten Up" / "Straighten Up-Electric_Guitar.pdf", 1)
    create_pdf(v / "Heartaches" / "Heartaches-Concert.pdf", 4)
    create_pdf(v / "Norwegian Wood" / "Norwegian Wood-Horn 1.pdf", 1)
    create_pdf(v / "Norwegian Wood" / "Norwegian Wood-Horn 2.pdf", 1)
    return v


def test_piano_prefers_piano_and_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vault = make_vault(tmp_path)
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n  match: exact\n")
    layout = write_layout(
        tmp_path / "gig.yaml",
        [{"title": "Mack the Knife"}, {"title": "Heartaches"}],
    )
    out = tmp_path / "book.pdf"

    render(layout, out, part="Piano")

    out_part = tmp_path / "book.Piano.pdf"
    assert out_part.exists()
    with pikepdf.Pdf.open(out_part) as pdf:
        # Mack Piano (2 pages) + Heartaches Concert fallback (4 pages)
        assert len(pdf.pages) == 6
    assert 'found    "Mack the Knife"' in capsys.readouterr().out


def test_tie_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    vault = make_vault(tmp_path)
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n  match: exact\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Mo Better Blues"}])

    render(layout, tmp_path / "book.pdf", part="Piano")

    out = capsys.readouterr().out
    assert "tie" in out
    assert "Mo Better Blues" in out


def test_library_priority_vault_beats_index(tmp_path: Path) -> None:
    vault = make_vault(tmp_path)
    book = create_pdf(tmp_path / "books" / "RealBook.pdf", 5)
    assert book.exists()
    (tmp_path / "book.json").write_text(
        json.dumps([{"title": "Heartaches", "page": 2, "pages": 2}])
    )
    (tmp_path / "tunery.yaml").write_text(
        "- library: book.json\n  source: books/RealBook.pdf\n- library: Vault\n"
    )
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Heartaches"}])

    render(layout, tmp_path / "book.pdf", part="Piano")

    # Vault Concert (4 pages) beats index (2 pages).
    with pikepdf.Pdf.open(tmp_path / "book.Piano.pdf") as pdf:
        assert len(pdf.pages) == 4


def test_indexed_answers_part_request(tmp_path: Path) -> None:
    create_pdf(tmp_path / "books" / "RealBook.pdf", 6)
    (tmp_path / "book.json").write_text(
        json.dumps([{"title": "Blue Monk", "page": 2, "pages": 2}])
    )
    (tmp_path / "tunery.yaml").write_text(
        "- library: book.json\n  source: books/RealBook.pdf\n"
    )
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Blue Monk"}])

    render(layout, tmp_path / "book.pdf", part="Piano")

    with pikepdf.Pdf.open(tmp_path / "book.Piano.pdf") as pdf:
        assert len(pdf.pages) == 2


def test_guitar_chain_uses_piano_fallback(tmp_path: Path) -> None:
    vault = make_vault(tmp_path)
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n")
    # Heartaches has only Concert; Guitar -> Piano -> Rhythm -> Concert.
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Heartaches"}])

    render(layout, tmp_path / "book.pdf", part="Guitar")

    assert (tmp_path / "book.Guitar.pdf").exists()
    with pikepdf.Pdf.open(tmp_path / "book.Guitar.pdf") as pdf:
        assert len(pdf.pages) == 4


def test_horn_role_selects_exact(tmp_path: Path) -> None:
    vault = make_vault(tmp_path)
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Norwegian Wood"}])

    render(layout, tmp_path / "book.pdf", part="Horn 1")

    assert (tmp_path / "book.Horn 1.pdf").exists()


def test_bass_does_not_match_bass_clarinet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdf(tmp_path / "Vault" / "Tune" / "Tune-Piano.pdf")
    create_pdf(tmp_path / "Vault" / "Tune" / "Tune-Bass_Clarinet.pdf")
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Tune"}])

    render(layout, tmp_path / "book.pdf", part="Bass")

    assert 'missing Bass for "Tune"' in capsys.readouterr().out


def test_fuzzy_title_keeps_part_from_stem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "Vault" / "Si Tu Voir Ma Mere"
    for tag in ("Alto_Saxophone", "Bass_Clarinet", "C_Major"):
        create_pdf(folder / f"si-tu-vois-ma-mere-sidney-bechet-lead-sheet-{tag}.pdf")
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n  match: fuzzy\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Si Tu Vois Ma Mare"}])

    render(layout, tmp_path / "book.pdf", part="Piano", verbose=1)

    out = capsys.readouterr().out
    assert "lead-sheet-C_Major.pdf" in out
    assert "tie for" not in out
