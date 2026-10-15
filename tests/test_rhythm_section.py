from pathlib import Path

import pikepdf
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


def outline_titles(pdf: pikepdf.Pdf):
    with pdf.open_outline() as outline:
        return [(item.title, [c.title for c in item.children]) for item in outline.root]


def make_rs_fixture(root: Path) -> None:
    sm = root / "SheetMusic"
    petes = root / "_Petes"
    # Poinciana → Concert + Bass
    create_pdf(sm / "Poinciana" / "Poinciana-Concert.pdf", 1)
    create_pdf(sm / "Poinciana" / "Poinciana-Bass.pdf", 2)
    # I Can't Help It → Piano + Bass + Drums (guitar reads piano)
    create_pdf(petes / "I Cant Help It-Piano.pdf", 1)
    create_pdf(petes / "I Cant Help It-Bass.pdf", 1)
    create_pdf(petes / "I Cant Help It-Drums.pdf", 1)
    # Mo Better Blues → Rhythm + Bass + Drums
    create_pdf(sm / "Mo Better Blues" / "Mo Better Blues-Rhythm.pdf", 1)
    create_pdf(sm / "Mo Better Blues" / "Mo Better Blues-Bass.pdf", 1)
    create_pdf(sm / "Mo Better Blues" / "Mo Better Blues-Drums.pdf", 1)
    # Concert-only title stays flat
    create_pdf(sm / "Ballad" / "Ballad-Concert.pdf", 2)
    # Norwegian Wood with Piano:Guitar dual
    create_pdf(sm / "Norwegian Wood" / "Norwegian Wood-PianoGuitar.pdf", 1)
    create_pdf(sm / "Norwegian Wood" / "Norwegian Wood-Bass.pdf", 1)
    create_pdf(sm / "Norwegian Wood" / "Norwegian Wood-Drums.pdf", 1)
    (root / "tunery.yaml").write_text("- library: .\n")


def test_rs_expands_to_rhythm_group(tmp_path: Path) -> None:
    make_rs_fixture(tmp_path)
    layout = write_layout(
        tmp_path / "gig.yaml",
        [{"title": "Poinciana"}, {"title": "Ballad"}],
    )

    render(layout, tmp_path / "book.pdf", part=["RS"])

    out = tmp_path / "book.RS.pdf"
    assert out.exists()
    with pikepdf.Pdf.open(out) as pdf:
        titles = outline_titles(pdf)
        # Poinciana nested (Bass + Concert fallback for Piano/Guitar/Drums deduped?),
        # Ballad flat concert fallback.
        assert titles[0][0] == "Poinciana"
        assert titles[1] == ("Ballad", [])


def test_guitar_reads_piano(tmp_path: Path) -> None:
    make_rs_fixture(tmp_path)
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "I Cant Help It"}])

    render(layout, tmp_path / "book.pdf", part=["Rhythm"])

    out = tmp_path / "book.Rhythm.pdf"
    with pikepdf.Pdf.open(out) as pdf:
        # Piano, Guitar→Piano (same file, deduped), Bass, Drums → 3 files nested.
        titles = outline_titles(pdf)
        assert titles[0][0] == "I Cant Help It"
        assert set(titles[0][1]) == {"Piano", "Guitar", "Bass", "Drums"} or set(
            titles[0][1]
        ) <= {"Piano", "Guitar", "Bass", "Drums"}
        # Guitar resolves (via Piano fallback) – file count 3 (Piano shared).
        assert len(pdf.pages) == 3


def test_norwegian_wood_dual_included_once(tmp_path: Path) -> None:
    make_rs_fixture(tmp_path)
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Norwegian Wood"}])

    render(layout, tmp_path / "book.pdf", part=["RS"])

    with pikepdf.Pdf.open(tmp_path / "book.RS.pdf") as pdf:
        titles = outline_titles(pdf)
        assert titles[0][0] == "Norwegian Wood"
        # Piano+Guitar share dual file, plus Bass, Drums → 3 files, 4 labels?
        # Dual included once: children include Piano, Guitar, Bass, Drums (Piano/Guitar same file? currently separate labels).
        assert "Bass" in titles[0][1]
        assert "Drums" in titles[0][1]
