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
    create_pdf(sm / "Mo Better Blues" / "Mo Better Blues-Trumpet.pdf", 1)
    create_pdf(sm / "Mo Better Blues" / "Mo Better Blues-Tenor Sax.pdf", 1)
    # Concert-only title stays flat
    create_pdf(sm / "Ballad" / "Ballad-Concert.pdf", 2)
    # Norwegian Wood with Piano:Guitar dual
    create_pdf(sm / "Norwegian Wood" / "Norwegian Wood-PianoGuitar.pdf", 1)
    create_pdf(sm / "Norwegian Wood" / "Norwegian Wood-Bass.pdf", 1)
    create_pdf(sm / "Norwegian Wood" / "Norwegian Wood-Drums.pdf", 1)
    (root / "tunery.yaml").write_text("- library: .\n")


def render_rs(root: Path, title: str) -> list[tuple[str, list[str]]]:
    layout = write_layout(root / "gig.yaml", [{"title": title}])
    render(layout, root / "book.pdf", part=["RS"])
    with pikepdf.Pdf.open(root / "book.RS.pdf") as pdf:
        return outline_titles(pdf)


def test_rs_concert_only_title_stays_flat(tmp_path: Path) -> None:
    make_rs_fixture(tmp_path)

    assert render_rs(tmp_path, "Ballad") == [("Ballad", [])]


def test_rs_children_named_by_chart_used(tmp_path: Path) -> None:
    make_rs_fixture(tmp_path)

    assert render_rs(tmp_path, "Mo Better Blues") == [
        ("Mo Better Blues", ["Rhythm", "Bass", "Drums"])
    ]
    assert render_rs(tmp_path, "Poinciana") == [("Poinciana", ["Concert", "Bass"])]


def test_rs_dual_score_included_once(tmp_path: Path) -> None:
    make_rs_fixture(tmp_path)

    assert render_rs(tmp_path, "Norwegian Wood") == [
        ("Norwegian Wood", ["Piano/Guitar", "Bass", "Drums"])
    ]


def test_guitar_reads_piano(tmp_path: Path) -> None:
    make_rs_fixture(tmp_path)

    assert render_rs(tmp_path, "I Cant Help It") == [
        ("I Cant Help It", ["Piano", "Bass", "Drums"])
    ]


def test_rhythm_expands_inside_comma_list(tmp_path: Path) -> None:
    make_rs_fixture(tmp_path)
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Mo Better Blues"}])

    render(layout, tmp_path / "book.pdf", part=["Trumpet,Tenor Sax,Rhythm"])

    with pikepdf.Pdf.open(tmp_path / "book.Trumpet-Tenor Sax-Rhythm.pdf") as pdf:
        assert outline_titles(pdf) == [
            ("Mo Better Blues", ["Trumpet", "Tenor Sax", "Rhythm", "Bass", "Drums"])
        ]


def test_explicit_bookmark_hierarchy_prefixes_parts(tmp_path: Path) -> None:
    make_rs_fixture(tmp_path)
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Mo Better Blues"}, {"title": "Ballad"}])

    render(layout, tmp_path / "book.pdf", part=["RS"], explicit_bookmark_hierarchy=True)

    with pikepdf.Pdf.open(tmp_path / "book.RS.pdf") as pdf:
        assert outline_titles(pdf) == [
            ("Mo Better Blues", ["_ Rhythm", "_ Bass", "_ Drums"]),
            ("Ballad", []),
        ]
