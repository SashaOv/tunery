from pathlib import Path

import pikepdf
import yaml

from tunery.render import output_path_for_label, render


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


def make_fixture(root: Path) -> None:
    v = root / "Vault"
    create_pdf(v / "TwoPart" / "TwoPart-Bass.pdf", 1)
    create_pdf(v / "TwoPart" / "TwoPart-Piano.pdf", 2)
    create_pdf(v / "ConcertOnly" / "ConcertOnly-Concert.pdf", 3)
    (root / "tunery.yaml").write_text("- library: Vault\n")


def outline_titles(pdf: pikepdf.Pdf):
    with pdf.open_outline() as outline:
        return [(item.title, [c.title for c in item.children]) for item in outline.root]


def test_two_part_options_produce_two_outputs(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "TwoPart"}])

    render(layout, tmp_path / "book.pdf", part=["Bass", "Piano"])

    assert (tmp_path / "book.Bass.pdf").exists()
    assert (tmp_path / "book.Piano.pdf").exists()
    with pikepdf.Pdf.open(tmp_path / "book.Bass.pdf") as pdf:
        assert len(pdf.pages) == 1
    with pikepdf.Pdf.open(tmp_path / "book.Piano.pdf") as pdf:
        assert len(pdf.pages) == 2


def test_combined_parts_nested_bookmarks(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "TwoPart"}])

    render(layout, tmp_path / "book.pdf", part=["Bass,Piano"])

    out = tmp_path / "book.Bass+Piano.pdf"
    assert out.exists()
    with pikepdf.Pdf.open(out) as pdf:
        assert len(pdf.pages) == 3
        titles = outline_titles(pdf)
        assert titles[0][0] == "TwoPart"
        assert titles[0][1] == ["Bass", "Piano"]


def test_combined_concert_only_stays_flat(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "ConcertOnly"}])

    render(layout, tmp_path / "book.pdf", part=["Bass,Piano"])

    out = tmp_path / "book.Bass+Piano.pdf"
    with pikepdf.Pdf.open(out) as pdf:
        assert len(pdf.pages) == 3
        titles = outline_titles(pdf)
        # One file (Concert fallback shared) → flat bookmark.
        assert titles == [("ConcertOnly", [])]


def test_missing_part_reported(tmp_path: Path, capsys) -> None:
    v = tmp_path / "Vault"
    create_pdf(v / "BbOnly" / "BbOnly-Bb.pdf", 1)
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "BbOnly"}])

    render(layout, tmp_path / "book.pdf", part=["Piano"])

    out = capsys.readouterr().out
    assert 'missing Piano for "BbOnly"' in out


def test_output_suffix_not_doubled(tmp_path: Path) -> None:
    assert output_path_for_label(tmp_path / "book.RS.pdf", "RS") == tmp_path / "book.RS.pdf"
    assert output_path_for_label(tmp_path / "spot.Piano.pdf", "Piano") == tmp_path / "spot.Piano.pdf"


def test_output_ending_in_part_alias_is_kept(tmp_path: Path) -> None:
    assert output_path_for_label(tmp_path / "x.rs.pdf", "Rhythm") == tmp_path / "x.rs.pdf"
    assert output_path_for_label(tmp_path / "x.RS.pdf", "Rhythm Section") == tmp_path / "x.RS.pdf"
    assert output_path_for_label(tmp_path / "x.tenor.pdf", "Tenor Sax") == tmp_path / "x.tenor.pdf"
    assert output_path_for_label(tmp_path / "x.pdf", "Rhythm") == tmp_path / "x.Rhythm.pdf"
    assert output_path_for_label(tmp_path / "x.bass.pdf", "Piano") == tmp_path / "x.bass.Piano.pdf"
