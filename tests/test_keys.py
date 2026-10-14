from pathlib import Path

import pikepdf
import pytest
import yaml

from tunery.keys import (
    expand_keys,
    is_concert,
    keyed_filename,
    normalize_key_name,
    output_path_for_key,
)
from tunery.render import render


def create_pdf(path: Path, page_count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pikepdf.Pdf.new()
    for _ in range(page_count):
        pdf.add_blank_page(page_size=(100, 100))
    pdf.save(path)
    pdf.close()
    return path


def write_layout(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(list(records), sort_keys=False))
    return path


def test_expand_keys_defaults_to_concert() -> None:
    assert expand_keys([]) == ["C"]
    assert expand_keys([""]) == ["C"]
    assert expand_keys([", ,"]) == ["C"]


def test_expand_keys_canonicalizes() -> None:
    assert expand_keys(["bb"]) == ["Bb"]
    assert expand_keys(["Eb"]) == ["Eb"]
    assert expand_keys(["F-sharp"]) == ["F#"]
    assert expand_keys(["gb"]) == ["Gb"]
    assert expand_keys(["B♭"]) == ["Bb"]


def test_expand_keys_accepts_concert_alias() -> None:
    assert expand_keys(["Concert"]) == ["C"]
    assert expand_keys(["concert"]) == ["C"]
    assert normalize_key_name("Concert") == "C"


def test_expand_keys_splits_commas_and_dedupes() -> None:
    assert expand_keys(["Bb,Eb", "F"]) == ["Bb", "Eb", "F"]
    assert expand_keys(["C,C", "Bb,Bb"]) == ["C", "Bb"]
    assert expand_keys(["c", "Concert"]) == ["C"]


def test_expand_keys_rejects_invalid_key() -> None:
    with pytest.raises(ValueError, match="Invalid key"):
        expand_keys(["H"])


def test_keyed_filename() -> None:
    assert keyed_filename("Song.pdf", "Bb") == "Song.Bb.pdf"
    assert keyed_filename("Song.pdf", "C") == "Song.Concert.pdf"
    assert keyed_filename("dir/Song.pdf", "Eb") == "dir/Song.Eb.pdf"
    assert keyed_filename("Song.txt", "Bb") is None


def test_output_path_for_key() -> None:
    assert output_path_for_key(Path("book.pdf"), "C") == Path("book.pdf")
    assert output_path_for_key(Path("book.pdf"), "Bb") == Path("book.Bb.pdf")
    assert output_path_for_key(Path("dir/book.pdf"), "Eb") == Path(
        "dir/book.Eb.pdf"
    )
    assert is_concert("C") and not is_concert("Bb")


def test_render_uses_key_edition_for_title_lookup(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    create_pdf(library_dir / "Song.pdf", 2)
    create_pdf(library_dir / "Song.Bb.pdf", 3)
    layout_path = write_layout(
        tmp_path / "setlist" / "gig.yaml",
        [
            {"config": [{"library": "../library", "match": "exact"}]},
            {"title": "Song"},
        ],
    )

    concert_output = tmp_path / "concert.pdf"
    render(layout_path, concert_output)
    with pikepdf.Pdf.open(concert_output) as pdf:
        assert len(pdf.pages) == 2

    render(layout_path, tmp_path / "setbook.pdf", ["Bb"])
    bb_output = tmp_path / "setbook.Bb.pdf"
    with pikepdf.Pdf.open(bb_output) as pdf:
        assert len(pdf.pages) == 3
        with pdf.open_outline() as outline:
            assert [item.title for item in outline.root] == ["Song"]


def test_render_prefers_concert_suffix(tmp_path: Path) -> None:
    library_dir = tmp_path / "library"
    create_pdf(library_dir / "Song.pdf", 2)
    create_pdf(library_dir / "Song.Concert.pdf", 4)
    layout_path = write_layout(
        tmp_path / "setlist" / "gig.yaml",
        [
            {"config": [{"library": "../library", "match": "exact"}]},
            {"title": "Song"},
        ],
    )
    output_path = tmp_path / "setbook.pdf"

    render(layout_path, output_path)

    with pikepdf.Pdf.open(output_path) as pdf:
        assert len(pdf.pages) == 4


def test_render_fans_out_one_setbook_per_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    library_dir = tmp_path / "library"
    create_pdf(library_dir / "Song.pdf", 2)
    create_pdf(library_dir / "Song.Bb.pdf", 3)
    layout_path = write_layout(
        tmp_path / "setlist" / "gig.yaml",
        [
            {"config": [{"library": "../library", "match": "exact"}]},
            {"title": "Song"},
        ],
    )
    output_path = tmp_path / "gig.pdf"

    render(layout_path, output_path, ["C", "Bb"])

    with pikepdf.Pdf.open(tmp_path / "gig.pdf") as pdf:
        assert len(pdf.pages) == 2
    with pikepdf.Pdf.open(tmp_path / "gig.Bb.pdf") as pdf:
        assert len(pdf.pages) == 3
    out = capsys.readouterr().out
    assert '[C] found    "Song"' in out
    assert '[Bb] found    "Song"' in out


def test_render_missing_key_edition_is_not_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    library_dir = tmp_path / "library"
    create_pdf(library_dir / "Song.pdf", 2)
    layout_path = write_layout(
        tmp_path / "setlist" / "gig.yaml",
        [
            {"config": [{"library": "../library", "match": "exact"}]},
            {"title": "Song"},
        ],
    )

    render(layout_path, tmp_path / "gig.pdf", ["C", "Eb"])

    assert (tmp_path / "gig.pdf").exists()
    assert not (tmp_path / "gig.Eb.pdf").exists()
    out = capsys.readouterr().out
    assert '[Eb] not found "Song"' in out


def test_render_single_non_concert_key_derives_suffixed_output(
    tmp_path: Path,
) -> None:
    library_dir = tmp_path / "library"
    create_pdf(library_dir / "Song.pdf", 2)
    create_pdf(library_dir / "Song.Bb.pdf", 3)
    layout_path = write_layout(
        tmp_path / "setlist" / "gig.yaml",
        [
            {"config": [{"library": "../library", "match": "exact"}]},
            {"title": "Song"},
        ],
    )

    render(layout_path, tmp_path / "Combo3.pdf", ["Bb"])

    assert not (tmp_path / "Combo3.pdf").exists()
    bb_output = tmp_path / "Combo3.Bb.pdf"
    assert bb_output.exists()
    with pikepdf.Pdf.open(bb_output) as pdf:
        assert len(pdf.pages) == 3


def test_render_file_entry_without_pages_uses_key_edition(
    tmp_path: Path,
) -> None:
    layout_dir = tmp_path / "setlist"
    create_pdf(layout_dir / "Chart.pdf", 1)
    create_pdf(layout_dir / "Chart.Bb.pdf", 5)
    layout_path = write_layout(
        layout_dir / "gig.yaml", [{"file": "Chart.pdf"}]
    )

    render(layout_path, tmp_path / "setbook.pdf", ["Bb"])

    with pikepdf.Pdf.open(tmp_path / "setbook.Bb.pdf") as pdf:
        assert len(pdf.pages) == 5
        with pdf.open_outline() as outline:
            assert [item.title for item in outline.root] == ["Chart"]


def test_render_file_entry_with_pages_skips_key_edition(
    tmp_path: Path,
) -> None:
    layout_dir = tmp_path / "setlist"
    create_pdf(layout_dir / "Chart.pdf", 4)
    create_pdf(layout_dir / "Chart.Bb.pdf", 5)
    layout_path = write_layout(
        layout_dir / "gig.yaml",
        [{"file": "Chart.pdf", "page": 2, "length": 2}],
    )

    render(layout_path, tmp_path / "setbook.pdf", ["Bb"])

    with pikepdf.Pdf.open(tmp_path / "setbook.Bb.pdf") as pdf:
        assert len(pdf.pages) == 2
