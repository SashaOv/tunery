from pathlib import Path

import pikepdf
import pytest

from tunery import Composer as PackageComposer
from tunery.composer import Composer


def create_pdf(path: Path, page_count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pikepdf.Pdf.new()
    for _ in range(page_count):
        pdf.add_blank_page(page_size=(100, 100))
    pdf.save(path)
    pdf.close()
    return path


def outline_page_index(pdf: pikepdf.Pdf, outline_item: pikepdf.OutlineItem) -> int:
    destination = outline_item.destination
    if destination is None:
        msg = f"Outline item '{outline_item.title}' is missing a destination"
        raise AssertionError(msg)
    return pdf.pages.index(destination[0])  # pyright: ignore


def test_composer_adds_files_to_output_pdf(tmp_path: Path) -> None:
    assert PackageComposer is Composer

    source = create_pdf(tmp_path / "Source.pdf", 4)
    output = tmp_path / "setbook.pdf"

    composer = Composer(output)
    composer.add("First Chart", source, start=2, pages=2)
    composer.add("Full Chart", source)

    with pikepdf.Pdf.open(output) as pdf:
        assert len(pdf.pages) == 6

        with pdf.open_outline() as outline:
            root_items = list(outline.root)
            assert [item.title for item in root_items] == [
                "First Chart",
                "Full Chart",
            ]
            assert [outline_page_index(pdf, item) for item in root_items] == [0, 2]


def test_composer_adds_nested_section_bookmarks(tmp_path: Path) -> None:
    opener = create_pdf(tmp_path / "Opener.pdf", 1)
    feature = create_pdf(tmp_path / "Feature.pdf", 1)
    closer = create_pdf(tmp_path / "Closer.pdf", 1)
    output = tmp_path / "setbook.pdf"

    composer = Composer(output)
    composer.add("Opener", opener)
    composer.start_section("Set 1")
    composer.add("Feature", feature)
    composer.start_section("Finale")
    composer.add("Closer", closer)
    composer.end_section()
    composer.end_section()

    with pikepdf.Pdf.open(output) as pdf:
        assert len(pdf.pages) == 3

        with pdf.open_outline() as outline:
            root_items = list(outline.root)
            assert [item.title for item in root_items] == ["Opener", "Set 1"]
            assert [outline_page_index(pdf, item) for item in root_items] == [0, 1]

            set_children = list(root_items[1].children)
            assert [item.title for item in set_children] == ["Feature", "Finale"]
            assert [outline_page_index(pdf, item) for item in set_children] == [1, 2]

            finale_children = list(set_children[1].children)
            assert [item.title for item in finale_children] == ["Closer"]
            assert outline_page_index(pdf, finale_children[0]) == 2


def test_composer_rejects_unmatched_section_end(tmp_path: Path) -> None:
    composer = Composer(tmp_path / "setbook.pdf")

    with pytest.raises(ValueError, match="No open section"):
        composer.end_section()
