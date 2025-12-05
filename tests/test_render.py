from pathlib import Path
from typing import Sequence

import pikepdf
import pytest
import yaml

from tunery.render import render, copy_pages, get_page_label_to_index_map


def create_pdf(path: Path, page_count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pikepdf.Pdf.new()
    for _ in range(page_count):
        pdf.add_blank_page(page_size=(100, 100))
    pdf.save(path)
    pdf.close()
    return path


def write_layout(path: Path, records: Sequence[dict]) -> Path:
    path.write_text(yaml.safe_dump(list(records), sort_keys=False))
    return path


def outline_page_index(pdf: pikepdf.Pdf, outline_item: pikepdf.OutlineItem) -> int:
    destination = outline_item.destination
    if destination is None:
        msg = f"Outline item '{outline_item.title}' is missing a destination"
        raise AssertionError(msg)
    return pdf.pages.index(destination[0])  # pyright: ignore


def test_bind_pdf_combines_sections_and_flat_entries(tmp_path: Path) -> None:
    layout_dir = tmp_path / "setlist"
    layout_dir.mkdir()

    schedule_pdf = create_pdf(layout_dir / "Schedule.pdf", 2)
    songbook_pdf = create_pdf(layout_dir / "Songbook.pdf", 6)
    groove_pdf = create_pdf(layout_dir / "Groove Standard.pdf", 1)
    ballad_pdf = create_pdf(layout_dir / "Ballad.pdf", 1)

    dropbox_dir = tmp_path / "absolute"
    feature_pdf = create_pdf(dropbox_dir / "Feature Piece.pdf", 1)
    duet_pdf = create_pdf(dropbox_dir / "Duet.pdf", 1)

    records = [
        {"file": str(schedule_pdf), "title": "Timeline"},
        {
            "section": "Set 1",
            "body": [
                {
                    "file": "Songbook.pdf",
                    "page": 1,
                    "title": "Opening Groove",
                },
                {
                    "file": "Songbook.pdf",
                    "page": 2,
                    "length": 2,
                    "title": "Latin Medley",
                },
                {
                    "file": "Groove Standard.pdf",
                    "title": "Groove Standard",
                },
            ],
        },
        {
            "section": "Set 1a (Feature Spotlight)",
            "body": [
                {
                    "file": str(feature_pdf),
                    "title": "Feature Piece",
                },
                {
                    "file": str(duet_pdf),
                    "title": "Duet",
                },
            ],
        },
        {"file": "Ballad.pdf", "title": "Ballad"},
    ]

    layout_path = write_layout(layout_dir / "25-11-09.yaml", records)
    output_path = layout_dir / "combined.pdf"

    render(layout_path, output_path)

    with pikepdf.Pdf.open(output_path) as merged:
        assert len(merged.pages) == 9

        with merged.open_outline() as outline:
            root_items = list(outline.root)
            assert [item.title for item in root_items] == [
                "Timeline",
                "Set 1",
                "Set 1a (Feature Spotlight)",
                "Ballad",
            ]

            set_one = root_items[1]
            set_one_children = list(set_one.children)
            assert [child.title for child in set_one_children] == [
                "Opening Groove",
                "Latin Medley",
                "Groove Standard",
            ]
            assert outline_page_index(merged, set_one) == 2
            assert [outline_page_index(merged, child) for child in set_one_children] == [
                2,
                3,
                5,
            ]

            set_one_a = root_items[2]
            set_one_a_children = list(set_one_a.children)
            assert [child.title for child in set_one_a_children] == [
                "Feature Piece",
                "Duet",
            ]
            assert outline_page_index(merged, set_one_a) == 6
            assert [outline_page_index(merged, child) for child in set_one_a_children] == [
                6,
                7,
            ]

            ballad_item = root_items[3]
            assert outline_page_index(merged, ballad_item) == 8


def test_bind_pdf_handles_empty_section(tmp_path: Path) -> None:
    layout_dir = tmp_path / "empty"
    layout_dir.mkdir()

    solo_pdf = create_pdf(layout_dir / "Solo Tune.pdf", 1)

    records = [
        {"section": "Empty Section", "body": []},
        {"file": "Solo Tune.pdf", "title": "Solo Tune"},
    ]

    layout_path = write_layout(layout_dir / "empty.yaml", records)
    output_path = layout_dir / "empty.pdf"

    render(layout_path, output_path)

    with pikepdf.Pdf.open(output_path) as merged:
        assert len(merged.pages) == 1

        with merged.open_outline() as outline:
            root_items = list(outline.root)
            assert root_items[0].title == "Empty Section"
            assert outline_page_index(merged, root_items[0]) == 0
            assert list(root_items[0].children) == []
            assert root_items[1].title == "Solo Tune"
            assert outline_page_index(merged, root_items[1]) == 0


def test_copy_pages_validates_ranges(tmp_path: Path) -> None:
    source_pdf = create_pdf(tmp_path / "source.pdf", 2)
    combined_pdf = pikepdf.Pdf.new()

    with pytest.raises(ValueError):
        copy_pages(combined_pdf, str(source_pdf), start_page=3)

    with pytest.raises(ValueError):
        copy_pages(combined_pdf, str(source_pdf), start_page=2, length=3)


def test_get_page_label_to_index_map_handles_missing_labels(tmp_path: Path) -> None:
    """Test that page label mapping works for PDFs without page labels."""
    pdf_path = create_pdf(tmp_path / "test.pdf", 5)
    
    with pikepdf.Pdf.open(pdf_path) as pdf:
        label_map = get_page_label_to_index_map(pdf)
        
        # Should create 1:1 mapping (page 1 = index 0, page 2 = index 1, etc.)
        assert label_map == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def test_get_page_label_to_index_map_handles_custom_labels(tmp_path: Path) -> None:
    """Test that page label mapping works for PDFs with custom page labels."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a PDF with custom page labels (e.g., starting at page 10)
    pdf = pikepdf.Pdf.new()
    for _ in range(5):
        pdf.add_blank_page(page_size=(100, 100))
    
    # Add page labels starting from 10
    page_labels = pikepdf.Dictionary(
        Nums=pikepdf.Array([
            0,  # Start at physical page 0
            pikepdf.Dictionary(St=10),  # Start numbering at 10
        ])
    )
    pdf.Root.PageLabels = page_labels
    pdf.save(pdf_path)
    pdf.close()
    
    with pikepdf.Pdf.open(pdf_path) as pdf:
        label_map = get_page_label_to_index_map(pdf)
        
        # Should map page label 10 to index 0, 11 to index 1, etc.
        assert label_map[10] == 0
        assert label_map[11] == 1
        assert label_map[12] == 2
        assert label_map[13] == 3
        assert label_map[14] == 4


def test_get_page_label_to_index_map_handles_malformed_labels(tmp_path: Path) -> None:
    """Test that page label mapping gracefully handles malformed page label structures."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a PDF with malformed page labels
    pdf = pikepdf.Pdf.new()
    for _ in range(3):
        pdf.add_blank_page(page_size=(100, 100))
    
    # Add malformed page labels (e.g., /S is a name instead of dict)
    page_labels = pikepdf.Dictionary(
        Nums=pikepdf.Array([
            0,
            pikepdf.Dictionary(S=pikepdf.Name.D),  # /S is a name, not a dict
        ])
    )
    pdf.Root.PageLabels = page_labels
    pdf.save(pdf_path)
    pdf.close()
    
    with pikepdf.Pdf.open(pdf_path) as pdf:
        label_map = get_page_label_to_index_map(pdf)
        
        # Should fall back to 1:1 mapping when labels are malformed
        assert label_map == {1: 0, 2: 1, 3: 2}


def test_get_page_label_to_index_map_handles_multiple_decimal_ranges(tmp_path: Path) -> None:
    """Test that page label mapping correctly handles multiple decimal ranges, preferring longer ranges."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a PDF similar to "The new real book vol 2.pdf" structure
    pdf = pikepdf.Pdf.new()
    for _ in range(20):
        pdf.add_blank_page(page_size=(100, 100))
    
    # Add page labels with multiple decimal ranges:
    # - Index 0: decimal starting at 1 (short range, 1 page)
    # - Index 1-3: alphabetic (should be skipped)
    # - Index 4-18: decimal starting at 1 (long range, main content)
    # - Index 19: decimal starting at 2 (short range, 1 page)
    page_labels = pikepdf.Dictionary(
        Nums=pikepdf.Array([
            0,
            pikepdf.Dictionary(S=pikepdf.Name.D),  # Decimal, defaults to start=1
            1,
            pikepdf.Dictionary(S=pikepdf.Name.a),  # Alphabetic, should be skipped
            4,
            pikepdf.Dictionary(S=pikepdf.Name.D),  # Decimal, defaults to start=1 (main range)
            19,
            pikepdf.Dictionary(S=pikepdf.Name.D, St=2),  # Decimal starting at 2
        ])
    )
    pdf.Root.PageLabels = page_labels
    pdf.save(pdf_path)
    pdf.close()
    
    with pikepdf.Pdf.open(pdf_path) as pdf:
        label_map = get_page_label_to_index_map(pdf)
        
        # Page 1 should map to index 4 (longer range), not index 0
        assert label_map[1] == 4
        # Page 2 should map to index 5 (longer range), not index 19
        assert label_map[2] == 5
        # Page 15 should map to index 18 (from the main range)
        assert label_map[15] == 18
