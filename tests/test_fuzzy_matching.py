"""Tests for fuzzy matching functionality."""

from pathlib import Path

import pytest

from tunery.index import ChartLocation, FuzzyMatch, Index


def create_index_json(path: Path, entries: list[dict]) -> Path:
    """Helper to create a JSON index file (array of entries)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.write_text(json.dumps(entries))
    return path


def create_main_index_json(path: Path, book_entries: list[dict]) -> Path:
    """Helper to create the main index.json file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.write_text(json.dumps(book_entries))
    return path


def create_pdf_file(base_dir: Path, source_path: str) -> Path:
    """Helper to create a PDF file (as an empty file)."""
    pdf_path = (base_dir / source_path).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.touch()
    return pdf_path


def test_girl_from_ipanema_matches_with_the(tmp_path: Path) -> None:
    """
    Test that 'Girl from Ipanema' matches 'The Girl from Ipanema'.
    
    This tests that fuzzy matching can handle missing articles (like "The")
    at the beginning of titles. The database stores titles in lowercase,
    so matched_title will be lowercase.
    """
    # Create index file with "The Girl from Ipanema"
    create_index_json(
        tmp_path / "book.json",
        entries=[
            {"title": "The Girl from Ipanema", "page": 100},
            {"title": "Blue In Green", "page": 51},
        ],
    )
    
    # Create PDF file
    create_pdf_file(tmp_path, "Book.pdf")
    
    # Create main index.json
    create_main_index_json(
        tmp_path / "index.json",
        book_entries=[
            {"source": "Book.pdf", "index": "book.json"},
        ],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(tmp_path / "index.json", output_path) as index:
        # Search for "Girl from Ipanema" (without "The")
        results = index.lookup_fuzzy_edit_distance("Girl from Ipanema", score_cutoff=80, limit=5)
        
        # Should find "The Girl from Ipanema"
        assert len(results) > 0, "Should find at least one match"
        
        # Check that we found the correct song
        # Note: The database stores titles in lowercase, so matched_title will be lowercase
        matched_titles = [r.matched_title for r in results]
        assert "the girl from ipanema" in matched_titles, f"Expected 'the girl from ipanema' in {matched_titles}"
        
        # The match should have a score above the threshold (90) since token_set_ratio handles missing articles
        best_match = results[0]
        assert best_match.score >= 90, f"Expected score >= 90, got {best_match.score}"
        assert best_match.matched_title == "the girl from ipanema"
        assert best_match.location.page == 100


def test_the_girl_from_ipanema_matches_without_the(tmp_path: Path) -> None:
    """
    Test that 'The Girl from Ipanema' matches 'Girl from Ipanema'.
    
    This tests fuzzy matching in the reverse direction - when the search
    includes "The" but the stored title doesn't.
    """
    # Create index file with "Girl from Ipanema" (without "The")
    create_index_json(
        tmp_path / "book.json",
        entries=[
            {"title": "Girl from Ipanema", "page": 100},
            {"title": "Blue In Green", "page": 51},
        ],
    )
    
    # Create PDF file
    create_pdf_file(tmp_path, "Book.pdf")
    
    # Create main index.json
    create_main_index_json(
        tmp_path / "index.json",
        book_entries=[
            {"source": "Book.pdf", "index": "book.json"},
        ],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(tmp_path / "index.json", output_path) as index:
        # Search for "The Girl from Ipanema" (with "The")
        results = index.lookup_fuzzy_edit_distance("The Girl from Ipanema", score_cutoff=80, limit=5)
        
        # Should find "Girl from Ipanema"
        assert len(results) > 0, "Should find at least one match"
        
        # Check that we found the correct song
        matched_titles = [r.matched_title for r in results]
        assert "girl from ipanema" in matched_titles, f"Expected 'girl from ipanema' in {matched_titles}"
        
        # The match should have a score above the threshold (90) since token_set_ratio handles extra articles
        best_match = results[0]
        assert best_match.score >= 90, f"Expected score >= 90, got {best_match.score}"
        assert best_match.matched_title == "girl from ipanema"
        assert best_match.location.page == 100
