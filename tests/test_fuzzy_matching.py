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


def test_feel_like_making_love_prefers_similar_title_over_substring(tmp_path: Path) -> None:
    """
    Test that 'Feel Like Making Love' matches 'Feel Like Makin' Love' over 'L-O-V-E'.
    
    This tests that fuzzy matching doesn't over-score short titles that happen
    to be substrings. 'L-O-V-E' normalizes to 'love' which is contained in
    'Feel Like Making Love', but 'Feel Like Makin' Love' is the correct match.
    """
    # Create index file with both titles
    create_index_json(
        tmp_path / "book.json",
        entries=[
            {"title": "L-O-V-E", "page": 50},
            {"title": "Feel Like Makin' Love", "page": 100},
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
        # Search for "Feel Like Making Love"
        results = index.lookup_fuzzy_edit_distance("Feel Like Making Love", score_cutoff=80, limit=5)
        
        # Should find "Feel Like Makin' Love" as the best match
        assert len(results) > 0, "Should find at least one match"
        
        best_match = results[0]
        # matched_title is lowercase (stored that way in the DB)
        assert best_match.matched_title == "feel like makin' love", \
            f"Expected 'feel like makin' love' as best match, got '{best_match.matched_title}'"
        assert best_match.location.page == 100
        
        # "L-O-V-E" should either not match at all (below cutoff) or score lower
        love_matches = [r for r in results if r.matched_title == "love"]
        if love_matches:
            assert love_matches[0].score < best_match.score, \
                f"'L-O-V-E' ({love_matches[0].score}) should score lower than 'Feel Like Makin Love' ({best_match.score})"


def test_lover_man_matches_with_parenthetical_suffix(tmp_path: Path) -> None:
    """
    Test that 'Lover Man' matches 'Lover Man (Oh, Where Can You Be?)'.
    
    This tests that fuzzy matching can handle titles with parenthetical suffixes
    by also matching against a stripped version of the title.
    """
    # Create index file with the full title including parenthetical
    create_index_json(
        tmp_path / "book.json",
        entries=[
            {"title": "Lover Man (Oh, Where Can You Be?)", "page": 100},
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
        # Search for "Lover Man" (without parenthetical suffix)
        results = index.lookup_fuzzy_edit_distance("Lover Man", score_cutoff=90, limit=5)
        
        # Should find "Lover Man (Oh, Where Can You Be?)"
        assert len(results) > 0, "Should find at least one match"
        
        best_match = results[0]
        # matched_title is lowercase (stored that way in the DB)
        assert best_match.matched_title == "lover man (oh, where can you be?)", \
            f"Expected 'lover man (oh, where can you be?)' as best match, got '{best_match.matched_title}'"
        assert best_match.location.page == 100
        assert best_match.score >= 90, f"Expected score >= 90, got {best_match.score}"
