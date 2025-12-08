"""Tests for the tunery.index module."""

import json
from pathlib import Path

import pytest

from tunery.index import ChartLocation, Index


def create_index_json(path: Path, entries: list[dict]) -> Path:
    """Helper to create a JSON index file (array of entries)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))
    return path


def create_main_index_json(path: Path, book_entries: list[dict]) -> Path:
    """Helper to create the main index.json file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(book_entries))
    return path


def create_pdf_file(base_dir: Path, source_path: str) -> Path:
    """Helper to create a PDF file (as an empty file)."""
    pdf_path = (base_dir / source_path).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.touch()
    return pdf_path


def test_build_creates_sqlite_file(tmp_path: Path) -> None:
    """Test that Index.build creates a SQLite database file."""
    # Create index file
    create_index_json(
        tmp_path / "book1.json",
        entries=[
            {"title": "Blue In Green", "page": 51},
            {"title": "Autumn Leaves", "page": 39},
        ],
    )
    
    # Create PDF file
    create_pdf_file(tmp_path, "Books/FakeBook.pdf")
    
    # Create main index.json
    create_main_index_json(
        tmp_path / "index.json",
        book_entries=[
            {
                "source": "Books/FakeBook.pdf",
                "index": "book1.json",
            }
        ],
    )

    output_path = tmp_path / "cache" / "index.sqlite"
    Index.build(tmp_path / "index.json", output_path).close()

    assert output_path.exists()


def test_build_indexes_all_entries(tmp_path: Path) -> None:
    """Test that all entries from JSON files are indexed."""
    # Create index files
    create_index_json(
        tmp_path / "book1.json",
        entries=[
            {"title": "Song A", "page": 10},
            {"title": "Song B", "page": 20, "pages": 2},
        ],
    )
    create_index_json(
        tmp_path / "book2.json",
        entries=[
            {"title": "Song C", "page": 5},
        ],
    )
    
    # Create PDF files
    create_pdf_file(tmp_path, "Books/Book1.pdf")
    create_pdf_file(tmp_path, "Books/Book2.pdf")
    
    # Create main index.json
    create_main_index_json(
        tmp_path / "index.json",
        book_entries=[
            {"source": "Books/Book1.pdf", "index": "book1.json"},
            {"source": "Books/Book2.pdf", "index": "book2.json"},
        ],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(tmp_path / "index.json", output_path) as index:
        # Verify all three songs are indexed
        assert index.lookup("Song A") is not None
        assert index.lookup("Song B") is not None
        assert index.lookup("Song C") is not None


def test_build_resolves_source_paths(tmp_path: Path) -> None:
    """Test that source paths are resolved relative to the index.json file."""
    # Create index file
    create_index_json(
        tmp_path / "book.json",
        entries=[{"title": "Test Song", "page": 1}],
    )
    
    # Create PDF file
    create_pdf_file(tmp_path, "Books/MyBook.pdf")
    
    # Create main index.json
    create_main_index_json(
        tmp_path / "index.json",
        book_entries=[
            {"source": "Books/MyBook.pdf", "index": "book.json"},
        ],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(tmp_path / "index.json", output_path) as index:
        result = index.lookup("Test Song")
        assert result is not None
        # Path should be resolved to absolute
        assert Path(result.source_path).is_absolute()
        assert "MyBook.pdf" in result.source_path


def test_build_stores_page_count(tmp_path: Path) -> None:
    """Test that multi-page entries have correct length."""
    # Create index file
    create_index_json(
        tmp_path / "book.json",
        entries=[
            {"title": "Single Page", "page": 10},
            {"title": "Multi Page", "page": 20, "pages": 3},
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
        single = index.lookup("Single Page")
        assert single is not None
        assert single.length == 1

        multi = index.lookup("Multi Page")
        assert multi is not None
        assert multi.length == 3


def test_build_skips_invalid_json(tmp_path: Path, capsys) -> None:
    """Test that invalid index files are skipped with a warning."""
    # Create valid index file
    create_index_json(
        tmp_path / "valid.json",
        entries=[{"title": "Good Song", "page": 1}],
    )
    
    # Create invalid index file (not an array)
    (tmp_path / "invalid.json").write_text('{"bad": "data"}')
    
    # Create PDF file
    create_pdf_file(tmp_path, "Book.pdf")
    
    # Create main index.json with both valid and invalid entries
    create_main_index_json(
        tmp_path / "index.json",
        book_entries=[
            {"source": "Book.pdf", "index": "valid.json"},
            {"source": "Book.pdf", "index": "invalid.json"},
        ],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(tmp_path / "index.json", output_path) as index:
        # Valid entry should still be indexed
        assert index.lookup("Good Song") is not None

    # Warning should be printed
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "Index file must be an array" in captured.out


def test_build_processes_multiple_books(tmp_path: Path) -> None:
    """Test that Index.build processes multiple books."""
    # Create index files
    create_index_json(
        tmp_path / "root.json",
        entries=[{"title": "Root Song", "page": 1}],
    )
    create_index_json(
        tmp_path / "sub1.json",
        entries=[{"title": "Sub1 Song", "page": 1}],
    )
    create_index_json(
        tmp_path / "deep.json",
        entries=[{"title": "Deep Song", "page": 1}],
    )
    
    # Create PDF files
    create_pdf_file(tmp_path, "Root.pdf")
    create_pdf_file(tmp_path, "Sub1.pdf")
    create_pdf_file(tmp_path, "Deep.pdf")
    
    # Create main index.json
    create_main_index_json(
        tmp_path / "index.json",
        book_entries=[
            {"source": "Root.pdf", "index": "root.json"},
            {"source": "Sub1.pdf", "index": "sub1.json"},
            {"source": "Deep.pdf", "index": "deep.json"},
        ],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(tmp_path / "index.json", output_path) as index:
        assert index.lookup("Root Song") is not None
        assert index.lookup("Sub1 Song") is not None
        assert index.lookup("Deep Song") is not None


def test_lookup_exact_match(tmp_path: Path) -> None:
    """Test exact title lookup."""
    # Create index file
    create_index_json(
        tmp_path / "book.json",
        entries=[
            {"title": "All Of Me", "page": 20},
            {"title": "All Of You", "page": 21},
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
        result = index.lookup("All Of Me")
        assert result is not None
        assert result.page == 20

        # Partial match should not work with exact lookup
        assert index.lookup("All Of") is None


def test_lookup_returns_none_for_missing(tmp_path: Path) -> None:
    """Test that lookup returns None for non-existent titles."""
    # Create index file
    create_index_json(
        tmp_path / "book.json",
        entries=[{"title": "Existing Song", "page": 1}],
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
        assert index.lookup("Non Existent") is None


def test_init_raises_for_missing_index(tmp_path: Path) -> None:
    """Test that Index raises FileNotFoundError if index doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Index(tmp_path / "nonexistent.sqlite")


def test_lookup_fuzzy_partial_match(tmp_path: Path) -> None:
    """Test fuzzy lookup with partial title match."""
    # Create index file
    create_index_json(
        tmp_path / "book.json",
        entries=[
            {"title": "Blue In Green", "page": 51},
            {"title": "Blue Monk", "page": 52},
            {"title": "Blue Train", "page": 54},
            {"title": "Autumn Leaves", "page": 39},
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
        # Search for "Blue" should return all three Blue songs
        results = index.lookup_fuzzy("Blue")
        assert len(results) == 3
        pages_found = {r.page for r in results}
        assert pages_found == {51, 52, 54}


def test_lookup_fuzzy_returns_empty_for_no_match(tmp_path: Path) -> None:
    """Test that fuzzy lookup returns empty list for no matches."""
    # Create index file
    create_index_json(
        tmp_path / "book.json",
        entries=[{"title": "Something Else", "page": 1}],
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
        results = index.lookup_fuzzy("Nonexistent")
        assert results == []


def test_context_manager(tmp_path: Path) -> None:
    """Test that Index works as a context manager."""
    # Create index file
    create_index_json(
        tmp_path / "book.json",
        entries=[{"title": "Test Song", "page": 1}],
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
    Index.build(tmp_path / "index.json", output_path).close()

    with Index(output_path) as index:
        result = index.lookup("Test Song")
        assert result is not None

    # Connection should be closed after exiting context
    assert index._conn is None


def test_chart_location_dataclass() -> None:
    """Test ChartLocation is a proper dataclass."""
    loc = ChartLocation(source_path="/path/to/book.pdf", page=42, length=2)
    assert loc.source_path == "/path/to/book.pdf"
    assert loc.page == 42
    assert loc.length == 2

