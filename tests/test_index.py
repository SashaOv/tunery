"""Tests for the tunery.index module."""

import json
from pathlib import Path

import pytest

from tunery.index import ChartLocation, Index


def create_index_json(path: Path, source: str, entries: list[dict]) -> Path:
    """Helper to create a JSON index file and the corresponding PDF file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"source": source, "index": entries}
    path.write_text(json.dumps(data))
    
    # Create the actual PDF file (as an empty file) so the index build doesn't fail
    pdf_path = (path.parent / source).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.touch()
    
    return path


def test_build_creates_sqlite_file(tmp_path: Path) -> None:
    """Test that Index.build creates a SQLite database file."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    create_index_json(
        index_dir / "book1.json",
        source="Books/FakeBook.pdf",
        entries=[
            {"title": "Blue In Green", "page": 51},
            {"title": "Autumn Leaves", "page": 39},
        ],
    )

    output_path = tmp_path / "cache" / "index.sqlite"
    Index.build(index_dir, output_path).close()

    assert output_path.exists()


def test_build_indexes_all_entries(tmp_path: Path) -> None:
    """Test that all entries from JSON files are indexed."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    create_index_json(
        index_dir / "book1.json",
        source="Books/Book1.pdf",
        entries=[
            {"title": "Song A", "page": 10},
            {"title": "Song B", "page": 20, "pages": 2},
        ],
    )
    create_index_json(
        index_dir / "book2.json",
        source="Books/Book2.pdf",
        entries=[
            {"title": "Song C", "page": 5},
        ],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(index_dir, output_path) as index:
        # Verify all three songs are indexed
        assert index.lookup("Song A") is not None
        assert index.lookup("Song B") is not None
        assert index.lookup("Song C") is not None


def test_build_resolves_source_paths(tmp_path: Path) -> None:
    """Test that source paths are resolved relative to the JSON file."""
    index_dir = tmp_path / "indexes" / "subdir"
    index_dir.mkdir(parents=True)

    create_index_json(
        index_dir / "book.json",
        source="../Books/MyBook.pdf",
        entries=[{"title": "Test Song", "page": 1}],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(tmp_path / "indexes", output_path) as index:
        result = index.lookup("Test Song")
        assert result is not None
        # Path should be resolved to absolute
        assert Path(result.source_path).is_absolute()
        assert "MyBook.pdf" in result.source_path


def test_build_stores_page_count(tmp_path: Path) -> None:
    """Test that multi-page entries have correct length."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    create_index_json(
        index_dir / "book.json",
        source="Book.pdf",
        entries=[
            {"title": "Single Page", "page": 10},
            {"title": "Multi Page", "page": 20, "pages": 3},
        ],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(index_dir, output_path) as index:
        single = index.lookup("Single Page")
        assert single is not None
        assert single.length == 1

        multi = index.lookup("Multi Page")
        assert multi is not None
        assert multi.length == 3


def test_build_skips_invalid_json(tmp_path: Path, capsys) -> None:
    """Test that invalid JSON files are skipped with a warning."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    # Valid file
    create_index_json(
        index_dir / "valid.json",
        source="Book.pdf",
        entries=[{"title": "Good Song", "page": 1}],
    )

    # Invalid file (missing required fields)
    (index_dir / "invalid.json").write_text('{"bad": "data"}')

    output_path = tmp_path / "index.sqlite"
    with Index.build(index_dir, output_path) as index:
        # Valid entry should still be indexed
        assert index.lookup("Good Song") is not None

    # Warning should be printed
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "invalid.json" in captured.out


def test_build_scans_recursively(tmp_path: Path) -> None:
    """Test that Index.build scans subdirectories."""
    index_dir = tmp_path / "indexes"
    (index_dir / "subdir1").mkdir(parents=True)
    (index_dir / "subdir2" / "nested").mkdir(parents=True)

    create_index_json(
        index_dir / "root.json",
        source="Root.pdf",
        entries=[{"title": "Root Song", "page": 1}],
    )
    create_index_json(
        index_dir / "subdir1" / "sub1.json",
        source="Sub1.pdf",
        entries=[{"title": "Sub1 Song", "page": 1}],
    )
    create_index_json(
        index_dir / "subdir2" / "nested" / "deep.json",
        source="Deep.pdf",
        entries=[{"title": "Deep Song", "page": 1}],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(index_dir, output_path) as index:
        assert index.lookup("Root Song") is not None
        assert index.lookup("Sub1 Song") is not None
        assert index.lookup("Deep Song") is not None


def test_lookup_exact_match(tmp_path: Path) -> None:
    """Test exact title lookup."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    create_index_json(
        index_dir / "book.json",
        source="Book.pdf",
        entries=[
            {"title": "All Of Me", "page": 20},
            {"title": "All Of You", "page": 21},
        ],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(index_dir, output_path) as index:
        result = index.lookup("All Of Me")
        assert result is not None
        assert result.page == 20

        # Partial match should not work with exact lookup
        assert index.lookup("All Of") is None


def test_lookup_returns_none_for_missing(tmp_path: Path) -> None:
    """Test that lookup returns None for non-existent titles."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    create_index_json(
        index_dir / "book.json",
        source="Book.pdf",
        entries=[{"title": "Existing Song", "page": 1}],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(index_dir, output_path) as index:
        assert index.lookup("Non Existent") is None


def test_init_raises_for_missing_index(tmp_path: Path) -> None:
    """Test that Index raises FileNotFoundError if index doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Index(tmp_path / "nonexistent.sqlite")


def test_lookup_fuzzy_partial_match(tmp_path: Path) -> None:
    """Test fuzzy lookup with partial title match."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    create_index_json(
        index_dir / "book.json",
        source="Book.pdf",
        entries=[
            {"title": "Blue In Green", "page": 51},
            {"title": "Blue Monk", "page": 52},
            {"title": "Blue Train", "page": 54},
            {"title": "Autumn Leaves", "page": 39},
        ],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(index_dir, output_path) as index:
        # Search for "Blue" should return all three Blue songs
        results = index.lookup_fuzzy("Blue")
        assert len(results) == 3
        pages_found = {r.page for r in results}
        assert pages_found == {51, 52, 54}


def test_lookup_fuzzy_returns_empty_for_no_match(tmp_path: Path) -> None:
    """Test that fuzzy lookup returns empty list for no matches."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    create_index_json(
        index_dir / "book.json",
        source="Book.pdf",
        entries=[{"title": "Something Else", "page": 1}],
    )

    output_path = tmp_path / "index.sqlite"
    with Index.build(index_dir, output_path) as index:
        results = index.lookup_fuzzy("Nonexistent")
        assert results == []


def test_context_manager(tmp_path: Path) -> None:
    """Test that Index works as a context manager."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()

    create_index_json(
        index_dir / "book.json",
        source="Book.pdf",
        entries=[{"title": "Test Song", "page": 1}],
    )

    output_path = tmp_path / "index.sqlite"
    Index.build(index_dir, output_path).close()

    with Index(output_path) as index:
        result = index.lookup("Test Song")
        assert result is not None

    # Connection should be closed after exiting context
    assert index._conn is None


def test_chart_location_namedtuple() -> None:
    """Test ChartLocation is a proper NamedTuple."""
    loc = ChartLocation(source_path="/path/to/book.pdf", page=42, length=2)
    assert loc.source_path == "/path/to/book.pdf"
    assert loc.page == 42
    assert loc.length == 2
    # Can unpack
    path, page, length = loc
    assert path == "/path/to/book.pdf"

