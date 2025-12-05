"""Index module for building and querying the chart index SQLite database."""

import json
import sqlite3
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, PositiveInt


class IndexEntry(BaseModel):
    """A single entry in a JSON index file."""

    title: str
    page: PositiveInt
    pages: PositiveInt = 1  # Number of pages, defaults to 1


class IndexFile(BaseModel):
    """Schema for a JSON index file."""

    source: str  # PDF path relative to the JSON file
    index: list[IndexEntry]


class BookEntry(BaseModel):
    """Schema for an entry in the main index.json file."""

    source: str  # PDF path relative to the index.json file
    index: str  # Path to the index JSON file relative to index.json
    title: str | None = None
    edition: int | None = None
    volume: int | None = None
    shift: int | None = None  # Page offset to apply


class ChartLocation(NamedTuple):
    """Result of a chart lookup."""

    source_path: str
    page: int
    length: int


class Index:
    """SQLite-based chart index for looking up charts by title."""

    def __init__(self, path: Path):
        """
        Open an existing index.

        Args:
            path: Path to the SQLite index file.

        Raises:
            FileNotFoundError: If the index file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Index not found: {path}")
        self.path = path
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.path)
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def lookup(self, title: str) -> ChartLocation | None:
        """
        Look up a chart by exact title, returning the highest priority match.
        Matching is case-insensitive.

        Args:
            title: The title to search for (exact match, case-insensitive).

        Returns:
            ChartLocation if found (highest priority), None otherwise.
        """
        cursor = self._get_connection().cursor()
        cursor.execute(
            "SELECT source_path, page, length FROM charts WHERE LOWER(title) = ? ORDER BY priority DESC LIMIT 1",
            (title.lower(),),
        )
        row = cursor.fetchone()

        if row:
            return ChartLocation(source_path=row[0], page=row[1], length=row[2])
        return None

    def lookup_all(self, title: str) -> list[ChartLocation]:
        """
        Look up all instances of a chart by exact title.
        Matching is case-insensitive.

        Args:
            title: The title to search for (exact match, case-insensitive).

        Returns:
            List of ChartLocation objects, ordered by priority (highest first).
        """
        cursor = self._get_connection().cursor()
        cursor.execute(
            "SELECT source_path, page, length FROM charts WHERE LOWER(title) = ? ORDER BY priority DESC",
            (title.lower(),),
        )
        rows = cursor.fetchall()
        return [ChartLocation(source_path=r[0], page=r[1], length=r[2]) for r in rows]

    def lookup_fuzzy(self, title: str) -> list[ChartLocation]:
        """
        Look up charts by partial title match.
        Matching is case-insensitive.

        Args:
            title: The title to search for (case-insensitive contains).

        Returns:
            List of matching ChartLocation objects.
        """
        cursor = self._get_connection().cursor()
        cursor.execute(
            "SELECT source_path, page, length FROM charts WHERE LOWER(title) LIKE ?",
            (f"%{title.lower()}%",),
        )
        rows = cursor.fetchall()
        return [ChartLocation(source_path=r[0], page=r[1], length=r[2]) for r in rows]

    @classmethod
    def build(cls, index_json_path: Path, output_path: Path) -> "Index":
        """
        Build a SQLite index from the main index.json file.

        Args:
            index_json_path: Path to the main index.json file.
            output_path: Path to the SQLite database file to create.

        Returns:
            An Index instance for the newly created index.
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing database if present
        if output_path.exists():
            output_path.unlink()

        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()

        # Create the charts table
        cursor.execute("""
            CREATE TABLE charts (
                title TEXT NOT NULL,
                source_path TEXT NOT NULL,
                page INTEGER NOT NULL,
                length INTEGER NOT NULL DEFAULT 1,
                priority INTEGER NOT NULL DEFAULT 0
            )
        """)

        # Create indexes for fast title lookups
        cursor.execute("CREATE INDEX idx_title ON charts (title)")
        cursor.execute("CREATE INDEX idx_title_priority ON charts (title, priority DESC)")

        # Read the main index.json file
        index_json_path = Path(index_json_path)
        if not index_json_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_json_path}")

        with open(index_json_path, "r", encoding="utf-8") as f:
            main_index_data = json.load(f)

        # Validate and process book entries
        book_entries = [BookEntry.model_validate(entry) for entry in main_index_data]
        index_dir = index_json_path.parent

        # Track duplicates
        title_counts: dict[str, int] = {}
        duplicate_titles: set[str] = set()

        total_charts = 0
        processed_books = 0

        # Process each book entry in order (later entries get higher priority)
        for priority, book_entry in enumerate(book_entries):
            try:
                # Resolve the index file path
                index_file_path = (index_dir / book_entry.index).resolve()
                if not index_file_path.exists():
                    print(f"Warning: Index file not found: {index_file_path} (referenced in {index_json_path})")
                    continue

                # Read and validate the index file
                with open(index_file_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)

                # Index files are now just arrays of entries
                if not isinstance(index_data, list):
                    raise ValueError(f"Index file must be an array, got {type(index_data).__name__}")

                # Validate all entries
                entries = [IndexEntry.model_validate(entry) for entry in index_data]

                # Resolve the source PDF path
                source_path = (index_dir / book_entry.source).resolve()
                
                # Check that the source file exists
                if not source_path.exists():
                    print(f"Warning: Source file not found: {source_path} (referenced in {index_json_path})")
                    continue

                # Calculate page shift if specified
                page_shift = book_entry.shift or 0

                # Insert all entries from this book
                for entry in entries:
                    # Apply page shift if specified
                    page = entry.page + page_shift
                    
                    # Store title in lowercase for case-insensitive matching
                    title_lower = entry.title.lower()
                    
                    # Track title occurrences (using lowercase for duplicate detection)
                    title_counts[title_lower] = title_counts.get(title_lower, 0) + 1
                    if title_counts[title_lower] > 1:
                        duplicate_titles.add(title_lower)

                    cursor.execute(
                        "INSERT INTO charts (title, source_path, page, length, priority) VALUES (?, ?, ?, ?, ?)",
                        (title_lower, str(source_path), page, entry.pages, priority),
                    )
                    total_charts += 1

                processed_books += 1

            except Exception as e:
                print(f"Warning: Skipping book entry {book_entry.source}: {e}")
                continue

        conn.commit()
        conn.close()

        # Print summary
        print(
            f"Indexed {total_charts} charts from {processed_books} books into {output_path}"
        )

        # List duplicates
        if duplicate_titles:
            print(f"\nFound {len(duplicate_titles)} titles with duplicates:")
            for title in sorted(duplicate_titles):
                count = title_counts[title]
                print(f"  - {title} ({count} instances)")

        return cls(output_path)

