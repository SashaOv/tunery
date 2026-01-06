"""Index module for building and querying the chart index SQLite database."""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pydantic import BaseModel, PositiveInt
from rapidfuzz import fuzz, process


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


@dataclass
class ChartLocation:
    """Result of a chart lookup."""

    source_path: str
    page: int
    length: int


@dataclass
class FuzzyMatch:
    """Result of a fuzzy chart lookup with similarity score."""

    location: ChartLocation
    score: float
    matched_title: str


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

    def _normalize_title(self, title: str) -> str:
        """Normalize a title for fuzzy matching: lowercase, remove punctuation, normalize whitespace."""
        # Convert to lowercase
        normalized = title.lower()
        # Remove punctuation except spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        return normalized

    def _strip_parenthetical(self, title: str) -> str:
        """Strip parenthetical suffixes from a title, e.g., '(Oh, Where Can You Be?)'."""
        return re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()

    def lookup_fuzzy_edit_distance(
        self, title: str, score_cutoff: int = 90, limit: int = 5
    ) -> List[FuzzyMatch]:
        """
        Look up charts using fuzzy string matching with edit distance.
        Returns matches with similarity scores and matched titles.

        Args:
            title: The title to search for.
            score_cutoff: Minimum similarity score (0-100). Default 90.
            limit: Maximum number of results to return. Default 5.

        Returns:
            List of FuzzyMatch objects, sorted by score (highest first).
        """
        cursor = self._get_connection().cursor()
        # Get all titles from the database
        cursor.execute("SELECT DISTINCT title FROM charts")
        all_titles = [row[0] for row in cursor.fetchall()]

        if not all_titles:
            return []

        # Normalize the search title
        normalized_search = self._normalize_title(title)

        # Build mapping of normalized titles to original titles.
        # Include both full titles and stripped versions (without parenthetical
        # suffixes like "(Oh, Where Can You Be?)") so that "Lover Man" can match
        # "Lover Man (Oh, Where Can You Be?)".
        title_map: dict[str, str] = {}
        for t in all_titles:
            normalized = self._normalize_title(t)
            title_map[normalized] = t
            # Also add stripped version if different
            stripped = self._strip_parenthetical(t)
            if stripped != t:
                stripped_normalized = self._normalize_title(stripped)
                if stripped_normalized and stripped_normalized not in title_map:
                    title_map[stripped_normalized] = t

        # Use WRatio which combines multiple strategies (ratio, partial_ratio,
        # token_sort_ratio, token_set_ratio) and picks the best one based on
        # length differences. This handles articles ("The Girl from Ipanema" vs
        # "Girl from Ipanema") while avoiding false matches where a short title
        # like "L-O-V-E" matches anything containing "love".
        matches = process.extract(
            normalized_search,
            title_map.keys(),
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff,
            limit=limit,
        )

        # For each match, get all chart locations with that title
        results: List[FuzzyMatch] = []
        seen_locations: set[tuple[str, int]] = set()  # (source_path, page) to avoid duplicates

        for matched_title_norm, score, _ in matches:
            original_title = title_map[matched_title_norm]
            cursor.execute(
                "SELECT source_path, page, length FROM charts WHERE title = ? ORDER BY priority DESC",
                (original_title,),
            )
            rows = cursor.fetchall()
            for row in rows:
                location_key = (row[0], row[1])
                if location_key not in seen_locations:
                    seen_locations.add(location_key)
                    results.append(
                        FuzzyMatch(
                            location=ChartLocation(source_path=row[0], page=row[1], length=row[2]),
                            score=score,
                            matched_title=original_title,
                        )
                    )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

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
        # title_counts uses lowercase keys for case-insensitive duplicate detection
        title_counts: dict[str, int] = {}
        # Map lowercase title to a representative original title for display
        title_representatives: dict[str, str] = {}
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
                    # Store the first original title we see for this lowercase key
                    if title_lower not in title_representatives:
                        title_representatives[title_lower] = entry.title
                    if title_counts[title_lower] > 1:
                        # Use the representative original title for display
                        duplicate_titles.add(title_representatives[title_lower])

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
                # title_counts uses lowercase keys, but duplicate_titles contains original titles
                # We need to look up by the lowercase version
                title_lower = title.lower()
                count = title_counts.get(title_lower, 0)
                print(f"  - {title} ({count} instances)")

        return cls(output_path)

