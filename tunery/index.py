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
        Look up a chart by exact title.

        Args:
            title: The title to search for (exact match).

        Returns:
            ChartLocation if found, None otherwise.
        """
        cursor = self._get_connection().cursor()
        cursor.execute(
            "SELECT source_path, page, length FROM charts WHERE title = ?", (title,)
        )
        row = cursor.fetchone()

        if row:
            return ChartLocation(source_path=row[0], page=row[1], length=row[2])
        return None

    def lookup_fuzzy(self, title: str) -> list[ChartLocation]:
        """
        Look up charts by partial title match.

        Args:
            title: The title to search for (case-insensitive contains).

        Returns:
            List of matching ChartLocation objects.
        """
        cursor = self._get_connection().cursor()
        cursor.execute(
            "SELECT source_path, page, length FROM charts WHERE title LIKE ?",
            (f"%{title}%",),
        )
        rows = cursor.fetchall()
        return [ChartLocation(source_path=r[0], page=r[1], length=r[2]) for r in rows]

    @classmethod
    def build(cls, index_dir: Path, output_path: Path) -> "Index":
        """
        Scan a directory for JSON index files and build a SQLite index.

        Args:
            index_dir: Directory to scan for *.json files.
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
                length INTEGER NOT NULL DEFAULT 1
            )
        """)

        # Create index for fast title lookups
        cursor.execute("CREATE INDEX idx_title ON charts (title)")

        # Scan for JSON files
        json_files = list(index_dir.rglob("*.json"))
        total_charts = 0

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Validate the JSON structure
                index_file = IndexFile.model_validate(data)

                # Resolve the source path relative to the JSON file
                source_path = (json_file.parent / index_file.source).resolve()
                
                # Check that the source file exists
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"Source file not found: {source_path} (referenced in {json_file})"
                    )

                # Insert all entries
                for entry in index_file.index:
                    cursor.execute(
                        "INSERT INTO charts (title, source_path, page, length) VALUES (?, ?, ?, ?)",
                        (entry.title, str(source_path), entry.page, entry.pages),
                    )
                    total_charts += 1

            except Exception as e:
                print(f"Warning: Skipping {json_file}: {e}")
                continue

        conn.commit()
        conn.close()

        print(
            f"Indexed {total_charts} charts from {len(json_files)} files into {output_path}"
        )

        return cls(output_path)

