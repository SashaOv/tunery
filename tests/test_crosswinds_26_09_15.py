"""Crosswinds 26-09-15: three books from one title list, against the author's
hand-made layouts. Library file names come from the author's SheetMusic
(tests/data), rebuilt as blank PDFs."""

import re
from pathlib import Path

import pytest
import yaml
from helpers import create_pdfs

from tunery.render import render

DATA = Path(__file__).parent / "data"
LAYOUT = DATA / "crosswinds-26-09-15.1.yaml"
HAND_BOOKS = yaml.safe_load((DATA / "crosswinds-26-09-15-books.yaml").read_text())
LIBRARY = [
    line
    for line in (DATA / "crosswinds-26-09-15.txt").read_text().splitlines()
    if line and not line.startswith("#")
]

# Where the books differ from the hand layouts by design (author, 2026-10-15),
# title -> (files removed, files added).
BB_PARTS_TENOR = {
    "Take Five": "Take Five Bb Tenor.pdf",
    "Ceora": "Ceora Bb Tenor.pdf",
    "Mercy, Mercy, Mercy": "Mercy Mercy Mercy Bb Tenor.pdf",
    "Red Clay": "Red Clay Bb Tenor.pdf",
}
DIFFERENCES: dict[str, dict[str, tuple[set[str], set[str]]]] = {
    # The hand layout missed the Piano:Guitar dual score.
    "rs": {"Norwegian Wood": (set(), {"Norwegian Wood - Piano:Guitar.pdf"})},
    # `Bb Parts` tenor charts tie with the title folders' Bb charts and win
    # on path order (warned).
    "norm": {
        title: (set(HAND_BOOKS["norm"][title]), {chart})
        for title, chart in BB_PARTS_TENOR.items()
    },
    "full": {
        **{title: (set(), {chart}) for title, chart in BB_PARTS_TENOR.items()},
        # Trumpet's Bb and the Concert chart exist only in SheetMusic's other
        # folder (warned as charts from 2 folders).
        "The Cape Verdean Blues": (
            set(),
            {"The Cape Verdean Blues.Bb.pdf", "The Cape Verdean Blues.pdf"},
        ),
        # The {Tenor Sax: Flute} mapping applies to every book.
        "Funk in Deep Freeze": ({"Funk In Deep Freeze - Tenor.pdf"}, set()),
    },
}


@pytest.fixture(scope="module")
def books(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[dict, str]]:
    root = tmp_path_factory.mktemp("crosswinds")
    create_pdfs(root / "SheetMusic", LIBRARY)
    (root / "tunery.yaml").write_text("- library: SheetMusic\n")
    layout = root / "show.yaml"
    layout.write_text(LAYOUT.read_text())
    out: dict[str, tuple[dict, str]] = {}
    for book in ("rs", "norm", "full"):
        log = root / f"{book}.log"
        with log.open("w") as stream, pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.stdout", stream)
            render(layout, root / "show.pdf", part=[book], verbose=1)
        text = log.read_text()
        charts: dict[str, set[str]] = {}
        for title, chart in re.findall(r'^  "(.+?)" [^:]+: (.+\.pdf)$', text, re.MULTILINE):
            charts.setdefault(title, set()).add(chart)
        out[book] = (charts, text)
    return out


@pytest.mark.parametrize("book", ["rs", "norm", "full"])
def test_book_matches_hand_layout(book: str, books: dict) -> None:
    charts, _ = books[book]
    for title, hand in HAND_BOOKS[book].items():
        removed, added = DIFFERENCES[book].get(title, (set(), set()))
        assert charts.get(title, set()) == (set(hand) - removed) | added, title


@pytest.mark.parametrize("book", ["norm", "full"])
def test_bb_parts_choices_are_warned(book: str, books: dict) -> None:
    _, text = books[book]
    for title, chart in BB_PARTS_TENOR.items():
        warning = re.search(rf'^warning "{re.escape(title)}":\n((?:  .*\n)+)', text, re.MULTILINE)
        assert warning and f"Bb Parts/{chart}" in warning.group(1), title
