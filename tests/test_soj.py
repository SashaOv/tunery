"""SOJ per-instrument books from title-only setlists (LIB7.4).

MasterList file names, the converted index and the hand books' charts come
from tests/data (names copied from the author's drive); the fixture rebuilds
them as blank PDFs. Jumbo/index-targeted charts get 60 pages so every index
range fits; part files found by name stay one page.
"""

from pathlib import Path

import pikepdf
import pytest
import yaml
from helpers import create_pdf

from tunery.library import load_project_libraries
from tunery.render import render

DATA = Path(__file__).parent / "data"
LIBRARY = [
    line
    for line in (DATA / "soj-masterlist.txt").read_text().splitlines()
    if line and not line.startswith("#")
]
INDEX = yaml.safe_load((DATA / "soj-masterlist-index.yaml").read_text())
BOOKS = yaml.safe_load((DATA / "soj-books.yaml").read_text())

# Part files found by name render whole (one fixture page); every other
# chart is a jumbo or explicit index target with room for its ranges.
PART_FILES = {
    "Hay Burner-Nestico/Bass.pdf",
    "Hay Burner-Nestico/Piano.pdf",
    "Gato Gordo-Burger/Gato - Bass.pdf",
    "Gato Gordo-Burger/Gato - Piano.pdf",
    "La Mer (Beyond the Sea)/Bass.pdf",
    "La Mer (Beyond the Sea)/Piano.pdf",
    "Hooked on a Feeling-Burger/Hooked On A Feeling - Bass.pdf",
    "Hooked on a Feeling-Burger/Hooked On A Feeling - Keys.pdf",
    "I Feel Good-Burger/Finale 2004a - [IFEELGOOD - 016 Bass].pdf",
    "I Feel Good-Burger/Finale 2004a - [IFEELGOOD - 015 Piano].pdf",
    "Mack the Knife (Bb) (Bb)-Berger/Mack The Knife bss.pdf",
}

JUMBO_PAGES = 60

# (file, page, length) the title-only rendering picks where it differs from
# the hand book. The hand books take Sweet Home Chicago from Repertoire's
# renamed copy (`-Clark.pdf`). MasterList has only `Sweet Home Chicago.Bb.pdf`;
# `.Bb` is the score key, so the index finds that jumbo with no `file:`.
# Walking My Baby Back Home's piano is [31, 5] in the converted index, unlike
# the hand layout's 31-34.
SWEET_HOME = "Sweet Home Chicago.Bb.pdf"
EXPECTED = {
    ("bass-26-07-09", "Sweet Home Chicago"): (SWEET_HOME, 27, 2),
    ("piano-26-07-09", "Sweet Home Chicago"): (SWEET_HOME, 29, 3),
    ("bass-26-08-07", "Sweet Home Chicago"): (SWEET_HOME, 27, 2),
    ("piano-26-07-09", "Walking My Baby Back Home"): (
        "Walking My Baby Back Home-May.pdf",
        31,
        5,
    ),
}

BOOK_PART = {
    "bass-26-07-09": "Bass",
    "piano-26-07-09": "Piano",
    "bass-26-08-07": "Bass",
    "bass-26-09-12": "Bass",
}


@pytest.fixture(scope="module")
def show(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("soj")
    for name in LIBRARY:
        create_pdf(root / "MasterList" / name, 1 if name in PART_FILES else 60)
    (root / "tunery.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "library": "MasterList",
                    "type": "big band",
                    "match": "fuzzy",
                    "index": INDEX,
                }
            ]
        )
    )
    return root


def render_book(root: Path, book: str, capsys) -> tuple[Path, str]:
    layout = root / f"{book}.yaml"
    layout.write_text(
        yaml.safe_dump([{"title": title} for title in BOOKS[book]["titles"]])
    )
    out = root / f"{book}.pdf"
    capsys.readouterr()
    render(layout, out, part=[BOOK_PART[book]], verbose=1)
    pdf = out.with_name(f"{out.stem}.{BOOK_PART[book]}{out.suffix}")
    return pdf, capsys.readouterr().out


def picked(root: Path, book: str, title: str) -> tuple[str, int | None, int | None]:
    libraries = load_project_libraries(root)
    selection = libraries.select_parts(title, [BOOK_PART[book]])
    assert selection is not None and len(selection.hits) == 1, title
    match = selection.hits[0].match
    return (match.source.name, match.page, match.length)


def check_pick(root: Path, book: str, title: str, hand: dict) -> None:
    """Picked file, start page and, where the hand pins one, length."""
    name, page, length = picked(root, book, title)
    assert name == hand["file"].split("/")[-1], title
    assert page == hand["page"], title
    if hand["length"] is not None:
        assert length == hand["length"], title


def spans(pdf_path: Path) -> list[tuple[str, int]]:
    with pikepdf.Pdf.open(pdf_path) as pdf, pdf.open_outline() as outline:
        items = list(outline.root)
        starts = [pdf.pages.index(item.destination[0]) for item in items]
        total = len(pdf.pages)
    return [
        (item.title, (starts[i + 1] if i + 1 < len(starts) else total) - starts[i])
        for i, item in enumerate(items)
    ]


def check_book(root: Path, capsys, book: str) -> None:
    pdf, out = render_book(root, book, capsys)
    assert "warning" not in out
    charts = BOOKS[book]["charts"]
    assert [title for title, _ in spans(pdf)] == BOOKS[book]["titles"]
    for title in BOOKS[book]["titles"]:
        hand = charts[title]
        expected = EXPECTED.get((book, title))
        if expected is not None:
            assert picked(root, book, title) == expected, title
        else:
            check_pick(root, book, title, hand)
        # A hand page without a length is one page, like a bare index page.
        length = hand["length"] if hand["length"] is not None else 1
        assert dict(spans(pdf))[title] == length, title


def test_sweet_home_chicago_index_names_no_file() -> None:
    entry = next(item for item in INDEX if item["title"] == "Sweet Home Chicago")
    assert "file" not in entry


def test_26_07_09_bass_15_titles(show: Path, capsys) -> None:
    check_book(show, capsys, "bass-26-07-09")


def test_26_07_09_piano_15_titles(show: Path, capsys) -> None:
    check_book(show, capsys, "piano-26-07-09")


def test_26_08_07_bass_21_titles(show: Path, capsys) -> None:
    check_book(show, capsys, "bass-26-08-07")


def test_26_09_12_bass_16_of_25(show: Path, capsys) -> None:
    book = "bass-26-09-12"
    pdf, out = render_book(show, book, capsys)
    assert "warning" not in out
    charts = BOOKS[book]["charts"]
    for title in BOOKS[book]["titles"]:
        if title in charts:
            hand = charts[title]
            expected = EXPECTED.get((book, title))
            if expected is not None:
                assert picked(show, book, title) == expected, title
            else:
                check_pick(show, book, title, hand)
    for title in BOOKS[book]["missing"]:
        assert out.count(f'missing Bass for "{title}"') == 1, title
    not_found = BOOKS[book].get("not_found", [])
    for title in not_found:
        assert f'not found "{title}"' in out, title
        assert f'missing Bass for "{title}"' not in out, title
    assert (
        'matched "I\'m Gonna Sit Right Down" to '
        '"I\u2019m Gonna Sit Right Down and Write Myself a Letter"'
    ) in out
    assert out.count("missing Bass") == len(BOOKS[book]["missing"])
    assert out.count("not found") == len(BOOKS[book]["missing"]) + len(not_found)
    assert [title for title, _ in spans(pdf)] == list(charts)
