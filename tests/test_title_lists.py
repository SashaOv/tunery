"""Plain title lists as layouts: one title per line (.txt) or a Title column (.csv)."""

from pathlib import Path

import pytest
from helpers import create_pdfs, outline_titles

import tunery.render as render_module
from tunery.main import main
from tunery.render import render

TITLES = ["Giant Steps", "Mercy, Mercy, Mercy", "So What"]


@pytest.fixture
def show(tmp_path: Path) -> Path:
    create_pdfs(tmp_path / "SheetMusic", [f"{title}.pdf" for title in TITLES])
    (tmp_path / "tunery.yaml").write_text("- library: SheetMusic\n")
    return tmp_path


def rendered_titles(show: Path, source: Path) -> list[str]:
    out = show / "book.pdf"
    render(source, out)
    return [title for title, _ in outline_titles(out)]


def test_txt_lines_are_trimmed_and_blank_lines_skipped(show: Path) -> None:
    source = show / "setlist.txt"
    source.write_text("  Giant Steps \n\n\t\nMercy, Mercy, Mercy\r\n   \r\nSo What")

    assert rendered_titles(show, source) == TITLES


def test_txt_tolerates_a_byte_order_mark_and_an_uppercase_suffix(show: Path) -> None:
    source = show / "SETLIST.TXT"
    source.write_bytes(b"\xef\xbb\xbfGiant Steps\nSo What\n")

    assert rendered_titles(show, source) == ["Giant Steps", "So What"]


def test_csv_takes_the_title_column_wherever_it_is(show: Path) -> None:
    source = show / "setlist.csv"
    source.write_text(
        'Key,Title,Notes\n'
        'F,Giant Steps,"fast, then slower"\n'
        'C,"Mercy, Mercy, Mercy",\n'
        'D,  So What  ,\n'
    )

    assert rendered_titles(show, source) == TITLES


def test_csv_column_name_ignores_case_and_spaces(show: Path) -> None:
    source = show / "setlist.csv"
    source.write_text(" TITLE ,Key\nGiant Steps,F\n")

    assert rendered_titles(show, source) == ["Giant Steps"]


def test_csv_skips_empty_cells_and_short_rows(show: Path) -> None:
    source = show / "setlist.csv"
    source.write_text("Key,Title\nF,Giant Steps\nC,\nD\n\nE,So What\n")

    assert rendered_titles(show, source) == ["Giant Steps", "So What"]


def test_csv_tolerates_a_byte_order_mark(show: Path) -> None:
    source = show / "setlist.csv"
    source.write_bytes(b"\xef\xbb\xbfTitle\nGiant Steps\n")

    assert rendered_titles(show, source) == ["Giant Steps"]


def test_csv_title_columns_come_from_a_constant(
    show: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = show / "setlist.csv"
    source.write_text("Title,Song\nSo What,Giant Steps\n")

    assert rendered_titles(show, source) == ["So What"]
    monkeypatch.setattr(render_module, "CSV_TITLE_COLUMNS", ("Song", "Title"))
    assert rendered_titles(show, source) == ["Giant Steps"]


def test_csv_without_a_title_column_names_the_columns_it_has(show: Path) -> None:
    source = show / "setlist.csv"
    source.write_text("Song,Key\nGiant Steps,F\n")

    with pytest.raises(ValueError, match=r"setlist\.csv.*Title.*Song, Key"):
        render(source, show / "book.pdf")


def test_an_empty_csv_has_no_title_column(show: Path) -> None:
    source = show / "empty.csv"
    source.write_text("")

    with pytest.raises(ValueError, match=r"No title column in .*empty\.csv.*found none"):
        render(source, show / "book.pdf")


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("empty.txt", ""),
        ("blank.txt", "\n  \n\t\n"),
        ("header-only.csv", "Title\n"),
        ("blank-cells.csv", "Title\n\n  \n"),
    ],
)
def test_an_input_without_titles_is_an_error(show: Path, name: str, content: str) -> None:
    source = show / name
    source.write_text(content)

    with pytest.raises(ValueError, match=rf"No titles in .*{name}"):
        render(source, show / "book.pdf")
    assert not (show / "book.pdf").exists()


@pytest.mark.parametrize("name", ["setlist.txt", "setlist.csv"])
def test_a_file_that_is_not_utf8_is_an_error(show: Path, name: str) -> None:
    source = show / name
    source.write_bytes("Title\nCaf\xe9\n".encode("latin-1"))

    with pytest.raises(ValueError, match=rf"{name}.*UTF-8"):
        render(source, show / "book.pdf")


def test_a_missing_input_file_is_an_error(show: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.txt"):
        render(show / "nope.txt", show / "book.pdf")


def test_titles_without_a_chart_are_reported_and_the_rest_rendered(
    show: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = show / "setlist.txt"
    source.write_text("Giant Steps\nNo Such Tune\nSo What\n")

    assert rendered_titles(show, source) == ["Giant Steps", "So What"]
    assert 'not found "No Such Tune"' in capsys.readouterr().out


def test_cli_writes_the_book_next_to_the_input(show: Path) -> None:
    (show / "lists").mkdir()
    source = show / "lists" / "gig.csv"
    source.write_text("Title\nGiant Steps\n")

    main(["render", str(source)])

    assert [title for title, _ in outline_titles(show / "lists" / "gig.pdf")] == [
        "Giant Steps"
    ]


def test_part_books_work_from_a_title_list(show: Path) -> None:
    create_pdfs(show / "SheetMusic", ["So What - Bass.pdf"])
    source = show / "setlist.txt"
    source.write_text("So What\n")

    render(source, show / "book.pdf", part=["Bass"])

    assert [t for t, _ in outline_titles(show / "book.Bass.pdf")] == ["So What"]
