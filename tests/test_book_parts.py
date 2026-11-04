import json
from pathlib import Path

import pikepdf
import pytest
from helpers import create_pdf, create_pdfs, outline_titles, write_layout

from tunery.library import LibraryRecord
from tunery.render import render

# The New Bob Book ships one index for three printings: concert, Bb and Eb.
BOB_SOURCES = {
    "Concert": "Books/The_New_Bob_Book.pdf",
    "Bb": "Books/NBB Rev 6b PN BM Bb.pdf",
    "Eb": "Books/NBB Rev 6b PN BM Eb.pdf",
}


def make_bob(root: Path, titles: list[str]) -> None:
    for source in BOB_SOURCES.values():
        create_pdf(root / source, len(titles))
    (root / "Bob.json").write_text(
        json.dumps([{"title": t, "page": i + 1} for i, t in enumerate(titles)])
    )


def book_record(part: str | None) -> str:
    field = "" if part is None else f"  part: {part}\n"
    return f"- library: Bob.json\n  source: {BOB_SOURCES[part or 'Concert']}\n{field}"


def render_part(
    root: Path, title: str, part: str, capsys: pytest.CaptureFixture[str]
) -> tuple[str, Path]:
    layout = write_layout(root / "gig.yaml", [{"title": title}])
    capsys.readouterr()
    render(layout, root / "book.pdf", part=[part], verbose=1)
    label = "-".join(p.strip() for p in part.split(","))
    return capsys.readouterr().out, root / f"book.{label}.pdf"


@pytest.mark.parametrize(
    ("part", "chart", "step"),
    [
        ("Alto Sax", "NBB Rev 6b PN BM Eb.pdf", "Eb"),
        ("Bari Sax", "NBB Rev 6b PN BM Eb.pdf", "Eb"),
        ("Eb", "NBB Rev 6b PN BM Eb.pdf", None),
        ("Tenor Sax", "NBB Rev 6b PN BM Bb.pdf", "Bb"),
        ("Trumpet", "NBB Rev 6b PN BM Bb.pdf", "Bb"),
        ("Trombone", "The_New_Bob_Book.pdf", "Concert"),
        ("Piano", "The_New_Bob_Book.pdf", "Concert"),
        ("Vocal", "The_New_Bob_Book.pdf", "Concert"),
        ("Concert", "The_New_Bob_Book.pdf", None),
    ],
)
def test_book_reaching_part_in_fewest_steps_wins(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    part: str,
    chart: str,
    step: str | None,
) -> None:
    make_bob(tmp_path, ["Cantaloupe Island"])
    # Keyed books in a collection, the concert book listed last (highest
    # priority), as in a project that includes a whole book shelf.
    (tmp_path / "Bob.yaml").write_text(book_record("Eb") + book_record("Bb"))
    (tmp_path / "tunery.yaml").write_text(
        "- library: Bob.yaml\n" + book_record(None)
    )

    out, book = render_part(tmp_path, "Cantaloupe Island", part, capsys)

    assert f'"Cantaloupe Island" {part}: {chart}' in out
    if step is not None:
        assert f"Cantaloupe Island: {part} → {step}" in out
    assert book.exists() or part == "Concert"


def test_combined_parts_take_each_from_its_own_book(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_bob(tmp_path, ["Cantaloupe Island"])
    (tmp_path / "tunery.yaml").write_text(book_record("Eb") + book_record(None))

    out, book = render_part(tmp_path, "Cantaloupe Island", "Alto Sax,Piano", capsys)

    assert '"Cantaloupe Island" Alto Sax: NBB Rev 6b PN BM Eb.pdf' in out
    assert '"Cantaloupe Island" Piano: The_New_Bob_Book.pdf' in out
    assert outline_titles(book) == [("Cantaloupe Island", ["Eb", "Concert"])]


@pytest.mark.parametrize(
    ("part", "found"),
    [("Alto Sax", True), ("Tenor Sax", False), ("Concert", False)],
)
def test_keyed_book_answers_only_parts_reaching_its_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], part: str, found: bool
) -> None:
    make_bob(tmp_path, ["Cantaloupe Island"])
    (tmp_path / "tunery.yaml").write_text(book_record("Eb"))

    out, _ = render_part(tmp_path, "Cantaloupe Island", part, capsys)

    if found:
        assert 'found    "Cantaloupe Island"' in out
    else:
        assert f'missing {part} for "Cantaloupe Island"' in out
        assert 'not found "Cantaloupe Island"' in out


def test_equally_good_books_go_by_priority(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_bob(tmp_path, ["Cantaloupe Island"])
    create_pdf(tmp_path / "Books" / "Other Eb.pdf")
    (tmp_path / "tunery.yaml").write_text(
        book_record("Eb")
        + "- library: Bob.json\n  source: Books/Other Eb.pdf\n  part: Eb\n"
        + book_record(None)
    )

    out, _ = render_part(tmp_path, "Cantaloupe Island", "Alto Sax", capsys)

    assert '"Cantaloupe Island" Alto Sax: Other Eb.pdf' in out


def test_folder_arrangement_keeps_priority_over_keyed_book(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_bob(tmp_path, ["Cold Duck Time"])
    create_pdfs(
        tmp_path / "Vault",
        ["Cold Duck Time/Cold Duck Time (JBM Arrangement)- C Instruments.pdf"],
    )
    (tmp_path / "tunery.yaml").write_text(book_record("Eb") + "- library: Vault\n")

    out, _ = render_part(tmp_path, "Cold Duck Time", "Alto Sax", capsys)

    assert (
        '"Cold Duck Time" Alto Sax: Cold Duck Time (JBM Arrangement)- C Instruments.pdf'
        in out
    )


def test_keyed_book_behind_folder_without_the_title(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_bob(tmp_path, ["Cantaloupe Island"])
    create_pdfs(tmp_path / "Vault", ["Blue Bossa/Blue Bossa.pdf"])
    (tmp_path / "tunery.yaml").write_text(
        book_record("Eb") + book_record(None) + "- library: Vault\n"
    )

    out, _ = render_part(tmp_path, "Cantaloupe Island", "Alto Sax", capsys)

    assert '"Cantaloupe Island" Alto Sax: NBB Rev 6b PN BM Eb.pdf' in out


def test_layout_config_declares_book_part(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_bob(tmp_path, ["Cantaloupe Island"])
    layout = write_layout(
        tmp_path / "gig.yaml",
        [
            {
                "config": [
                    {"library": "Bob.json", "source": BOB_SOURCES["Eb"], "part": "Eb"},
                    {"library": "Bob.json", "source": BOB_SOURCES["Concert"]},
                    {"part": {"horns": "Alto Sax"}},
                ]
            },
            {"title": "Cantaloupe Island"},
        ],
    )
    capsys.readouterr()

    render(layout, tmp_path / "book.pdf", part=["horns"], verbose=1)

    assert '"Cantaloupe Island" Alto Sax: NBB Rev 6b PN BM Eb.pdf' in capsys.readouterr().out


def test_keyed_book_pages_follow_its_index(tmp_path: Path) -> None:
    make_bob(tmp_path, ["Blue Bossa", "Cantaloupe Island"])
    (tmp_path / "tunery.yaml").write_text(book_record("Eb") + book_record(None))
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Cantaloupe Island"}])

    render(layout, tmp_path / "book.pdf", part=["Alto Sax"])

    with pikepdf.Pdf.open(tmp_path / "book.Alto Sax.pdf") as pdf:
        assert len(pdf.pages) == 1


def test_book_part_accepts_instrument_and_aliases(tmp_path: Path) -> None:
    make_bob(tmp_path, ["Cantaloupe Island"])
    library = LibraryRecord(
        library="Bob.json", source=BOB_SOURCES["Eb"], part="alto"
    ).load(tmp_path)

    selection = library.select_parts("Cantaloupe Island", ["Alto Sax", "Bari Sax"])

    assert selection is not None
    assert [(h.requested, h.step, h.label) for h in selection.hits] == [
        ("Alto Sax", "Alto Sax", "Alto Sax")
    ]
    assert selection.missing == ["Bari Sax"]


def test_book_part_rejects_unknown_value(tmp_path: Path) -> None:
    make_bob(tmp_path, ["Cantaloupe Island"])

    with pytest.raises(ValueError, match='Unknown part "E flat"') as error:
        LibraryRecord(
            library="Bob.json", source=BOB_SOURCES["Eb"], part="E flat"
        ).load(tmp_path)
    assert "Bob.json" in str(error.value)


@pytest.mark.parametrize("target", ["charts", "shelf.yaml"])
def test_part_is_only_for_books(tmp_path: Path, target: str) -> None:
    (tmp_path / "charts").mkdir()
    (tmp_path / "shelf.yaml").write_text("- library: charts\n")

    with pytest.raises(ValueError, match="must not specify part"):
        LibraryRecord(library=target, part="Eb").load(tmp_path)
