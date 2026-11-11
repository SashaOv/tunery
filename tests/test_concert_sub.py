from pathlib import Path

import pytest
from helpers import create_pdf, create_pdfs, outline_titles, write_layout

from tunery.library import LibraryRecord
from tunery.parts import chain_for_part, parse_part_name
from tunery.render import render
from tunery.settings import Settings


@pytest.mark.parametrize("name", ["C", "c", " C "])
def test_c_is_concert(name: str) -> None:
    assert parse_part_name(name) == "Concert"
    assert chain_for_part(name) == chain_for_part("Concert")


@pytest.mark.parametrize(
    ("part", "chain"),
    [
        ("Alto Sax", ["Alto Sax", "Eb"]),
        ("Bari Sax", ["Bari Sax", "Eb"]),
        ("Alto Sax 2", ["Alto Sax 2", "Alto Sax", "Eb"]),
        ("Tenor Sax", ["Tenor Sax", "Bb"]),
        ("Trumpet", ["Trumpet", "Bb"]),
        ("Eb", ["Eb"]),
        ("Bb", ["Bb"]),
    ],
)
def test_transposing_part_stops_at_its_instrument_group(
    part: str, chain: list[str]
) -> None:
    assert chain_for_part(part, concert_sub=False) == chain
    assert chain_for_part(part)[: len(chain)] == chain
    assert "Concert" in chain_for_part(part)


@pytest.mark.parametrize(
    "part",
    [
        "Concert",
        "Lead Sheet",
        "Melody",
        "Piano",
        "Guitar",
        "Bass",
        "Drums",
        "Rhythm",
        "Flute",
        "Vocal",
        "Trombone",
        "Bass Clef",
        "Saxophone",
        "Horn 1",
    ],
)
def test_concert_parts_keep_their_chain(part: str) -> None:
    assert chain_for_part(part, concert_sub=False) == chain_for_part(part)


def render_part(
    root: Path,
    part: str,
    capsys: pytest.CaptureFixture[str],
    *,
    concert_sub: bool,
    title: str = "Blue Bossa",
) -> tuple[str, Path]:
    layout = write_layout(root / "gig.yaml", [{"title": title}])
    capsys.readouterr()
    render(
        layout,
        root / "book.pdf",
        part=[part],
        verbose=1,
        settings=Settings(concert_sub=concert_sub),
    )
    label = "-".join(p.strip() for p in part.split(","))
    return capsys.readouterr().out, root / f"book.{label}.pdf"


def configure(root: Path, charts: list[str]) -> None:
    create_pdfs(root / "Vault", charts)
    (root / "tunery.yaml").write_text("- library: Vault\n")


def test_concert_chart_is_not_an_alto_part_without_concert_sub(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(tmp_path, ["Blue Bossa/Blue Bossa - Concert.pdf"])

    out, book = render_part(tmp_path, "Alto Sax", capsys, concert_sub=False)

    assert 'missing Alto Sax for "Blue Bossa"' in out
    assert 'not found "Blue Bossa"' in out
    assert not book.exists() or outline_titles(book) == []


def test_concert_sub_is_the_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(tmp_path, ["Blue Bossa/Blue Bossa - Concert.pdf"])
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Blue Bossa"}])
    capsys.readouterr()

    render(layout, tmp_path / "book.pdf", part=["Alto Sax"], verbose=1)

    out = capsys.readouterr().out
    assert "Blue Bossa: Alto Sax → Concert" in out


@pytest.mark.parametrize("concert_sub", [True, False])
def test_eb_chart_still_serves_alto_sax(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], concert_sub: bool
) -> None:
    configure(
        tmp_path,
        ["Blue Bossa/Blue Bossa - Concert.pdf", "Blue Bossa/Blue Bossa - Eb.pdf"],
    )

    out, _ = render_part(tmp_path, "Alto Sax", capsys, concert_sub=concert_sub)

    assert '"Blue Bossa" Alto Sax: Blue Bossa - Eb.pdf' in out
    assert "missing" not in out


def test_combined_parts_substitute_only_concert_instruments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(tmp_path, ["Blue Bossa/Blue Bossa - Concert.pdf"])

    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Blue Bossa"}])
    capsys.readouterr()

    render(layout, tmp_path / "book.pdf", part=["Alto Sax,Piano,C"], verbose=1, settings=Settings(concert_sub=False))

    out = capsys.readouterr().out
    assert 'missing Alto Sax for "Blue Bossa"' in out
    assert '"Blue Bossa" Piano: Blue Bossa - Concert.pdf' in out
    assert (tmp_path / "book.Alto Sax-Piano-Concert.pdf").exists()


def test_concert_part_requested_as_c(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(tmp_path, ["Blue Bossa/Blue Bossa - Concert.pdf"])
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Blue Bossa"}])

    render(layout, tmp_path / "book.pdf", part=["C"], settings=Settings(concert_sub=False))

    assert (tmp_path / "book.pdf").exists()


def test_mapped_part_follows_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure(tmp_path, ["Blue Bossa/Blue Bossa - Concert.pdf"])
    layout = write_layout(
        tmp_path / "gig.yaml",
        [{"title": "Blue Bossa", "part": {"Alto Sax": "Tenor Sax"}}],
    )
    capsys.readouterr()

    render(layout, tmp_path / "book.pdf", part=["Alto Sax"], verbose=1, settings=Settings(concert_sub=False))

    assert 'missing Tenor Sax for "Blue Bossa"' in capsys.readouterr().out


def test_concert_book_does_not_serve_alto_sax_without_concert_sub(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdf(tmp_path / "Books" / "Real Book.pdf", 2)
    (tmp_path / "book.json").write_text('[{"title": "Blue Bossa", "page": 1}]')
    (tmp_path / "tunery.yaml").write_text(
        "- library: book.json\n  source: Books/Real Book.pdf\n"
    )

    out, _ = render_part(tmp_path, "Alto Sax", capsys, concert_sub=False)

    assert 'missing Alto Sax for "Blue Bossa"' in out
    assert 'not found "Blue Bossa"' in out


@pytest.mark.parametrize(
    ("part", "chart"),
    [("Alto Sax", "Eb.pdf"), ("Piano", "Concert.pdf"), ("Concert", "Concert.pdf")],
)
def test_books_follow_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], part: str, chart: str
) -> None:
    create_pdf(tmp_path / "Books" / "Concert.pdf", 2)
    create_pdf(tmp_path / "Books" / "Eb.pdf", 2)
    (tmp_path / "book.json").write_text('[{"title": "Blue Bossa", "page": 1}]')
    (tmp_path / "tunery.yaml").write_text(
        "- library: book.json\n  source: Books/Eb.pdf\n  part: Eb\n"
        "- library: book.json\n  source: Books/Concert.pdf\n"
    )

    out, _ = render_part(tmp_path, part, capsys, concert_sub=False)

    assert f'"Blue Bossa" {part}: {chart}' in out


def test_library_part_accepts_c(tmp_path: Path) -> None:
    create_pdf(tmp_path / "Real Book.pdf")
    (tmp_path / "book.json").write_text("[]")

    library = LibraryRecord(
        library="book.json", source="Real Book.pdf", part="C"
    ).load(tmp_path)

    assert library.part == "Concert"
