from pathlib import Path

import pikepdf
import pytest
import yaml

from tunery.render import render


def create_pdf(path: Path, page_count: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pikepdf.Pdf.new()
    for _ in range(page_count):
        pdf.add_blank_page(page_size=(100, 100))
    pdf.save(path)
    pdf.close()
    return path


def write_layout(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(list(records), sort_keys=False))
    return path


def make_fixture(root: Path) -> None:
    v = root / "Vault"
    create_pdf(v / "Aint Misbehavin - Concert F" / "Aint Misbehavin-Piano.pdf", 2)
    create_pdf(v / "Aint Misbehavin - Concert Bb" / "Aint Misbehavin-Piano.pdf", 5)
    (root / "tunery.yaml").write_text("- library: Vault\n")


def test_key_f_picks_concert_f(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    layout = write_layout(
        tmp_path / "gig.yaml", [{"title": "Aint Misbehavin", "key": "F"}]
    )

    render(layout, tmp_path / "book.pdf", part="Piano")

    with pikepdf.Pdf.open(tmp_path / "book.Piano.pdf") as pdf:
        assert len(pdf.pages) == 2


def test_key_bb_picks_concert_bb(tmp_path: Path) -> None:
    make_fixture(tmp_path)
    layout = write_layout(
        tmp_path / "gig.yaml", [{"title": "Aint Misbehavin", "key": "Bb"}]
    )

    render(layout, tmp_path / "book.pdf", part="Piano")

    with pikepdf.Pdf.open(tmp_path / "book.Piano.pdf") as pdf:
        assert len(pdf.pages) == 5


def test_key_e_warns_and_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_fixture(tmp_path)
    layout = write_layout(
        tmp_path / "gig.yaml", [{"title": "Aint Misbehavin", "key": "E"}]
    )

    render(layout, tmp_path / "book.pdf", part="Piano")

    out = capsys.readouterr().out
    assert 'no E version of "Aint Misbehavin"' in out
    assert (tmp_path / "book.Piano.pdf").exists()
