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
    create_pdf(v / "Mack" / "Mack-Piano.pdf", 1)
    create_pdf(v / "Heartaches" / "Heartaches-Concert.pdf", 1)
    (root / "tunery.yaml").write_text("- library: Vault\n")


def test_default_output_unchanged(tmp_path: Path, capsys) -> None:
    make_fixture(tmp_path)
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Mack"}])

    render(layout, tmp_path / "book.pdf", part="Piano")

    out = capsys.readouterr().out
    assert 'found    "Mack"' in out
    assert "Piano →" not in out
    assert "candidate" not in out


def test_verbose1_reports_file_and_substitution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_fixture(tmp_path)
    layout = write_layout(
        tmp_path / "gig.yaml", [{"title": "Mack"}, {"title": "Heartaches"}]
    )

    render(layout, tmp_path / "book.pdf", part="Piano", verbose=1)

    out = capsys.readouterr().out
    assert "Mack-Piano.pdf" in out
    assert "Heartaches-Concert.pdf" in out
    assert "  Piano → Concert" not in out


def test_verbose2_adds_candidates_and_rules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_fixture(tmp_path)
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Mack"}])

    render(layout, tmp_path / "book.pdf", part="Piano", verbose=2)

    out = capsys.readouterr().out
    assert "candidate" in out
    assert "Mack-Piano.pdf" in out
    assert "part=Piano" in out
    assert "rules=" in out
    assert "instrument" in out


def test_substitutions_warned_once_per_title_without_verbose(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_fixture(tmp_path)
    layout = write_layout(
        tmp_path / "gig.yaml", [{"title": "Mack"}, {"title": "Heartaches"}]
    )

    render(layout, tmp_path / "book.pdf", part="Piano")

    out = capsys.readouterr().out
    assert "Heartaches: Piano → Concert" in out
    assert out.count("Heartaches: Piano → Concert") == 1
    assert "Mack:" not in out
