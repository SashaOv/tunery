"""Part index in the library definition (LIB7.2)."""

from pathlib import Path

import pikepdf
import pytest
import yaml
from pydantic import ValidationError

from tunery.library import LibraryIndexEntry, _index_ranges
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


def test_value_shapes() -> None:
    assert _index_ranges(15) == [(15, 1)]
    assert _index_ranges((17, 3)) == [(17, 3)]
    assert _index_ranges([(26, 1), (25, 1)]) == [(26, 1), (25, 1)]
    entry = LibraryIndexEntry.model_validate(
        {"title": "T", "parts": {"bass": 15, "piano": [17, 3]}}
    )
    assert _index_ranges(entry.parts["bass"]) == [(15, 1)]
    assert _index_ranges(entry.parts["piano"]) == [(17, 3)]
    with pytest.raises(ValidationError):
        LibraryIndexEntry.model_validate({"title": "T", "parts": {"bass": [27]}})


def test_entry_without_file_uses_jumbo(tmp_path: Path) -> None:
    create_pdf(tmp_path / "Band" / "Bandstand Boogie-Lowden.pdf", 20)
    (tmp_path / "tunery.yaml").write_text(
        "- library: Band\n"
        "  type: big band\n"
        "  index:\n"
        "    - title: Bandstand Boogie\n"
        "      parts: {bass: 15, piano: [17, 3]}\n"
    )
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Bandstand Boogie"}])

    render(layout, tmp_path / "show.pdf", part=["Piano"])

    with pikepdf.Pdf.open(tmp_path / "show.Piano.pdf") as pdf:
        assert len(pdf.pages) == 3


def test_file_names_jumbo_explicitly(tmp_path: Path) -> None:
    create_pdf(tmp_path / "Band" / "Skin-A.pdf", 30)
    create_pdf(tmp_path / "Band" / "Skin-B.pdf", 30)
    (tmp_path / "tunery.yaml").write_text(
        "- library: Band\n"
        "  type: big band\n"
        "  index:\n"
        "    - title: Skin\n"
        "      file: Skin-A.pdf\n"
        "      parts: {bass: [5, 2]}\n"
    )
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Skin"}])

    render(layout, tmp_path / "show.pdf", part=["Bass"])

    with pikepdf.Pdf.open(tmp_path / "show.Bass.pdf") as pdf:
        assert len(pdf.pages) == 2


def test_no_file_and_several_files_warn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdf(tmp_path / "Band" / "Two-A.pdf", 10)
    create_pdf(tmp_path / "Band" / "Two-B.pdf", 10)
    (tmp_path / "tunery.yaml").write_text(
        "- library: Band\n"
        "  type: big band\n"
        "  index:\n"
        "    - title: Gone\n"
        "      parts: {bass: 3}\n"
        "    - title: Two\n"
        "      parts: {bass: 3}\n"
    )
    layout = write_layout(
        tmp_path / "show.yaml", [{"title": "Gone"}, {"title": "Two"}]
    )

    render(layout, tmp_path / "show.pdf", part=["Bass"])

    out = capsys.readouterr().out
    assert 'no jumbo file for "Gone"' in out
    assert 'several jumbo files for "Two"' in out
    assert out.count('missing Bass for "Gone"') == 1
    assert out.count('missing Bass for "Two"') == 1


def test_index_part_and_part_file_coexist(tmp_path: Path) -> None:
    create_pdf(tmp_path / "Band" / "Gato Gordo" / "Gato-Gordo-Piano.pdf", 4)
    create_pdf(tmp_path / "Band" / "Gato Gordo" / "jumbo.pdf", 10)
    (tmp_path / "tunery.yaml").write_text(
        "- library: Band\n"
        "  type: big band\n"
        "  index:\n"
        "    - title: Gato Gordo\n"
        "      file: Gato Gordo/jumbo.pdf\n"
        "      parts: {bass: 5}\n"
    )
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Gato Gordo"}])

    render(layout, tmp_path / "show.pdf", part=["Piano,Bass"])

    with pikepdf.Pdf.open(tmp_path / "show.Piano+Bass.pdf") as pdf:
        assert len(pdf.pages) == 5
        with pdf.open_outline() as outline:
            root = list(outline.root)
            assert root[0].title == "Gato Gordo"
            assert sorted(c.title for c in root[0].children) == ["Bass", "Piano"]


def test_ranges_render_in_order_under_one_bookmark(tmp_path: Path) -> None:
    jumbo = tmp_path / "Band" / "Come Dance With Me.pdf"
    jumbo.parent.mkdir(parents=True, exist_ok=True)
    pdf = pikepdf.Pdf.new()
    for n in range(1, 31):
        pdf.add_blank_page(page_size=(200, 200) if n == 26 else (100, 100))
    pdf.save(jumbo)
    pdf.close()
    (tmp_path / "tunery.yaml").write_text(
        "- library: Band\n"
        "  type: big band\n"
        "  index:\n"
        "    - title: Come Dance With Me\n"
        "      parts: {bass: [[26, 1], [25, 1]]}\n"
    )
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Come Dance With Me"}])

    render(layout, tmp_path / "show.pdf", part=["Bass"])

    with pikepdf.Pdf.open(tmp_path / "show.Bass.pdf") as out:
        assert len(out.pages) == 2
        assert [int(p.mediabox[2]) for p in out.pages] == [200, 100]
        with out.open_outline() as outline:
            root = list(outline.root)
            assert [item.title for item in root] == ["Come Dance With Me"]
            assert list(root[0].children) == []
