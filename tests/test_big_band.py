"""Big-band library type: jumbo, combined-set and conductor scores (LIB7.1)."""

from pathlib import Path

import pikepdf
import pytest
from helpers import create_pdf, create_pdfs, write_layout

from tunery.render import render


def make_band(root: Path) -> None:
    create_pdfs(
        root / "MasterList",
        [
            "Bandstand Boogie-Lowden.pdf",
            "Combo Set/Rhythm Parts.pdf",
            "Hay Burner-Nestico/Conductor.pdf",
            "Hay Burner-Nestico/Bass.pdf",
        ],
    )
    (root / "tunery.yaml").write_text("- library: MasterList\n  type: big band\n")


def test_jumbo_and_combined_set_listed_not_matched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_band(tmp_path)
    layout = write_layout(
        tmp_path / "show.yaml",
        [{"title": "Bandstand Boogie"}, {"title": "Combo Set"}],
    )

    render(layout, tmp_path / "show.pdf", part=["Bass"], verbose=2)

    out = capsys.readouterr().out
    assert "part=Score" in out
    assert "part=Rhythm" in out
    assert out.count('missing Bass for "Bandstand Boogie"') == 1
    assert out.count('missing Bass for "Combo Set"') == 1


def test_conductor_never_answers_piano(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_band(tmp_path)
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Hay Burner"}])

    render(layout, tmp_path / "show.pdf", part=["Piano"], verbose=2)

    out = capsys.readouterr().out
    assert "part=Conductor" in out
    assert 'missing Piano for "Hay Burner"' in out


def test_conductor_stays_concert_in_combo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(tmp_path / "Charts", ["Hay Burner-Nestico/Conductor.pdf"])
    (tmp_path / "tunery.yaml").write_text("- library: Charts\n")
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Hay Burner"}])

    render(layout, tmp_path / "show.pdf", part=["Piano"], verbose=2)

    out = capsys.readouterr().out
    assert "part=Concert" in out
    assert "missing Piano" not in out


def test_sweet_home_chicago_bb_is_jumbo_with_score_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdf(tmp_path / "MasterList" / "Sweet Home Chicago.Bb.pdf", 30)
    (tmp_path / "tunery.yaml").write_text(
        "- library: MasterList\n"
        "  type: big band\n"
        "  index:\n"
        "    - title: Sweet Home Chicago\n"
        "      parts: {bass: [27, 2]}\n"
    )
    layout = write_layout(
        tmp_path / "show.yaml", [{"title": "Sweet Home Chicago"}]
    )

    render(layout, tmp_path / "show.pdf", part=["Bass"], verbose=2)

    out = capsys.readouterr().out
    assert "part=Score" in out
    assert "key=Bb" in out
    assert "no jumbo file" not in out
    assert "missing Bass" not in out
    with pikepdf.Pdf.open(tmp_path / "show.Bass.pdf") as pdf:
        assert len(pdf.pages) == 2


def test_sweet_home_chicago_bb_stays_bb_part_in_combo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(tmp_path / "Charts", ["Sweet Home Chicago.Bb.pdf"])
    (tmp_path / "tunery.yaml").write_text("- library: Charts\n")
    layout = write_layout(
        tmp_path / "show.yaml", [{"title": "Sweet Home Chicago"}]
    )

    render(layout, tmp_path / "show.pdf", part=["Bb"], verbose=2)

    out = capsys.readouterr().out
    assert "part=Bb" in out
    assert "key=None" in out
    assert "missing" not in out


def test_collection_passes_type_like_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(tmp_path / "shared" / "Inherited", ["Tune.pdf"])
    create_pdfs(tmp_path / "shared" / "Explicit", ["Tune.pdf"])
    (tmp_path / "shared" / "collection.yaml").write_text(
        "- library: Inherited\n- library: Explicit\n  type: combo\n"
    )
    (tmp_path / "tunery.yaml").write_text(
        "- library: shared/collection.yaml\n  type: big band\n"
    )
    layout = write_layout(tmp_path / "show.yaml", [{"title": "Tune"}])

    render(layout, tmp_path / "show.pdf", part=["Piano"], verbose=2)

    out = capsys.readouterr().out
    # Inherited dir reads big band (untagged Tune is a Score, unreached);
    # the explicit combo dir answers with its Concert chart.
    assert "part=Score" in out
    assert "part=Concert" in out
    assert 'missing Piano for "Tune"' not in out
