from pathlib import Path

import pytest
from helpers import create_pdfs, outline_titles, write_layout

from tunery.render import render

SHEET_MUSIC = [
    "Norwegian Wood/Norwegian Wood.pdf",
    "_Petes/Norwegian Wood (in F)/Norwegian Wood - Horn 1.pdf",
    "_Petes/Norwegian Wood (in F)/Norwegian Wood - Horn 2.pdf",
    "_Petes/Norwegian Wood (in F)/Norwegian Wood - Bass.pdf",
    "_Petes/Funk In Deep Freeze/Funk In Deep Freeze - Tenor.pdf",
    "_Petes/Funk In Deep Freeze/Funk In Deep Freeze - Flute.pdf",
    "_Petes/I Love Lucy/I Love Lucy - Tenor Sax.pdf",
    "_Petes/I Love Lucy/I Love Lucy - Trumpet.pdf",
    "I Love Lucy/I Love Lucy - Tenor Sax.v2.pdf",
]


def make_show(root: Path, charts: list[dict], config_key: str = "part") -> Path:
    create_pdfs(root / "SheetMusic", SHEET_MUSIC)
    (root / "tunery.yaml").write_text("- library: SheetMusic\n")
    config = {"config": [{config_key: {"norm": "Tenor Sax", "duo": "Tenor Sax, Trumpet"}}]}
    return write_layout(root / "show.yaml", [config, *charts])


def run(root: Path, layout: Path, part: str, capsys) -> str:
    capsys.readouterr()
    render(layout, root / "show.pdf", part=[part], verbose=1)
    return capsys.readouterr().out


def test_named_part_from_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    layout = make_show(tmp_path, [{"title": "Funk In Deep Freeze"}])

    out = run(tmp_path, layout, "norm", capsys)

    assert (tmp_path / "show.norm.pdf").exists()
    assert '"Funk In Deep Freeze" Tenor Sax: Funk In Deep Freeze - Tenor.pdf' in out


@pytest.mark.parametrize(("part_key", "tag_key"), [("part", "tag"), ("parts", "tags")])
def test_chart_part_mapping_and_tag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], part_key: str, tag_key: str
) -> None:
    layout = make_show(
        tmp_path,
        [
            {"title": "Norwegian Wood", part_key: {"Tenor Sax": "Horn 1", "Trumpet": "Horn 2"}},
            {"title": "Funk In Deep Freeze", part_key: {"Tenor": "Flute"}},
            {"title": "I Love Lucy", tag_key: "v2"},
        ],
        config_key=part_key,
    )

    out = run(tmp_path, layout, "duo", capsys)

    assert '"Norwegian Wood" Horn 1: Norwegian Wood - Horn 1.pdf' in out
    assert '"Funk In Deep Freeze" Flute: Funk In Deep Freeze - Flute.pdf' in out
    assert '"I Love Lucy" Tenor Sax: I Love Lucy - Tenor Sax.v2.pdf' in out
    assert outline_titles(tmp_path / "show.duo.pdf")[0] == (
        "Norwegian Wood",
        ["Horn 1", "Horn 2"],
    )


def test_tag_list_and_missing_tag_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = make_show(tmp_path, [{"title": "I Love Lucy", "tag": ["v3"]}])

    out = run(tmp_path, layout, "norm", capsys)

    assert 'no v3 version of "I Love Lucy"' in out


def test_folder_in_key_selects_score_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = make_show(tmp_path, [{"title": "Norwegian Wood", "key": "F"}])

    out = run(tmp_path, layout, "Bass", capsys)

    assert '"Norwegian Wood" Bass: Norwegian Wood - Bass.pdf' in out
    assert "no F version" not in out


def test_unknown_part_name_is_an_error(tmp_path: Path) -> None:
    layout = make_show(tmp_path, [{"title": "Norwegian Wood"}])

    with pytest.raises(ValueError, match='Unknown part "nrom"'):
        render(layout, tmp_path / "show.pdf", part=["nrom"])
