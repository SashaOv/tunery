from pathlib import Path

import pytest
from helpers import create_pdfs, outline_titles, write_layout

from tunery.render import render

FOLDER = "Vault/Instrumentals/Si Tu Voir Ma Mere"
STEMS = ["Alto_Saxophone", "Bass_Clarinet", "C_Major"]


def make_show(root: Path, file: str) -> Path:
    create_pdfs(
        root,
        [f"{FOLDER}/si-tu-vois-ma-mere-sidney-bechet-lead-sheet-{s}.pdf" for s in STEMS],
    )
    (root / "tunery.yaml").write_text("[]\n")
    return write_layout(
        root / "Shows" / "gig" / "gig.yaml",
        [{"title": "Si Tu Vois Ma Mare", "file": file}],
    )


def test_score_folder_picks_part_without_title_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = make_show(tmp_path, f"../../{FOLDER}")

    render(layout, tmp_path / "book.pdf", part=["Alto Sax,Piano"], verbose=1)

    out = capsys.readouterr().out
    assert '"Si Tu Vois Ma Mare" Alto Sax: si-tu-vois-ma-mere-sidney-bechet-lead-sheet-Alto_Saxophone.pdf' in out
    assert '"Si Tu Vois Ma Mare" Piano: si-tu-vois-ma-mere-sidney-bechet-lead-sheet-C_Major.pdf' in out
    assert outline_titles(tmp_path / "book.Alto Sax+Piano.pdf") == [
        ("Si Tu Vois Ma Mare", ["Alto Sax", "Lead Sheet"])
    ]


def test_missing_score_folder_is_an_error(tmp_path: Path) -> None:
    layout = make_show(tmp_path, "../../Vault/Instrumentals/Nowhere")

    with pytest.raises(ValueError, match="Score folder not found: .*Vault/Instrumentals/Nowhere"):
        render(layout, tmp_path / "book.pdf", part=["Piano"])
