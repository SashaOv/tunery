from pathlib import Path

import pytest
from helpers import create_pdf, create_pdfs, write_layout

from tunery.render import render


def setup_project(root: Path) -> Path:
    """A project library and a layout folder holding its own chart of `Tune`."""
    create_pdfs(root / "Vault", ["Tune/Tune - Concert.pdf", "Other/Other.pdf"])
    (root / "tunery.yaml").write_text("- library: Vault\n")
    create_pdf(root / "gig" / "Tune.Eb.pdf")
    return write_layout(root / "gig" / "gig.yaml", [{"title": "Tune"}, {"title": "Other"}])


def run(
    layout: Path, capsys: pytest.CaptureFixture[str], **options: object
) -> str:
    capsys.readouterr()
    render(layout, layout.parent / "book.pdf", part=["Alto Sax"], verbose=1, **options)
    return capsys.readouterr().out


def test_layout_folder_is_searched_before_project_libraries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = run(setup_project(tmp_path), capsys)

    assert '"Tune" Alto Sax: Tune.Eb.pdf' in out
    assert '"Other" Alto Sax: Other.pdf' in out


def test_layout_folder_is_searched_without_a_project_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdf(tmp_path / "Tune.Eb.pdf")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Tune"}])

    out = run(layout, capsys)

    assert '"Tune" Alto Sax: Tune.Eb.pdf' in out


def test_layout_folder_beats_layout_config_libraries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdfs(tmp_path / "extra", ["Tune/Tune - Eb.pdf"])
    create_pdf(tmp_path / "Tune.Eb.pdf")
    layout = write_layout(
        tmp_path / "gig.yaml",
        [{"config": [{"library": "extra"}]}, {"title": "Tune"}],
    )

    out = run(layout, capsys)

    assert '"Tune" Alto Sax: Tune.Eb.pdf' in out


def test_layout_folder_skips_folders_other_libraries_cover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The project's Vault sits below the layout: its charts are the Vault
    # library's, not a second copy under the layout folder.
    create_pdfs(tmp_path / "Vault", ["Tune/Tune - Eb.pdf"])
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n  match: fuzzy\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Tune"}])

    out = run(layout, capsys)

    assert f'found    "Tune" in "{tmp_path / "Vault" / "Tune"}"' in out


@pytest.mark.parametrize(
    "how",
    ["option", "layout config", "project config", "layout config over project"],
)
def test_no_current_skips_the_layout_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], how: str
) -> None:
    layout = setup_project(tmp_path)
    options: dict[str, object] = {}
    if how == "option":
        options["current"] = False
    elif how == "layout config":
        layout = write_layout(
            layout, [{"config": [{"no-current": True}]}, {"title": "Tune"}, {"title": "Other"}]
        )
    elif how == "project config":
        (tmp_path / "tunery.yaml").write_text("- library: Vault\n- no-current: true\n")
    else:
        (tmp_path / "tunery.yaml").write_text("- library: Vault\n- no-current: false\n")
        layout = write_layout(
            layout, [{"config": [{"no-current": True}]}, {"title": "Tune"}, {"title": "Other"}]
        )

    out = run(layout, capsys, **options)

    assert '"Tune" Alto Sax: Tune - Concert.pdf' in out
    assert "Tune.Eb.pdf" not in out


def test_current_option_overrides_configured_no_current(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = setup_project(tmp_path)
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n- no-current: true\n")

    out = run(layout, capsys, current=True)

    assert '"Tune" Alto Sax: Tune.Eb.pdf' in out


def test_no_current_without_configuration_leaves_no_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdf(tmp_path / "Tune.pdf")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Tune"}])

    out = run(layout, capsys, current=False)

    assert 'not found "Tune"' in out


def test_project_config_rejects_unknown_setting(tmp_path: Path) -> None:
    (tmp_path / "tunery.yaml").write_text("- no-current: true\n  colour: red\n")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Tune"}])

    with pytest.raises(ValueError, match="Invalid project configuration"):
        render(layout, tmp_path / "book.pdf")


def test_no_current_must_be_a_boolean(tmp_path: Path) -> None:
    layout = write_layout(
        tmp_path / "gig.yaml", [{"config": [{"no-current": "maybe"}]}, {"title": "Tune"}]
    )

    with pytest.raises(ValueError, match="no-current"):
        render(layout, tmp_path / "book.pdf")
