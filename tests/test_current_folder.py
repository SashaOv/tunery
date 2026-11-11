from pathlib import Path

import pytest
from helpers import create_pdf, create_pdfs, write_layout

from tunery.render import render
from tunery.settings import Settings


def setup_project(root: Path) -> Path:
    """A project library and a layout folder holding its own chart of `Tune`."""
    create_pdfs(root / "Vault", ["Tune/Tune - Concert.pdf", "Other/Other.pdf"])
    (root / "tunery.yaml").write_text("- library: Vault\n")
    create_pdf(root / "gig" / "Tune.Eb.pdf")
    return write_layout(root / "gig" / "gig.yaml", [{"title": "Tune"}, {"title": "Other"}])


def run(
    layout: Path, capsys: pytest.CaptureFixture[str], settings: Settings | None = None
) -> str:
    capsys.readouterr()
    render(
        layout, layout.parent / "book.pdf", part=["Alto Sax"], verbose=1, settings=settings
    )
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
def test_current_dir_off_skips_the_layout_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], how: str
) -> None:
    layout = setup_project(tmp_path)
    cli: Settings | None = None
    if how == "option":
        cli = Settings(current_dir=False)
    elif how == "layout config":
        layout = write_layout(
            layout, [{"config": {"current-dir": False}}, {"title": "Tune"}, {"title": "Other"}]
        )
    elif how == "project config":
        (tmp_path / "tunery.yaml").write_text("- library: Vault\n- current-dir: false\n")
    else:
        (tmp_path / "tunery.yaml").write_text("- library: Vault\n- current-dir: true\n")
        layout = write_layout(
            layout, [{"config": {"current-dir": False}}, {"title": "Tune"}, {"title": "Other"}]
        )

    out = run(layout, capsys, settings=cli)

    assert '"Tune" Alto Sax: Tune - Concert.pdf' in out
    assert "Tune.Eb.pdf" not in out


def test_current_dir_option_overrides_configured_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = setup_project(tmp_path)
    (tmp_path / "tunery.yaml").write_text("- library: Vault\n- current-dir: false\n")

    out = run(layout, capsys, settings=Settings(current_dir=True))

    assert '"Tune" Alto Sax: Tune.Eb.pdf' in out


def test_current_dir_off_without_configuration_leaves_no_library(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdf(tmp_path / "Tune.pdf")
    layout = write_layout(tmp_path / "gig.yaml", [{"title": "Tune"}])

    out = run(layout, capsys, settings=Settings(current_dir=False))

    assert 'not found "Tune"' in out
