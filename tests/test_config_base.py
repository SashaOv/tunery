from pathlib import Path

import pikepdf
import pytest
import yaml

from tunery.library import find_project_config, load_project_libraries
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


def test_find_project_config_supports_dotted_name(tmp_path: Path) -> None:
    config = tmp_path / ".tunery.yaml"
    config.write_text("[]\n")

    assert find_project_config(tmp_path) == config


def test_find_project_config_prefers_plain_name(tmp_path: Path) -> None:
    # Only one present at a time is valid; ensure plain is found.
    config = tmp_path / "tunery.yaml"
    config.write_text("[]\n")

    assert find_project_config(tmp_path) == config


def test_find_project_config_rejects_both_names(tmp_path: Path) -> None:
    (tmp_path / "tunery.yaml").write_text("[]\n")
    (tmp_path / ".tunery.yaml").write_text("[]\n")

    with pytest.raises(ValueError, match="Both"):
        find_project_config(tmp_path)


def test_render_absolute_layout_uses_layout_dir_lexically(tmp_path: Path) -> None:
    # project/apps/JBM/tunery.yaml reachable only through logical GDrive symlink.
    project = tmp_path / "project"
    real_gdrive = tmp_path / "real-gdrive"
    gig_real = real_gdrive / "Shows" / "gig"
    gig_real.mkdir(parents=True)
    chart = create_pdf(project / "Vault" / "chart.pdf", 1)
    assert chart.exists()
    (project / "tunery.yaml").write_text("- library: Vault\n")
    linked = project / "GDrive"
    linked.symlink_to(real_gdrive, target_is_directory=True)
    layout_path = write_layout(
        linked / "Shows" / "gig" / "gig.yaml", [{"title": "chart"}]
    )
    # Physical layout dir is under real-gdrive (no config); lexical is under
    # project/GDrive (config in project/). Absolute layout path must use layout dir.
    output = tmp_path / "out.pdf"

    render(Path(str(layout_path)), output)

    with pikepdf.Pdf.open(output) as pdf:
        assert len(pdf.pages) == 1


def test_render_relative_layout_uses_logical_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    real_gdrive = tmp_path / "real-gdrive"
    (real_gdrive / "Shows" / "gig").mkdir(parents=True)
    create_pdf(project / "Vault" / "chart.pdf", 1)
    (project / "tunery.yaml").write_text("- library: Vault\n")
    linked = project / "GDrive"
    linked.symlink_to(real_gdrive, target_is_directory=True)
    gig_dir = linked / "Shows" / "gig"
    write_layout(gig_dir / "gig.yaml", [{"title": "chart"}])

    monkeypatch.chdir(gig_dir)
    monkeypatch.setenv("PWD", str(gig_dir))

    render(Path("gig.yaml"), tmp_path / "out.pdf")

    with pikepdf.Pdf.open(tmp_path / "out.pdf") as pdf:
        assert len(pdf.pages) == 1


def make_linked_project(root: Path) -> Path:
    """proj/tunery.yaml with proj/GDrive -> a separate tree holding the show."""
    project = root / "proj"
    real_gdrive = root / "real-gdrive"
    (real_gdrive / "Shows" / "x").mkdir(parents=True)
    create_pdf(project / "Vault" / "chart.pdf", 1)
    (project / "tunery.yaml").write_text("- library: Vault\n")
    (project / "GDrive").symlink_to(real_gdrive, target_is_directory=True)
    write_layout(project / "GDrive" / "Shows" / "x" / "x.yaml", [{"title": "chart"}])
    return project


def in_dir(monkeypatch: pytest.MonkeyPatch, directory: Path) -> None:
    monkeypatch.chdir(directory)
    monkeypatch.setenv("PWD", str(directory))


def test_relative_layout_from_parent_folder_finds_project_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    make_linked_project(tmp_path)
    create_pdf(tmp_path / "chart.pdf", 3)  # must not be scanned
    in_dir(monkeypatch, tmp_path)

    render(Path("proj/GDrive/Shows/x/x.yaml"), Path("out.pdf"), verbose=1)

    out = capsys.readouterr().out
    assert f"configuration {tmp_path / 'proj' / 'tunery.yaml'}" in out
    assert "Project configuration was not found" not in out
    with pikepdf.Pdf.open(tmp_path / "out.pdf") as pdf:
        assert len(pdf.pages) == 1


def test_base_anchors_relative_command_line_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = make_linked_project(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    in_dir(monkeypatch, elsewhere)

    render(Path("GDrive/Shows/x/x.yaml"), Path("out.pdf"), base=project)

    with pikepdf.Pdf.open(project / "out.pdf") as pdf:
        assert len(pdf.pages) == 1


def test_without_config_only_layout_folder_is_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdf(tmp_path / "chart.pdf", 1)
    write_layout(tmp_path / "show" / "gig.yaml", [{"title": "chart"}])
    in_dir(monkeypatch, tmp_path)

    render(Path("show/gig.yaml"), Path("out.pdf"))

    out = capsys.readouterr().out
    assert "Project configuration was not found" in out
    assert 'not found "chart"' in out


def test_found_shows_paths_through_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "proj"
    create_pdf(tmp_path / "cloud" / "Tune" / "Tune.pdf", 1)
    project.mkdir()
    (project / "Link").symlink_to(tmp_path / "cloud", target_is_directory=True)
    (project / "tunery.yaml").write_text("- library: Link\n")
    write_layout(project / "gig.yaml", [{"title": "Tune"}])
    in_dir(monkeypatch, project)

    render(Path("gig.yaml"), Path("out.pdf"))

    assert f'found    "Tune" in "{project / "Link" / "Tune"}"' in capsys.readouterr().out


def test_load_project_libraries_uses_base_without_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_pdf(tmp_path / "chart.pdf", 1)

    libraries = load_project_libraries(tmp_path)

    assert libraries.lookup_title("chart") is not None
    assert "Project configuration was not found" in capsys.readouterr().out
