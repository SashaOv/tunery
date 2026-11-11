from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from helpers import create_pdf, create_pdfs, outline_titles, write_layout

from tunery.render import render
from tunery.settings import Settings

# Each setting's non-default value and how a render shows it took effect, in
# the project `setup` builds: Vault holds Tune's concert and piano charts, the
# layout's folder holds Other.
Effect = Callable[[str, Path], bool]
EFFECTS: dict[str, tuple[bool, Effect]] = {
    "concert-sub": (False, lambda out, book: 'missing Alto Sax for "Tune"' in out),
    "current-dir": (False, lambda out, book: 'not found "Other"' in out),
    "explicit-bookmark-hierarchy": (
        True,
        lambda out, book: outline_titles(book)[0][1][0].startswith("_ "),
    ),
}


def setup(
    root: Path, project: object = None, layout_config: object = None
) -> Path:
    """Write the project; `project` and `layout_config` are the YAML values
    of tunery.yaml and the layout's `config:` (default: Vault, no config)."""
    create_pdfs(root / "Vault", ["Tune/Tune - Concert.pdf", "Tune/Tune - Piano.pdf"])
    create_pdf(root / "gig" / "Other.pdf")
    if project is None:
        project = [{"library": "Vault"}]
    (root / "tunery.yaml").write_text(yaml.safe_dump(project, sort_keys=False))
    records: list[dict] = [{"title": "Tune"}, {"title": "Other"}]
    if layout_config is not None:
        records.insert(0, {"config": layout_config})
    return write_layout(root / "gig" / "gig.yaml", records)


def run(
    layout: Path, capsys: pytest.CaptureFixture[str], cli: Settings | None = None
) -> tuple[str, Path]:
    capsys.readouterr()
    render(
        layout, layout.parent / "book.pdf", part=["Alto Sax,Piano"], verbose=1, settings=cli
    )
    return capsys.readouterr().out, layout.parent / "book.Alto Sax-Piano.pdf"


def place(root: Path, where: str, key: str, value: object) -> Path:
    setting = {key: value}
    if where == "project list":
        return setup(root, project=[{"library": "Vault"}, setting])
    if where == "project mapping":
        return setup(root, project=setting, layout_config=[{"library": "../Vault"}])
    if where == "layout list":
        return setup(root, layout_config=[setting])
    return setup(root, layout_config=setting)


PLACES = ["project list", "project mapping", "layout list", "layout mapping"]


@pytest.mark.parametrize("key", EFFECTS)
def test_defaults_leave_settings_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str
) -> None:
    out, book = run(setup(tmp_path), capsys)

    assert not EFFECTS[key][1](out, book)


@pytest.mark.parametrize("where", PLACES)
@pytest.mark.parametrize("key", EFFECTS)
def test_every_setting_is_a_config_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str, where: str
) -> None:
    value, took_effect = EFFECTS[key]

    out, book = run(place(tmp_path, where, key, value), capsys)

    assert took_effect(out, book)


@pytest.mark.parametrize("key", EFFECTS)
def test_every_setting_is_a_cli_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str
) -> None:
    value, took_effect = EFFECTS[key]

    out, book = run(
        setup(tmp_path), capsys, Settings.model_validate({key: value})
    )

    assert took_effect(out, book)


@pytest.mark.parametrize(
    ("project", "layout", "cli", "concert_sub"),
    [
        (False, None, None, False),
        (False, True, None, True),
        (True, False, None, False),
        (False, None, True, True),
        (None, True, False, False),
        (False, False, True, True),
    ],
)
def test_command_line_beats_layout_beats_project(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    project: bool | None,
    layout: bool | None,
    cli: bool | None,
    concert_sub: bool,
) -> None:
    project_config: list[dict] = [{"library": "Vault"}]
    if project is not None:
        project_config.append({"concert-sub": project})
    layout_path = setup(
        tmp_path,
        project=project_config,
        layout_config=None if layout is None else {"concert-sub": layout},
    )

    out, book = run(
        layout_path, capsys, None if cli is None else Settings(concert_sub=cli)
    )

    assert EFFECTS["concert-sub"][1](out, book) == (not concert_sub)


def test_later_record_in_a_file_wins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = setup(
        tmp_path,
        project=[{"library": "Vault"}, {"concert-sub": False}, {"concert-sub": True}],
    )

    out, book = run(layout, capsys)

    assert not EFFECTS["concert-sub"][1](out, book)


def test_one_record_holds_several_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = setup(
        tmp_path, layout_config={"concert-sub": False, "current-dir": False}
    )

    out, book = run(layout, capsys)

    assert EFFECTS["concert-sub"][1](out, book)
    assert EFFECTS["current-dir"][1](out, book)


@pytest.mark.parametrize("where", PLACES)
@pytest.mark.parametrize("key", ["no-current", "colour"])
def test_unknown_setting_is_an_error(tmp_path: Path, where: str, key: str) -> None:
    layout = place(tmp_path, where, key, True)
    file = tmp_path / "tunery.yaml" if where.startswith("project") else layout

    with pytest.raises(ValueError, match=key) as error:
        render(layout, layout.parent / "book.pdf")
    assert str(file) in str(error.value)


@pytest.mark.parametrize("where", PLACES)
def test_setting_must_be_a_boolean(tmp_path: Path, where: str) -> None:
    layout = place(tmp_path, where, "concert-sub", "maybe")

    with pytest.raises(ValueError, match="concert-sub"):
        render(layout, layout.parent / "book.pdf")


@pytest.mark.parametrize("where", ["project mapping", "layout mapping"])
def test_record_mixing_library_and_setting_is_an_error(
    tmp_path: Path, where: str
) -> None:
    mixed = {"library": "Vault", "concert-sub": False}
    if where == "project mapping":
        layout = setup(tmp_path, project=mixed)
    else:
        layout = setup(tmp_path, layout_config={**mixed, "library": "../Vault"})

    with pytest.raises(ValueError, match="concert-sub"):
        render(layout, layout.parent / "book.pdf")


def test_named_parts_reject_extra_keys(tmp_path: Path) -> None:
    layout = setup(
        tmp_path, layout_config={"part": {"horns": "Alto Sax"}, "concert-sub": False}
    )

    with pytest.raises(ValueError, match="concert-sub"):
        render(layout, layout.parent / "book.pdf")
