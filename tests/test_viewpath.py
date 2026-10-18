"""ViewPath: a Path navigated lexically through links, relative to a base."""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from tunery.viewpath import ViewPath


@dataclass
class World:
    """`project/GDrive` is a link to `real`, like a cloud-storage folder."""

    root: Path
    project: Path
    real: Path

    @property
    def gdrive(self) -> Path:
        return self.project / "GDrive"


def link(path: Path, target: Path) -> None:
    try:
        path.symlink_to(target, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")


def in_dir(monkeypatch: pytest.MonkeyPatch, directory: Path, pwd: str | None = None) -> None:
    monkeypatch.chdir(directory)
    monkeypatch.setenv("PWD", pwd or str(directory))


@pytest.fixture
def world(tmp_path: Path) -> World:
    real = tmp_path / "real"
    (real / "Shows").mkdir(parents=True)
    (real / "Shows" / "a.txt").write_text("hello")
    (real / "cfg").mkdir()
    project = tmp_path / "project"
    project.mkdir()
    link(project / "GDrive", real)
    return World(tmp_path, project, real)


# Lexical navigation


def test_dot_dot_returns_through_the_link_not_to_the_linked_parent(world: World) -> None:
    shows = ViewPath(world.gdrive / "Shows")

    assert shows / ".." == world.gdrive
    assert shows.parent == world.gdrive
    assert [str(p) for p in list(shows.parents)[:2]] == [str(world.gdrive), str(world.project)]
    assert (shows / ".." / "..") == world.project
    assert shows.resolve().parent == world.real  # the physical parent differs


def test_dot_dot_above_the_root_stays_at_the_root() -> None:
    assert ViewPath("/..") == Path("/")
    assert ViewPath("/a/../../b") == Path("/b")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((), "."),
        (("",), "."),
        ((".",), "."),
        (("a/./b/../c",), "a/c"),
        (("a", "..", ".."), ".."),
        (("a", "b", "../c"), "a/c"),
    ],
)
def test_relative_paths_collapse_lexically(args: tuple[str, ...], expected: str) -> None:
    assert ViewPath(*args) == Path(expected)


def test_absolute_right_operand_replaces_the_left() -> None:
    assert ViewPath("/x") / "/abs/y" == Path("/abs/y")
    assert ViewPath("rel", "/abs", "z") == Path("/abs/z")


def test_segments_may_be_str_path_or_viewpath() -> None:
    assert ViewPath(Path("a"), ViewPath("b"), "c") == Path("a/b/c")


# Base


def test_relative_path_joins_the_base(tmp_path: Path) -> None:
    assert ViewPath("gig.yaml", base=tmp_path).absolute() == tmp_path / "gig.yaml"
    assert ViewPath("../gig.yaml", base=tmp_path).absolute() == tmp_path.parent / "gig.yaml"


def test_absolute_path_ignores_the_base(tmp_path: Path) -> None:
    other = tmp_path / "other"

    assert ViewPath(other / "x", base=tmp_path / "base").absolute() == other / "x"
    assert ViewPath("/a/../b", base=tmp_path).absolute() == Path("/b")


def test_relative_base_joins_the_logical_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_dir(monkeypatch, tmp_path)

    assert ViewPath("x", base="sub").absolute() == tmp_path / "sub" / "x"
    assert ViewPath("x", base="sub/../other").absolute() == tmp_path / "other" / "x"
    assert ViewPath("x", base=ViewPath("sub")).base == tmp_path / "sub"


def test_base_defaults_to_the_working_directory_and_is_its_own_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_dir(monkeypatch, tmp_path)

    assert ViewPath("x").base == tmp_path
    base = ViewPath("x", base="b").base
    assert base.base is base
    assert base.is_absolute()


def test_derived_paths_keep_the_base(tmp_path: Path) -> None:
    rel = ViewPath("a/b.pdf", base=tmp_path)

    assert (rel / "c").absolute() == tmp_path / "a" / "b.pdf" / "c"
    assert (rel / "..").absolute() == tmp_path / "a"
    assert rel.parent.absolute() == tmp_path / "a"
    assert rel.with_suffix(".txt").absolute() == tmp_path / "a" / "b.txt"
    assert rel.with_name("z.pdf").absolute() == tmp_path / "a" / "z.pdf"
    assert rel.relative_to("a").absolute() == tmp_path / "b.pdf"
    assert all(p.base == tmp_path for p in rel.parents)
    assert ("c" / rel).base == tmp_path


def test_absolute_returns_an_absolute_path_unchanged(tmp_path: Path) -> None:
    path = ViewPath(tmp_path / "x", base="/elsewhere")

    assert path.absolute() == tmp_path / "x"


# Logical working directory


def test_cwd_keeps_the_link_the_shell_went_through(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_dir(monkeypatch, world.real / "Shows", pwd=str(world.gdrive / "Shows"))

    assert ViewPath.cwd() == world.gdrive / "Shows"
    assert ViewPath("x").absolute() == world.gdrive / "Shows" / "x"
    assert os.getcwd() == str(world.real / "Shows")


@pytest.mark.parametrize(
    "pwd",
    [
        None,  # unset
        "",  # empty
        ".",  # relative
        "relative/dir",
        "{missing}",  # names nothing
        "{other}",  # names another directory (stale)
        "{gdrive}/../real/Shows",  # lexically a path that doesn't exist
    ],
)
def test_cwd_falls_back_to_the_physical_directory_when_pwd_is_unusable(
    world: World, monkeypatch: pytest.MonkeyPatch, pwd: str | None
) -> None:
    monkeypatch.chdir(world.real / "Shows")
    if pwd is None:
        monkeypatch.delenv("PWD", raising=False)
    else:
        monkeypatch.setenv(
            "PWD",
            pwd.format(
                missing=world.root / "missing", other=world.project, gdrive=world.gdrive
            ),
        )

    assert ViewPath.cwd() == Path(os.getcwd())
    assert ViewPath.cwd().is_absolute()


# I/O goes through the base and the links


def test_relative_path_reads_from_the_base_not_the_process_directory(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_dir(monkeypatch, world.project)  # nothing named a.txt here
    path = ViewPath("a.txt", base=world.real / "Shows")

    assert path.exists() and path.is_file()
    assert path.read_text() == "hello"
    assert path.stat().st_size == 5
    assert ViewPath(".", base=world.real / "Shows").is_dir()
    assert str(path) == "a.txt"  # messages show the path as given
    assert os.fspath(path) == str(world.real / "Shows" / "a.txt")


def test_io_through_a_link(world: World) -> None:
    path = ViewPath(world.gdrive / "Shows" / "a.txt")

    assert path.read_text() == "hello"
    assert str(path) == str(world.gdrive / "Shows" / "a.txt")


def test_missing_file_is_reported_not_raised_by_queries(world: World) -> None:
    path = ViewPath("nope.txt", base=world.real)

    assert not path.exists() and not path.is_file() and not path.is_dir()
    with pytest.raises(FileNotFoundError, match="nope.txt"):
        path.read_text()


def test_children_of_a_relative_directory_stay_relative_and_in_the_view(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_dir(monkeypatch, world.project)
    shows = ViewPath("Shows", base=world.real)

    [child] = shows.iterdir()
    [globbed] = shows.glob("*.txt")

    for found in (child, globbed):
        assert isinstance(found, ViewPath)
        assert found == Path("Shows/a.txt")
        assert found.absolute() == world.real / "Shows" / "a.txt"
        assert found.read_text() == "hello"


# resolve leaves the view


def test_resolve_gives_a_plain_physical_path(world: World) -> None:
    resolved = ViewPath(world.gdrive / "Shows" / "a.txt").resolve()

    assert resolved == world.real / "Shows" / "a.txt"
    assert not isinstance(resolved, ViewPath)


def test_resolve_a_relative_path_uses_the_base(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_dir(monkeypatch, world.root)

    assert ViewPath("GDrive/Shows", base=world.project).resolve() == world.real / "Shows"


def test_resolve_strict_reports_a_missing_path(world: World) -> None:
    missing = ViewPath("nope", base=world.real)

    assert missing.resolve() == world.real / "nope"
    with pytest.raises(FileNotFoundError):
        missing.resolve(strict=True)


# relative_base


def test_relative_base_of_a_plain_file_is_its_parent(world: World) -> None:
    anchor = ViewPath(world.real / "Shows" / "a.txt").relative_base()

    assert anchor == world.real / "Shows"
    assert isinstance(anchor, ViewPath)


def test_relative_base_of_a_file_in_a_linked_folder_stays_in_the_view(world: World) -> None:
    anchor = ViewPath(world.gdrive / "Shows" / "a.txt").relative_base()

    assert anchor == world.gdrive / "Shows"
    assert str(anchor) == str(world.gdrive / "Shows")


def test_relative_base_of_a_symlinked_file_is_the_real_parent(world: World) -> None:
    (world.real / "cfg" / "tunery.yaml").write_text("[]")
    link(world.project / "tunery.yaml", world.real / "cfg" / "tunery.yaml")

    assert ViewPath(world.project / "tunery.yaml").relative_base() == world.real / "cfg"
    # even when the symlink itself sits in a linked folder
    link(world.real / "Shows" / "alias.yaml", world.real / "cfg" / "tunery.yaml")
    assert ViewPath(world.gdrive / "Shows" / "alias.yaml").relative_base() == world.real / "cfg"


def test_relative_base_of_a_relative_file_uses_the_base(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_dir(monkeypatch, world.root)
    (world.real / "cfg" / "tunery.yaml").write_text("[]")
    link(world.project / "tunery.yaml", world.real / "cfg" / "tunery.yaml")

    assert ViewPath("tunery.yaml", base=world.project).relative_base() == world.real / "cfg"
    viewed = ViewPath("GDrive/Shows/a.txt", base=world.project).relative_base()
    assert viewed.absolute() == world.gdrive / "Shows"


def test_relative_base_of_a_dangling_symlink_or_missing_file_does_not_raise(
    world: World,
) -> None:
    link(world.project / "dangling.yaml", world.root / "gone" / "x.yaml")

    assert ViewPath(world.project / "dangling.yaml").relative_base() == world.root / "gone"
    assert ViewPath(world.project / "missing.yaml").relative_base() == world.project


def test_relative_base_keeps_the_base(world: World) -> None:
    anchor = ViewPath("a.txt", base=world.real / "Shows").relative_base()

    assert anchor.base == world.real / "Shows"


# walk


def tree(world: World) -> Path:
    top = world.real / "Tree"
    for name in ("b/y.pdf", "b/x.pdf", "a/z.pdf", "a/deep/d.pdf", "top.pdf"):
        (top / name).parent.mkdir(parents=True, exist_ok=True)
        (top / name).touch()
    return top


def walked(root: ViewPath, **kwargs) -> list[tuple[str, list[str], list[str]]]:
    return [
        (str(dirpath.relative_to(root)), dirnames, filenames)
        for dirpath, dirnames, filenames in root.walk(**kwargs)
    ]


def test_walk_lists_in_name_order_top_down(world: World) -> None:
    assert walked(ViewPath(tree(world))) == [
        (".", ["a", "b"], ["top.pdf"]),
        ("a", ["deep"], ["z.pdf"]),
        ("a/deep", [], ["d.pdf"]),
        ("b", [], ["x.pdf", "y.pdf"]),
    ]


def test_walk_bottom_up_visits_children_first(world: World) -> None:
    order = [dirpath for dirpath, _, _ in walked(ViewPath(tree(world)), top_down=False)]

    assert order == ["a/deep", "a", "b", "."]


def test_walk_prunes_directories_removed_from_the_list(world: World) -> None:
    visited = []
    for dirpath, dirnames, _ in ViewPath(tree(world)).walk():
        visited.append(dirpath.name)
        dirnames[:] = [name for name in dirnames if name != "a"]

    assert visited == ["Tree", "b"]


def test_walk_yields_view_paths_that_keep_the_base(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    tree(world)
    in_dir(monkeypatch, world.root)

    found = [
        dirpath / name
        for dirpath, _, filenames in ViewPath("Tree", base=world.real).walk()
        for name in filenames
    ]

    assert sorted(str(p) for p in found) == [
        "Tree/a/deep/d.pdf",
        "Tree/a/z.pdf",
        "Tree/b/x.pdf",
        "Tree/b/y.pdf",
        "Tree/top.pdf",
    ]
    assert all(isinstance(p, ViewPath) and p.is_file() for p in found)


def test_walk_follows_links_and_reports_view_paths(world: World) -> None:
    tree(world)
    shows = world.real / "Shows"
    link(shows / "Tree", world.real / "Tree")

    paths = {
        str(dirpath / name)
        for dirpath, _, filenames in ViewPath(world.gdrive / "Shows").walk(follow_symlinks=True)
        for name in filenames
    }

    assert str(world.gdrive / "Shows" / "Tree" / "a" / "z.pdf") in paths
    assert str(world.gdrive / "Shows" / "a.txt") in paths


def test_walk_without_follow_lists_a_link_but_does_not_enter_it(world: World) -> None:
    tree(world)
    link(world.real / "Shows" / "Tree", world.real / "Tree")

    result = walked(ViewPath(world.real / "Shows"))

    assert result == [(".", ["Tree"], ["a.txt"])]


def test_walk_enters_a_root_that_is_itself_a_link(world: World) -> None:
    result = walked(ViewPath(world.gdrive), follow_symlinks=False)

    assert [dirpath for dirpath, _, _ in result] == [".", "Shows", "cfg"]


def test_walk_ends_on_a_link_back_to_an_ancestor(world: World) -> None:
    tree(world)
    link(world.real / "Tree" / "a" / "up", world.real / "Tree")
    link(world.real / "Tree" / "a" / "deep" / "self", world.real / "Tree" / "a" / "deep")

    result = walked(ViewPath(world.real / "Tree"), follow_symlinks=True)

    assert [dirpath for dirpath, _, _ in result] == [".", "a", "a/deep", "b"]


def test_walk_enters_a_directory_reached_by_two_links_once(world: World) -> None:
    (world.real / "shared").mkdir()
    (world.real / "shared" / "s.pdf").touch()
    link(world.real / "cfg" / "first", world.real / "shared")
    link(world.real / "cfg" / "second", world.real / "shared")

    result = walked(ViewPath(world.real / "cfg"), follow_symlinks=True)

    assert [(d, f) for d, _, f in result] == [(".", []), ("first", ["s.pdf"])]


def test_walk_follow_never_enters_the_real_root_a_second_time(world: World) -> None:
    link(world.real / "Shows" / "again", world.real / "Shows")

    result = walked(ViewPath(world.gdrive / "Shows"), follow_symlinks=True)

    assert [d for d, _, _ in result] == ["."]


def test_walk_lists_broken_and_looping_symlinks_as_files(world: World) -> None:
    folder = world.real / "Odd"
    folder.mkdir()
    link(folder / "broken", world.root / "gone")
    link(folder / "loop", folder / "loop")

    assert walked(ViewPath(folder), follow_symlinks=True) == [(".", [], ["broken", "loop"])]


@pytest.mark.parametrize("name", ["missing", "a.txt"])
def test_walk_reports_a_root_it_cannot_list(world: World, name: str) -> None:
    errors: list[OSError] = []
    root = ViewPath(world.real / "Shows" / name)

    assert list(root.walk(on_error=errors.append)) == []
    assert len(errors) == 1 and isinstance(errors[0], OSError)
    assert list(root.walk()) == []  # no handler: skipped silently


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="permissions are not enforced",
)
def test_walk_skips_an_unreadable_directory_and_goes_on(world: World) -> None:
    top = tree(world)
    (top / "a").chmod(0)
    errors: list[OSError] = []
    try:
        result = walked(ViewPath(top), on_error=errors.append)
    finally:
        (top / "a").chmod(0o755)

    assert [d for d, _, _ in result] == [".", "b"]
    assert len(errors) == 1 and isinstance(errors[0], PermissionError)
