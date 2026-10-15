from pathlib import Path

import pikepdf

from tunery.library import DirectoryLibrary


def create_pdf(path: Path, page_count: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pikepdf.Pdf.new()
    for _ in range(page_count):
        pdf.add_blank_page(page_size=(100, 100))
    pdf.save(path)
    pdf.close()
    return path


def test_recursive_discovers_nested_pdf(tmp_path: Path) -> None:
    root = tmp_path
    nested = create_pdf(root / "Category" / "Title" / "slug-Piano.pdf")
    lib = DirectoryLibrary(root)

    assert lib.lookup_title("slug-Piano") is not None
    assert lib.lookup_title("slug-Piano").source == nested


def test_archive_excluded(tmp_path: Path) -> None:
    root = tmp_path
    create_pdf(root / "_archive" / "Old Tune.pdf")
    create_pdf(root / "New Tune.pdf")
    lib = DirectoryLibrary(root)

    assert lib.lookup_title("Old Tune") is None
    assert lib.lookup_title("New Tune") is not None


def test_transparent_folder_ignored(tmp_path: Path) -> None:
    root = tmp_path
    nested = create_pdf(root / "_Petes" / "My Tune" / "My Tune - Piano.pdf")
    lib = DirectoryLibrary(root)

    result = lib.lookup_title("My Tune - Piano")
    assert result is not None
    assert result.source == nested
    # Folder name is a name text too.
    assert lib.lookup_title("My Tune") is not None


def test_symlink_no_cycle(tmp_path: Path) -> None:
    root = tmp_path
    create_pdf(root / "real" / "Tune.pdf")
    (root / "link").symlink_to(root / "real", target_is_directory=True)
    # Cycle: real/cycle -> root
    (root / "real" / "cycle").symlink_to(root, target_is_directory=True)
    lib = DirectoryLibrary(root)

    assert lib.lookup_title("Tune") is not None


def test_innermost_folder_is_name_text(tmp_path: Path) -> None:
    root = tmp_path
    nested = create_pdf(root / "Category" / "Blue Monk" / "slug-Piano.pdf")
    lib = DirectoryLibrary(root)

    result = lib.lookup_title("Blue Monk")
    assert result is not None
    assert result.source == nested
