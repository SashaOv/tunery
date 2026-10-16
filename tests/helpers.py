"""Shared builders for library fixtures: blank PDFs named like real charts."""

from collections.abc import Iterable
from pathlib import Path

import pikepdf
import yaml


def create_pdf(path: Path, page_count: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pikepdf.Pdf.new()
    for _ in range(page_count):
        pdf.add_blank_page(page_size=(100, 100))
    pdf.save(path)
    pdf.close()
    return path


def create_pdfs(root: Path, names: Iterable[str]) -> None:
    for name in names:
        create_pdf(root / name)


def write_layout(path: Path, records: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(list(records), sort_keys=False))
    return path


def outline_titles(pdf_path: Path) -> list[tuple[str, list[str]]]:
    with pikepdf.Pdf.open(pdf_path) as pdf, pdf.open_outline() as outline:
        return [(item.title, [c.title for c in item.children]) for item in outline.root]
