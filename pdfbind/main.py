import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import yaml
import pikepdf


@dataclass
class OutlineChild:
    title: str
    page: int


@dataclass
class SectionOutlineEntry:
    title: str
    page: int
    children: list[OutlineChild] = field(default_factory=list)


@dataclass
class FileOutlineEntry:
    title: str
    page: int


OutlineEntry = SectionOutlineEntry | FileOutlineEntry

def verify(condition: bool, message: str):
    if not condition:
        raise ValueError(message)

def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine PDFs based on YAML input")
    parser.add_argument("layout", help="Path to the input YAML file")
    parser.add_argument("-o", "--output", default=None, help="Path to the output PDF file (default: <input_base>.pdf)")
    return parser.parse_args(args)

def extract_pages_from_pdf(
    combined_pdf: pikepdf.Pdf,
    input_pdf_path: str,
    start_page: int | None = None,
    length: int | None = None,
) -> int:
    """
    Extract pages from a PDF and add them to the combined PDF.
    Returns the number of pages added.
    """
    # Validate page and length if provided
    if start_page is not None:
        verify(type(start_page) is int, "page must be an integer")
        verify(start_page >= 1, "page must be >= 1")
    if length is not None:
        verify(type(length) is int, "length must be an integer")
        verify(length >= 1, "length must be >= 1")
    
    # Calculate end_page from start_page and length
    end_page = None
    if start_page is not None:
        if length is not None:
            end_page = start_page + length - 1  # -1 because start_page is inclusive
        else:
            # If only page is specified, extract just that single page
            end_page = start_page

    with pikepdf.Pdf.open(input_pdf_path) as input_pdf:
        total_pages = len(input_pdf.pages)
        
        # Determine which pages to extract
        if start_page is not None:
            # Convert to 0-based indexing (pikepdf uses 0-based)
            start_idx = start_page - 1
            # end_page is guaranteed to be set when start_page is not None
            assert end_page is not None, "end_page should be set when start_page is specified"
            # end_page is 1-based and inclusive, so end_idx should be end_page (exclusive in slice)
            end_idx = end_page
            verify(end_idx <= total_pages, f"page {start_page} + length {length if length else 1} exceeds total pages {total_pages}")
            verify(start_idx < total_pages, f"page {start_page} exceeds total pages {total_pages}")
            verify(start_idx >= 0, f"page {start_page} must be >= 1")
            
            # Extract the specified page range (start_idx inclusive, end_idx exclusive)
            pages_to_add = input_pdf.pages[start_idx:end_idx]
            combined_pdf.pages.extend(pages_to_add)
            pages_added = len(pages_to_add)
        else:
            # No page range specified, add all pages
            combined_pdf.pages.extend(input_pdf.pages)
            pages_added = len(input_pdf.pages)
    
    return pages_added


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """
    Resolve a file path relative to base_dir if it's relative, otherwise return as-is.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def process_pdf_entry(
    entry: dict[str, Any], yaml_dir: Path, combined_pdf: pikepdf.Pdf
) -> tuple[str, int]:
    """
    Add the requested pages for an entry to the combined PDF and return its title and start page index.
    """
    input_pdf_path = resolve_path(entry["file"], yaml_dir)
    title = entry.get("title", input_pdf_path.stem)
    start_page = entry.get("page")
    length = entry.get("length")

    pages_added = extract_pages_from_pdf(
        combined_pdf, str(input_pdf_path), start_page, length
    )
    entry_page = len(combined_pdf.pages) - pages_added
    return title, entry_page


def add_outline_entries(
    combined_pdf: pikepdf.Pdf, toc_entries: Sequence[OutlineEntry]
) -> None:
    """
    Populate the PDF outline from the collected table-of-contents entries.
    """
    with combined_pdf.open_outline() as pdf_outline:
        for entry in toc_entries:
            if isinstance(entry, SectionOutlineEntry):
                section_outline = pikepdf.OutlineItem(entry.title, entry.page)
                for child in entry.children:
                    section_outline.children.append(
                        pikepdf.OutlineItem(child.title, child.page)
                    )
                pdf_outline.root.append(section_outline)
            else:
                pdf_outline.root.append(
                    pikepdf.OutlineItem(entry.title, entry.page)
                )


def bind_pdf(layout: Path, output: Path):
    with open(str(layout), 'r') as file:
        records = yaml.safe_load(file)

    # Get the directory of the YAML file for resolving relative paths
    yaml_dir = layout.parent.resolve()

    combined_pdf = pikepdf.Pdf.new()
    
    toc: list[OutlineEntry] = []

    for record in records:
        # Check if this is a section (hierarchical) or a file (flat)
        if 'section' in record:
            # This is a section with nested items
            section_name = record['section']
            # Get the body (list of items) for this section
            # Handle None explicitly to ensure we always have a list
            items = record.get('body') or []
            
            # Process all items in this section
            section_children_data: list[OutlineChild] = []
            first_item_page = None
            for item in items:
                item_title, item_page = process_pdf_entry(item, yaml_dir, combined_pdf)
                if first_item_page is None:
                    first_item_page = item_page
                section_children_data.append(OutlineChild(item_title, item_page))
            
            # Store section data (create section even if empty)
            toc.append(
                SectionOutlineEntry(
                    title=section_name,
                    page=first_item_page if first_item_page is not None else 0,
                    children=section_children_data,
                )
            )
        elif 'file' in record:
            # This is a flat file entry
            title, page = process_pdf_entry(record, yaml_dir, combined_pdf)
            
            # Store outline item data (page index is 0-based)
            toc.append(FileOutlineEntry(title=title, page=page))
        else:
            raise ValueError("Record must have either 'section' or 'file' key")
    
    # Add the outline to the combined PDF
    add_outline_entries(combined_pdf, toc)

    # Save the combined PDF
    combined_pdf.save(str(output))
    combined_pdf.close()



def main(args: Sequence[str] | None = None):
    parsed = parse_args(args)
    input_path = Path(parsed.layout)
    output_path = Path(parsed.output) if parsed.output else input_path.with_name(f"{input_path.stem}.pdf")  
    bind_pdf(input_path, output_path)


if __name__ == "__main__":
    main()
