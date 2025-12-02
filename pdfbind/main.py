import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pikepdf
import yaml


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
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to the output PDF file (default: <input_base>.pdf)",
    )
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
            assert end_page is not None, (
                "end_page should be set when start_page is specified"
            )
            # end_page is 1-based and inclusive, so end_idx should be end_page (exclusive in slice)
            end_idx = end_page
            verify(
                end_idx <= total_pages,
                f"page {start_page} + length {length if length else 1} exceeds total pages {total_pages}",
            )
            verify(
                start_idx < total_pages,
                f"page {start_page} exceeds total pages {total_pages}",
            )
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


def escape_pdf_string(text: str) -> str:
    """Escape special characters for PDF string literals."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


# Approximate character widths for Helvetica (in 1/1000 of font size)
# This is a simplified table; full AFM data would be more accurate
HELVETICA_WIDTHS: dict[str, int] = {
    " ": 278,
    "!": 278,
    '"': 355,
    "#": 556,
    "$": 556,
    "%": 889,
    "&": 667,
    "'": 191,
    "(": 333,
    ")": 333,
    "*": 389,
    "+": 584,
    ",": 278,
    "-": 333,
    ".": 278,
    "/": 278,
    "0": 556,
    "1": 556,
    "2": 556,
    "3": 556,
    "4": 556,
    "5": 556,
    "6": 556,
    "7": 556,
    "8": 556,
    "9": 556,
    ":": 278,
    ";": 278,
    "<": 584,
    "=": 584,
    ">": 584,
    "?": 556,
    "@": 1015,
    "A": 667,
    "B": 667,
    "C": 722,
    "D": 722,
    "E": 667,
    "F": 611,
    "G": 778,
    "H": 722,
    "I": 278,
    "J": 500,
    "K": 667,
    "L": 556,
    "M": 833,
    "N": 722,
    "O": 778,
    "P": 667,
    "Q": 778,
    "R": 722,
    "S": 667,
    "T": 611,
    "U": 722,
    "V": 667,
    "W": 944,
    "X": 667,
    "Y": 667,
    "Z": 611,
    "[": 278,
    "\\": 278,
    "]": 278,
    "^": 469,
    "_": 556,
    "`": 333,
    "a": 556,
    "b": 556,
    "c": 500,
    "d": 556,
    "e": 556,
    "f": 278,
    "g": 556,
    "h": 556,
    "i": 222,
    "j": 222,
    "k": 500,
    "l": 222,
    "m": 833,
    "n": 556,
    "o": 556,
    "p": 556,
    "q": 556,
    "r": 333,
    "s": 500,
    "t": 278,
    "u": 556,
    "v": 500,
    "w": 722,
    "x": 500,
    "y": 500,
    "z": 500,
    "{": 334,
    "|": 260,
    "}": 334,
    "~": 584,
}
DEFAULT_CHAR_WIDTH = 556  # Average width for unknown characters


def measure_text_width(text: str, font_size: float) -> float:
    """Calculate the approximate width of text in points using Helvetica metrics."""
    width_units = sum(HELVETICA_WIDTHS.get(c, DEFAULT_CHAR_WIDTH) for c in text)
    return width_units * font_size / 1000


def wrap_text(text: str, max_width: float, font_size: float) -> list[str]:
    """
    Wrap text to fit within max_width, breaking at word boundaries.
    Returns a list of lines.
    """
    words = text.split()
    lines: list[str] = []
    current_line: list[str] = []
    current_width = 0.0
    space_width = measure_text_width(" ", font_size)

    for word in words:
        word_width = measure_text_width(word, font_size)

        if current_line:
            # Check if adding this word (with space) exceeds max width
            if current_width + space_width + word_width <= max_width:
                current_line.append(word)
                current_width += space_width + word_width
            else:
                # Start a new line
                lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width
        else:
            # First word on the line
            current_line.append(word)
            current_width = word_width

    # Don't forget the last line
    if current_line:
        lines.append(" ".join(current_line))

    return lines


def calculate_notes_height(
    notes: str,
    page_width: float,
    font_size: float = 10.0,
    margin: float = 36.0,
    padding: float = 12.0,
) -> tuple[list[str], float]:
    """
    Calculate the required height for notes based on text wrapping.

    Returns:
        A tuple of (wrapped_lines, notes_height).
    """
    available_width = page_width - (2 * margin)
    lines = wrap_text(notes, available_width, font_size)

    # Line height is typically 1.2x font size
    line_height = font_size * 1.2
    text_height = len(lines) * line_height

    # Add padding above and below
    notes_height = text_height + (2 * padding)

    return lines, notes_height


def add_page_with_notes(
    combined_pdf: pikepdf.Pdf,
    source_page: pikepdf.Page,
    notes: str,
    font_size: float = 10.0,
    margin: float = 36.0,
) -> None:
    """
    Add a page to combined_pdf with the source_page content scaled to make room
    for notes at the bottom.

    Args:
        combined_pdf: The destination PDF to add the page to.
        source_page: The source page to copy and scale.
        notes: The text to add at the bottom of the page.
        font_size: Font size for notes text (default 10pt).
        margin: Left margin for notes text (default 36 = 0.5 inch).
    """
    mediabox = pikepdf.Rectangle(source_page.mediabox)

    # Calculate notes height based on wrapped text
    lines, notes_height = calculate_notes_height(
        notes, mediabox.width, font_size, margin
    )

    # Create a new page with same dimensions
    new_page_dict = pikepdf.Dictionary(
        Type=pikepdf.Name.Page,
        MediaBox=source_page.mediabox,
        Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary()),
    )
    new_page = pikepdf.Page(combined_pdf.make_indirect(new_page_dict))

    # Convert original page to form xobject and copy to destination
    form_xobj = source_page.as_form_xobject()
    form_xobj_copy = combined_pdf.copy_foreign(form_xobj)

    # Add the form xobject as a resource
    xobj_name = new_page.add_resource(form_xobj_copy, pikepdf.Name.XObject)

    # Calculate scaling to leave room for notes at bottom
    scale_factor = (mediabox.height - notes_height) / mediabox.height

    # Draw the scaled form xobject, translated up to make room for notes
    # Transformation matrix: a b c d e f where a=scale_x, d=scale_y, e=tx, f=ty
    content = f"""
q
{scale_factor} 0 0 {scale_factor} 0 {notes_height} cm
{xobj_name} Do
Q
""".encode()
    new_page.contents_add(combined_pdf.make_stream(content))

    # Add a standard PDF font for the notes
    font_dict = pikepdf.Dictionary(
        Type=pikepdf.Name.Font,
        Subtype=pikepdf.Name.Type1,
        BaseFont=pikepdf.Name.Helvetica,
    )
    new_page.Resources.Font.F1 = combined_pdf.make_indirect(font_dict)

    # Calculate line height and starting Y position
    line_height = font_size * 1.2
    total_text_height = len(lines) * line_height
    start_y = (notes_height + total_text_height) / 2 - font_size

    # Build the text content with multiple lines
    # TL sets leading (line height), T* moves to next line
    text_ops: list[str] = [
        "q",
        "BT",
        f"/F1 {font_size} Tf",
        f"{line_height} TL",  # Set leading (line spacing)
        f"{margin} {start_y} Td",
    ]

    for line in lines:
        line_escaped = escape_pdf_string(line)
        text_ops.append(f"({line_escaped}) Tj T*")

    text_ops.extend(["ET", "Q"])

    notes_content = "\n".join(text_ops).encode()
    new_page.contents_add(combined_pdf.make_stream(notes_content))

    combined_pdf.pages.append(new_page)


def extract_pages_with_notes(
    combined_pdf: pikepdf.Pdf,
    input_pdf_path: str,
    notes: str,
    start_page: int | None = None,
    length: int | None = None,
) -> int:
    """
    Extract pages from a PDF and add them to the combined PDF with notes overlay.
    Each page is scaled down to make room for the notes at the bottom.
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
            end_page = start_page + length - 1
        else:
            end_page = start_page

    with pikepdf.Pdf.open(input_pdf_path) as input_pdf:
        total_pages = len(input_pdf.pages)

        # Determine which pages to extract
        if start_page is not None:
            start_idx = start_page - 1
            assert end_page is not None
            end_idx = end_page
            verify(
                end_idx <= total_pages,
                f"page {start_page} + length {length if length else 1} exceeds total pages {total_pages}",
            )
            verify(
                start_idx < total_pages,
                f"page {start_page} exceeds total pages {total_pages}",
            )
            verify(start_idx >= 0, f"page {start_page} must be >= 1")
        else:
            start_idx = 0
            end_idx = total_pages

        # Process each page individually with notes
        pages_added = 0
        for page_idx in range(start_idx, end_idx):
            source_page = input_pdf.pages[page_idx]
            add_page_with_notes(combined_pdf, source_page, notes)
            pages_added += 1

    return pages_added


def process_pdf_entry(
    entry: dict[str, Any], yaml_dir: Path, combined_pdf: pikepdf.Pdf
) -> tuple[str, int]:
    """
    Add the requested pages for an entry to the combined PDF and return its title and start page index.

    If the entry contains a 'notes' key, the page content will be scaled down
    and the notes will be added at the bottom of the page.
    """
    input_pdf_path = resolve_path(entry["file"], yaml_dir)
    title = entry.get("title", input_pdf_path.stem)
    start_page = entry.get("page")
    length = entry.get("length")
    notes = entry.get("notes")

    entry_page = len(combined_pdf.pages)

    if notes:
        # When notes are present, we need to process pages individually
        # to add the notes overlay to each page
        extract_pages_with_notes(
            combined_pdf, str(input_pdf_path), notes, start_page, length
        )
    else:
        # No notes - use the standard bulk extraction
        extract_pages_from_pdf(combined_pdf, str(input_pdf_path), start_page, length)

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
                pdf_outline.root.append(pikepdf.OutlineItem(entry.title, entry.page))


def bind_pdf(layout: Path, output: Path):
    with open(str(layout), "r") as file:
        records = yaml.safe_load(file)

    # Get the directory of the YAML file for resolving relative paths
    yaml_dir = layout.parent.resolve()

    combined_pdf = pikepdf.Pdf.new()

    toc: list[OutlineEntry] = []

    for record in records:
        # Check if this is a section (hierarchical) or a file (flat)
        if "section" in record:
            # This is a section with nested items
            section_name = record["section"]
            # Get the body (list of items) for this section
            # Handle None explicitly to ensure we always have a list
            items = record.get("body") or []

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
        elif "file" in record:
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
    output_path = (
        Path(parsed.output)
        if parsed.output
        else input_path.with_name(f"{input_path.stem}.pdf")
    )
    bind_pdf(input_path, output_path)


if __name__ == "__main__":
    main()
