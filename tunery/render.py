"""PDF rendering logic for combining PDFs with table of contents."""

from pathlib import Path
from typing import List

import pikepdf
import yaml
from pydantic import BaseModel, Field, PositiveInt, RootModel

from tunery.index import Index


class FileEntry(BaseModel):
    """A file entry specifying a PDF source and optional page range/notes.

    Either 'file' or 'title' must be provided. If only 'title' is provided,
    the file will be looked up in the index.
    """

    file: str | None = None
    title: str | None = None
    page: PositiveInt | None = None
    length: PositiveInt | None = None
    notes: str | None = None

    def model_post_init(self, __context) -> None:
        if not self.file and not self.title:
            raise ValueError("Either 'file' or 'title' must be provided")


class SectionEntry(BaseModel):
    """A section containing nested file entries."""

    section: str
    body: List[FileEntry] = Field(default_factory=list)


class Layout(RootModel[List[SectionEntry | FileEntry]]):
    """The complete layout schema - a list of sections and/or file entries."""

    pass


def get_page_range(
    total_pages: int,
    start_page: int | None = None,
    length: int | None = None,
) -> tuple[int, int]:
    """
    Calculate 0-based page range indices from 1-based start_page and length.

    Args:
        total_pages: Total number of pages in the PDF.
        start_page: 1-based starting page number (optional).
        length: Number of pages to extract (optional, defaults to 1 if start_page is set).

    Returns:
        Tuple of (start_idx, end_idx) for slicing (end_idx is exclusive).
    """
    if start_page is not None:
        start_idx = start_page - 1
        end_idx = start_page + (length - 1 if length else 0)
        if end_idx > total_pages:
            raise ValueError(
                f"page {start_page} + length {length or 1} exceeds total pages {total_pages}"
            )
        return start_idx, end_idx
    return 0, total_pages


def copy_pages(
    combined_pdf: pikepdf.Pdf,
    input_pdf_path: str,
    start_page: int | None = None,
    length: int | None = None,
) -> int:
    """
    Copy pages from a source PDF to the combined PDF.
    Returns the number of pages added.
    """
    with pikepdf.Pdf.open(input_pdf_path) as input_pdf:
        start_idx, end_idx = get_page_range(len(input_pdf.pages), start_page, length)
        pages_to_add = input_pdf.pages[start_idx:end_idx]
        combined_pdf.pages.extend(pages_to_add)
        return len(pages_to_add)


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


# Helvetica character widths (in 1000ths of a point at 1pt size)
# This is a simplified version - only includes common ASCII characters
HELVETICA_WIDTHS = {
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


def measure_text_width(text: str, font_size: float) -> float:
    """Measure the width of text in Helvetica font at given size."""
    width = 0.0
    for char in text:
        # Use width if available, otherwise estimate with average
        width += HELVETICA_WIDTHS.get(char, 556)
    # Convert from 1000ths of a point to actual points and scale by font size
    return width * font_size / 1000.0


def wrap_text(text: str, max_width: float, font_size: float, margin: float) -> list[str]:
    """
    Wrap text to fit within max_width, accounting for margins.

    Args:
        text: Text to wrap.
        max_width: Maximum width available for text (page width).
        font_size: Font size in points.
        margin: Left/right margin in points.

    Returns:
        List of wrapped lines.
    """
    available_width = max_width - 2 * margin
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        if measure_text_width(test_line, font_size) <= available_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                # Single word is too long, add it anyway
                lines.append(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def calculate_notes_height(
    notes: str, page_width: float, font_size: float, margin: float
) -> tuple[list[str], float]:
    """
    Calculate the height needed for notes text.

    Returns:
        Tuple of (wrapped_lines, total_height).
    """
    lines = wrap_text(notes, page_width, font_size, margin)
    line_height = font_size * 1.2
    # Add some padding above and below
    total_height = len(lines) * line_height + 2 * font_size
    return lines, total_height


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


def copy_pages_with_notes(
    combined_pdf: pikepdf.Pdf,
    input_pdf_path: str,
    notes: str,
    start_page: int | None = None,
    length: int | None = None,
) -> int:
    """
    Copy pages from a source PDF to the combined PDF, adding notes to each page.
    Returns the number of pages added.
    """
    with pikepdf.Pdf.open(input_pdf_path) as input_pdf:
        start_idx, end_idx = get_page_range(len(input_pdf.pages), start_page, length)

        for page_idx in range(start_idx, end_idx):
            add_page_with_notes(combined_pdf, input_pdf.pages[page_idx], notes)

        return end_idx - start_idx


def process_file_entry(
    entry: FileEntry,
    default_dir: Path,
    combined_pdf: pikepdf.Pdf,
    index: Index | None = None,
) -> tuple[str, int]:
    """
    Add the requested pages for an entry to the combined PDF and return its title and start page index.

    If the entry contains notes, the page content will be scaled down
    and the notes will be added at the bottom of the page.

    If the entry has no 'file' but has a 'title', the file will be looked up in the index.
    """
    # Determine the input PDF path and page/length
    if entry.file:
        # File is explicitly specified
        input_pdf_path = resolve_path(entry.file, default_dir)
        title = entry.title if entry.title else input_pdf_path.stem
        page = entry.page
        length = entry.length
    else:
        # Look up in the index by title
        if not entry.title:
            raise ValueError("Entry must have either 'file' or 'title'")
        if index is None:
            raise ValueError(
                f"Cannot look up '{entry.title}': no index available. "
                "Run 'tunery index <dir>' first."
            )

        location = index.lookup(entry.title)
        if not location:
            raise ValueError(f"Title '{entry.title}' not found in index")

        input_pdf_path = Path(location.source_path)
        title = entry.title
        # Use entry's page/length if specified, otherwise use from index
        page = entry.page if entry.page else location.page
        length = entry.length if entry.length else location.length

    entry_page = len(combined_pdf.pages)

    if entry.notes:
        # When notes are present, we need to process pages individually
        # to add the notes overlay to each page
        copy_pages_with_notes(
            combined_pdf, str(input_pdf_path), entry.notes, page, length
        )
    else:
        # No notes - use the standard bulk extraction
        copy_pages(combined_pdf, str(input_pdf_path), page, length)

    return title, entry_page


def render(layout_path: Path, output: Path, index_path: Path | None = None) -> None:
    """Combine PDFs according to the YAML layout file."""
    with open(str(layout_path), "r") as file:
        raw_records = yaml.safe_load(file)

    # Parse and validate the YAML records
    layout = Layout.model_validate(raw_records)

    # Get the directory of the YAML file for resolving relative paths
    default_dir = layout_path.parent.resolve()

    # Open index if it exists (for title lookups)
    index: Index | None = None
    if index_path and index_path.exists():
        index = Index(index_path)

    try:
        combined_pdf = pikepdf.Pdf.new()
        outline_items: list[pikepdf.OutlineItem] = []

        for record in layout.root:
            if isinstance(record, SectionEntry):
                # Process all items in this section
                children: list[pikepdf.OutlineItem] = []
                first_item_page: int | None = None

                for item in record.body:
                    item_title, item_page = process_file_entry(
                        item, default_dir, combined_pdf, index
                    )
                    if first_item_page is None:
                        first_item_page = item_page
                    children.append(pikepdf.OutlineItem(item_title, item_page))

                # Create section outline item with children
                section_item = pikepdf.OutlineItem(
                    record.section, first_item_page if first_item_page is not None else 0
                )
                section_item.children.extend(children)
                outline_items.append(section_item)
            else:
                # FileEntry - flat file entry
                title, page = process_file_entry(record, default_dir, combined_pdf, index)
                outline_items.append(pikepdf.OutlineItem(title, page))

        # Add the outline to the combined PDF
        with combined_pdf.open_outline() as pdf_outline:
            pdf_outline.root.extend(outline_items)

        # Save the combined PDF
        combined_pdf.save(str(output))
        combined_pdf.close()
    finally:
        if index:
            index.close()

