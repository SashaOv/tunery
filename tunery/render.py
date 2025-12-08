"""PDF rendering logic for combining PDFs with table of contents."""

import re
from pathlib import Path
from typing import List

import pikepdf
import yaml
from pydantic import BaseModel, Field, PositiveInt, RootModel, ValidationError
from rapidfuzz import fuzz, process

from tunery.index import FuzzyMatch, Index


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


class ConfigEntry(BaseModel):
    """Configuration entry specifying override directory."""

    override: str


class Layout(RootModel[List[ConfigEntry | SectionEntry | FileEntry]]):
    """The complete layout schema - a list of config, sections, and/or file entries."""

    pass


def get_page_label_to_index_map(pdf: pikepdf.Pdf) -> dict[int, int]:
    """
    Map page labels (what appears on the page) to physical page indices (0-based).
    
    PDFs can have custom page numbering (e.g., "i, ii, iii, 1, 2, 3" or "A-1, A-2").
    This function creates a mapping from page label numbers to physical page indices.
    
    Args:
        pdf: The PDF to analyze.
        
    Returns:
        Dictionary mapping page label number to physical page index (0-based).
    """
    label_to_index: dict[int, int] = {}
    
    # Try to get page labels from the PDF
    try:
        root = pdf.Root
        if '/PageLabels' in root:
            page_labels = root['/PageLabels']
            if '/Nums' in page_labels:
                nums = page_labels['/Nums']
                # Nums is an array of [page_index, label_dict, page_index, label_dict, ...]
                for i in range(0, len(nums), 2):
                    start_index_obj = nums[i]
                    start_index = int(start_index_obj) if hasattr(start_index_obj, '__int__') else 0
                    
                    if i + 1 < len(nums):
                        label_dict = nums[i + 1]
                        # Get the starting number and style for this label range
                        start_num = 1
                        style = None
                        
                        # Check if label_dict is actually a dictionary
                        if isinstance(label_dict, pikepdf.Dictionary):
                            # Get /St (starting number) if present
                            if '/St' in label_dict:
                                st_obj = label_dict['/St']
                                start_num = int(st_obj) if hasattr(st_obj, '__int__') else 1
                            
                            # Get /S (style) - /D = decimal, /a = lowercase alpha, /r = lowercase roman, etc.
                            if '/S' in label_dict:
                                style_obj = label_dict['/S']
                                if isinstance(style_obj, pikepdf.Name):
                                    style = str(style_obj)
                                elif isinstance(style_obj, pikepdf.Dictionary):
                                    # Nested style dict (uncommon)
                                    if '/St' in style_obj:
                                        st_obj = style_obj['/St']
                                        start_num = int(st_obj) if hasattr(st_obj, '__int__') else 1
                        
                        # Find where this range ends
                        if i + 2 < len(nums):
                            next_start_obj = nums[i + 2]
                            next_start = int(next_start_obj) if hasattr(next_start_obj, '__int__') else len(pdf.pages)
                        else:
                            next_start = len(pdf.pages)
                        
                        # Only map numeric labels (/D = decimal, or no /S which defaults to decimal)
                        # Skip non-numeric styles like /a (alpha), /r (roman), /A (uppercase alpha)
                        if style is None or style == '/D':
                            # Map all pages in this range with numeric labels
                            # If a label number already exists, prefer the longer range (main content)
                            for page_idx in range(start_index, next_start):
                                label_num = start_num + (page_idx - start_index)
                                # Only add if not already mapped, or if this range is longer
                                range_length = next_start - start_index
                                if label_num not in label_to_index:
                                    label_to_index[label_num] = page_idx
                                else:
                                    # If already mapped, check if current range is longer
                                    existing_idx = label_to_index[label_num]
                                    # Prefer the mapping that's part of a longer range
                                    # (This handles cases where there are multiple decimal ranges)
                                    existing_range_length = 0
                                    for j in range(0, len(nums), 2):
                                        if j + 1 < len(nums):
                                            existing_start = int(nums[j])
                                            existing_next = int(nums[j + 2]) if j + 2 < len(nums) else len(pdf.pages)
                                            if existing_start <= existing_idx < existing_next:
                                                existing_range_length = existing_next - existing_start
                                                break
                                    if range_length > existing_range_length:
                                        label_to_index[label_num] = page_idx
    except (KeyError, AttributeError, TypeError, IndexError, ValueError):
        # If page labels don't exist or are malformed, assume 1:1 mapping
        pass
    
    # If no labels found, create 1:1 mapping (page 1 = index 0, page 2 = index 1, etc.)
    if not label_to_index:
        for i in range(len(pdf.pages)):
            label_to_index[i + 1] = i
    
    return label_to_index


def get_page_range(
    pdf: pikepdf.Pdf,
    start_page: int | None = None,
    length: int | None = None,
) -> tuple[int, int]:
    """
    Calculate 0-based page range indices from page number.
    
    First tries to map page labels (what appears on the page) to physical page indices.
    If the page number doesn't exist as a label, treats it as a 1-based physical page index.

    Args:
        pdf: The PDF to extract pages from.
        start_page: Page number - either a page label or 1-based physical index.
        length: Number of pages to extract (optional, defaults to 1 if start_page is set).

    Returns:
        Tuple of (start_idx, end_idx) for slicing (end_idx is exclusive).
    """
    total_pages = len(pdf.pages)
    
    if start_page is not None:
        # Map page label to physical index
        # The index stores page labels (what appears on the page), not physical indices
        label_map = get_page_label_to_index_map(pdf)
        start_idx = label_map.get(start_page)
        
        if start_idx is None:
            # Page label not found - fallback to treating as 1-based physical index
            # (Some indexes might store physical indices instead of page labels)
            start_idx = start_page - 1
        
        # Calculate end index
        end_idx = start_idx + (length if length else 1)
        
        if end_idx > total_pages:
            raise ValueError(
                f"page {start_page} + length {length or 1} exceeds total pages {total_pages}"
            )
        if start_idx < 0:
            raise ValueError(f"page {start_page} is invalid (would be negative index)")
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
        start_idx, end_idx = get_page_range(input_pdf, start_page, length)
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
        start_idx, end_idx = get_page_range(input_pdf, start_page, length)

        for page_idx in range(start_idx, end_idx):
            add_page_with_notes(combined_pdf, input_pdf.pages[page_idx], notes)

        return end_idx - start_idx


def process_file_entry(
    entry: FileEntry,
    default_dir: Path,
    combined_pdf: pikepdf.Pdf,
    index: Index | None = None,
    override_dir: Path | None = None,
    preferred_source: Path | None = None,
    layout_path: Path | None = None,
) -> tuple[str, int, Path, str] | None:
    """
    Add the requested pages for an entry to the combined PDF and return its title, start page index, source path, and status message.

    Returns: (title, page, source_path, status_message) or None if not found.
    Status message format examples:
    - "found in \"The Real Book.Vol I\""
    - "found in \"The Real Book.Vol I\" (fuzzy matching)"
    - "found in ../../Handouts"
    - "not found. Is this \"Sweet Georgia Bright\"?"
    """
    # Determine the input PDF path and page/length
    input_pdf_path: Path | None = None
    if entry.file:
        # File is explicitly specified
        input_pdf_path = resolve_path(entry.file, default_dir)
        title = entry.title if entry.title else input_pdf_path.stem
        page = entry.page
        length = entry.length
        status = f'found    "{entry.title if entry.title else input_pdf_path.stem}" in {input_pdf_path.parent}'
    else:
        # Look up in the index by title
        if not entry.title:
            raise ValueError("Entry must have either 'file' or 'title'")
        
        title = entry.title
        
        # Check override directory first if specified
        override_fuzzy_score: float | None = None
        if override_dir and override_dir.exists():
            # Try exact match first
            override_file = override_dir / f"{title}.pdf"
            if not override_file.exists():
                # Try case-insensitive match
                title_lower = title.lower()
                for pdf_file in override_dir.glob("*.pdf"):
                    if pdf_file.stem.lower() == title_lower:
                        override_file = pdf_file
                        break
                else:
                    # No exact or case-insensitive match found, try fuzzy matching
                    pdf_files = list(override_dir.glob("*.pdf"))
                    if pdf_files:
                        # Normalize title for fuzzy matching (same as index normalization)
                        normalized_search = title.lower()
                        normalized_search = re.sub(r'[^\w\s]', '', normalized_search)
                        normalized_search = ' '.join(normalized_search.split())
                        
                        # Create mapping of normalized filenames to actual paths
                        file_map = {}
                        for pdf_file in pdf_files:
                            normalized_name = pdf_file.stem.lower()
                            normalized_name = re.sub(r'[^\w\s]', '', normalized_name)
                            normalized_name = ' '.join(normalized_name.split())
                            file_map[normalized_name] = pdf_file
                        
                        # Find best match
                        matches = process.extract(
                            normalized_search,
                            file_map.keys(),
                            scorer=fuzz.ratio,
                            score_cutoff=90,
                            limit=1,
                        )
                        if matches:
                            matched_name, score, _ = matches[0]
                            override_file = file_map[matched_name]
                            # Store score for status message
                            override_fuzzy_score = score
                        else:
                            override_file = None
                    else:
                        override_file = None
            
            if override_file and override_file.exists():
                # Use override file, skip index lookup
                input_pdf_path = override_file
                page = entry.page if entry.page else 1
                length = entry.length if entry.length else 1
                # Get relative path for display
                try:
                    if layout_path:
                        override_rel = override_file.relative_to(layout_path.parent)
                        override_path = override_rel.parent
                    else:
                        override_path = override_file.parent
                except ValueError:
                    override_path = override_file.parent
                
                # Check if this was a fuzzy match
                if override_fuzzy_score is not None:
                    status = f'found    "{title}" in {override_path} (fuzzy score {override_fuzzy_score:.0f}%)'
                else:
                    status = f'found    "{title}" in {override_path}'
            else:
                # No override file found, fall back to index lookup
                if index is None:
                    # Check for fuzzy matches in override directory as hints
                    fuzzy_hints = []
                    if override_dir and override_dir.exists():
                        pdf_files = list(override_dir.glob("*.pdf"))
                        if pdf_files:
                            normalized_search = title.lower()
                            normalized_search = re.sub(r'[^\w\s]', '', normalized_search)
                            normalized_search = ' '.join(normalized_search.split())
                            file_map = {}
                            for pdf_file in pdf_files:
                                normalized_name = pdf_file.stem.lower()
                                normalized_name = re.sub(r'[^\w\s]', '', normalized_name)
                                normalized_name = ' '.join(normalized_name.split())
                                file_map[normalized_name] = pdf_file
                            matches = process.extract(
                                normalized_search,
                                file_map.keys(),
                                scorer=fuzz.ratio,
                                score_cutoff=70,  # Lower threshold for hints
                                limit=1,
                            )
                            if matches:
                                matched_name, score, _ = matches[0]
                                fuzzy_hints.append(f'"{file_map[matched_name].stem}"')
                    status = f'not found "{title}"'
                    if fuzzy_hints:
                        status += f'. Is this {fuzzy_hints[0]}?'
                    print(f'{status}')
                    return None

                # Try to find a match in the preferred source first
                exact_match = False
                if preferred_source:
                    all_matches = index.lookup_all(title)
                    preferred_match = next(
                        (m for m in all_matches if Path(m.source_path).resolve() == preferred_source.resolve()),
                        None
                    )
                    if preferred_match:
                        location = preferred_match
                        exact_match = True
                    else:
                        location = index.lookup(title)
                        exact_match = location is not None
                else:
                    location = index.lookup(title)
                    exact_match = location is not None
                
                # If exact match failed, try fuzzy matching
                index_fuzzy_match_used: FuzzyMatch | None = None
                if not location:
                    fuzzy_matches = index.lookup_fuzzy_edit_distance(title, score_cutoff=90, limit=1)
                    if fuzzy_matches:
                        index_fuzzy_match_used = fuzzy_matches[0]
                        location = index_fuzzy_match_used.location
                
                if not location:
                    # Not found - get hints from fuzzy matches (even below threshold)
                    fuzzy_hints = []
                    fuzzy_matches = index.lookup_fuzzy_edit_distance(title, score_cutoff=70, limit=1)
                    if fuzzy_matches:
                        hint_match = fuzzy_matches[0]
                        matched_title = hint_match.matched_title
                        # Get source name
                        source_name = Path(hint_match.location.source_path).stem
                        fuzzy_hints.append(f'"{matched_title}" in "{source_name}"')
                    
                    status = f'not found "{title}"'
                    if fuzzy_hints:
                        status += f'. Is this {fuzzy_hints[0]}?'
                    print(f'{status}')
                    return None

                input_pdf_path = Path(location.source_path)
                source_name = input_pdf_path.stem
                if exact_match:
                    status = f'found    "{title}" in "{source_name}"'
                else:
                    # index_fuzzy_match_used was set earlier and is not None
                    assert index_fuzzy_match_used is not None
                    status = f'found    "{title}" in "{source_name}" (fuzzy score {index_fuzzy_match_used.score:.0f}%)'
                # Use entry's page/length if specified, otherwise use from index
                page = entry.page if entry.page else location.page
                length = entry.length if entry.length else location.length
        else:
            # No override directory, proceed with index lookup
            if index is None:
                raise ValueError(
                    f"Cannot look up '{title}': no index available. "
                    "Run 'tunery index <dir>' first."
                )

            # Try to find a match in the preferred source first
            exact_match = False
            if preferred_source:
                all_matches = index.lookup_all(title)
                preferred_match = next(
                    (m for m in all_matches if Path(m.source_path).resolve() == preferred_source.resolve()),
                    None
                )
                if preferred_match:
                    location = preferred_match
                    exact_match = True
                else:
                    location = index.lookup(title)
                    exact_match = location is not None
            else:
                location = index.lookup(title)
                exact_match = location is not None
            
            # If exact match failed, try fuzzy matching
            index_fuzzy_match: FuzzyMatch | None = None
            if not location:
                fuzzy_matches = index.lookup_fuzzy_edit_distance(title, score_cutoff=90, limit=1)
                if fuzzy_matches:
                    index_fuzzy_match = fuzzy_matches[0]
                    location = index_fuzzy_match.location
            
            if not location:
                # Not found - get hints from fuzzy matches (even below threshold)
                fuzzy_hints = []
                fuzzy_matches = index.lookup_fuzzy_edit_distance(title, score_cutoff=70, limit=1)
                if fuzzy_matches:
                    hint_match = fuzzy_matches[0]
                    matched_title = hint_match.matched_title
                    # Get source name
                    source_name = Path(hint_match.location.source_path).stem
                    fuzzy_hints.append(f'"{matched_title}" in "{source_name}"')
                
                    status = f'not found "{title}"'
                    if fuzzy_hints:
                        status += f'. Is this {fuzzy_hints[0]}?'
                    print(f'{status}')
                return None

            input_pdf_path = Path(location.source_path)
            source_name = input_pdf_path.stem
            if exact_match:
                status = f'found    "{title}" in "{source_name}"'
            else:
                # index_fuzzy_match was set earlier and is not None
                assert index_fuzzy_match is not None
                status = f'found    "{title}" in "{source_name}" (fuzzy score {index_fuzzy_match.score:.0f}%)'
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

    return title, entry_page, input_pdf_path, status


def render(
    layout_path: Path,
    output: Path,
    index_path: Path | None = None,
    override_dir: Path | None = None,
) -> None:
    """Combine PDFs according to the YAML layout file."""
    try:
        with open(str(layout_path), "r") as file:
            file_content = file.read()
            raw_data = yaml.safe_load(file_content)
    except yaml.YAMLError as e:
        # Add file path and line information to YAML parsing errors
        error_msg = f"YAML parsing error in {layout_path}"
        # Some YAML errors have problem_mark with line/column info
        problem_mark = getattr(e, "problem_mark", None)
        if problem_mark is not None:
            error_msg += f" at line {problem_mark.line + 1}, column {problem_mark.column + 1}"
        error_msg += f": {e}"
        raise ValueError(error_msg) from e

    # Parse and validate the YAML records using Pydantic
    try:
        layout = Layout.model_validate(raw_data)
    except ValidationError as e:
        # Use Pydantic's built-in error formatting
        error_msg = f"Validation error in {layout_path}:\n{e}"
        raise ValueError(error_msg) from e

    # Extract config entries and override directory
    layout_entries: list[SectionEntry | FileEntry] = []
    for record in layout.root:
        if isinstance(record, ConfigEntry):
            # Override from YAML takes precedence over command line
            override_dir = resolve_path(record.override, layout_path.parent)
        else:
            # Keep non-config entries for processing
            layout_entries.append(record)

    # Get the directory of the YAML file for resolving relative paths
    default_dir = layout_path.parent.resolve()

    # Open index if it exists (for title lookups)
    index: Index | None = None
    if index_path and index_path.exists():
        index = Index(index_path)

    try:
        combined_pdf = pikepdf.Pdf.new()
        outline_items: list[pikepdf.OutlineItem] = []
        last_source: Path | None = None  # Track the last used source file

        for record in layout_entries:
            if isinstance(record, SectionEntry):
                # Process all items in this section
                children: list[pikepdf.OutlineItem] = []
                first_item_page: int | None = None

                for item in record.body:
                    result = process_file_entry(
                        item, default_dir, combined_pdf, index, override_dir, last_source, layout_path
                    )
                    if result is None:
                        # Tune not found - status was already printed in process_file_entry
                        continue
                    item_title, item_page, item_source, status = result
                    print(f'{status}')
                    if first_item_page is None:
                        first_item_page = item_page
                    children.append(pikepdf.OutlineItem(item_title, item_page))
                    # Update last_source for next lookup
                    last_source = item_source

                # Create section outline item with children
                section_item = pikepdf.OutlineItem(
                    record.section, first_item_page if first_item_page is not None else 0
                )
                section_item.children.extend(children)
                outline_items.append(section_item)
            else:
                # FileEntry - flat file entry
                result = process_file_entry(record, default_dir, combined_pdf, index, override_dir, last_source, layout_path)
                if result is None:
                    # Tune not found - status was already printed in process_file_entry
                    continue
                title, page, source, status = result
                print(f'{status}')
                outline_items.append(pikepdf.OutlineItem(title, page))
                # Update last_source for next entry
                last_source = source

        # Add the outline to the combined PDF
        with combined_pdf.open_outline() as pdf_outline:
            pdf_outline.root.extend(outline_items)

        # Save the combined PDF
        combined_pdf.save(str(output))
        combined_pdf.close()
    finally:
        if index:
            index.close()

