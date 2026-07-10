from dataclasses import dataclass, field
from pathlib import Path

import pikepdf


def get_page_label_to_index_map(pdf: pikepdf.Pdf) -> dict[int, int]:
    """
    Map page labels to physical page indices.

    PDFs can have custom page numbering such as front matter followed by main
    decimal pages. This function maps decimal labels to 0-based physical
    indices and falls back to 1:1 page numbering when labels are unavailable.
    """
    label_to_index: dict[int, int] = {}

    try:
        root = pdf.Root
        if "/PageLabels" in root:
            page_labels = root["/PageLabels"]
            if "/Nums" in page_labels:
                nums = page_labels["/Nums"]
                for i in range(0, len(nums), 2):
                    start_index_obj = nums[i]
                    start_index = (
                        int(start_index_obj)
                        if hasattr(start_index_obj, "__int__")
                        else 0
                    )

                    if i + 1 < len(nums):
                        label_dict = nums[i + 1]
                        start_num = 1
                        style = None

                        if isinstance(label_dict, pikepdf.Dictionary):
                            if "/St" in label_dict:
                                st_obj = label_dict["/St"]
                                start_num = (
                                    int(st_obj) if hasattr(st_obj, "__int__") else 1
                                )

                            if "/S" in label_dict:
                                style_obj = label_dict["/S"]
                                if isinstance(style_obj, pikepdf.Name):
                                    style = str(style_obj)
                                elif isinstance(style_obj, pikepdf.Dictionary):
                                    if "/St" in style_obj:
                                        st_obj = style_obj["/St"]
                                        start_num = (
                                            int(st_obj)
                                            if hasattr(st_obj, "__int__")
                                            else 1
                                        )

                        if i + 2 < len(nums):
                            next_start_obj = nums[i + 2]
                            next_start = (
                                int(next_start_obj)
                                if hasattr(next_start_obj, "__int__")
                                else len(pdf.pages)
                            )
                        else:
                            next_start = len(pdf.pages)

                        if style is None or style == "/D":
                            for page_idx in range(start_index, next_start):
                                label_num = start_num + (page_idx - start_index)
                                range_length = next_start - start_index
                                if label_num not in label_to_index:
                                    label_to_index[label_num] = page_idx
                                else:
                                    existing_idx = label_to_index[label_num]
                                    existing_range_length = 0
                                    for j in range(0, len(nums), 2):
                                        if j + 1 < len(nums):
                                            existing_start = int(nums[j])
                                            existing_next = (
                                                int(nums[j + 2])
                                                if j + 2 < len(nums)
                                                else len(pdf.pages)
                                            )
                                            if existing_start <= existing_idx < existing_next:
                                                existing_range_length = (
                                                    existing_next - existing_start
                                                )
                                                break
                                    if range_length > existing_range_length:
                                        label_to_index[label_num] = page_idx
    except (KeyError, AttributeError, TypeError, IndexError, ValueError):
        pass

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

    First interpret `start_page` as a PDF page label. If no label exists, treat
    it as a 1-based physical page index.
    """
    total_pages = len(pdf.pages)

    if start_page is not None:
        label_map = get_page_label_to_index_map(pdf)
        start_idx = label_map.get(start_page)

        if start_idx is None:
            start_idx = start_page - 1

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
    """Copy pages from a source PDF to the combined PDF."""
    with pikepdf.Pdf.open(input_pdf_path) as input_pdf:
        start_idx, end_idx = get_page_range(input_pdf, start_page, length)
        pages_to_add = input_pdf.pages[start_idx:end_idx]
        combined_pdf.pages.extend(pages_to_add)
        return len(pages_to_add)


def escape_pdf_string(text: str) -> str:
    """Escape special characters for PDF string literals."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


# Helvetica character widths in 1000ths of a point at 1pt size.
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
        width += HELVETICA_WIDTHS.get(char, 556)
    return width * font_size / 1000.0


def wrap_text(text: str, max_width: float, font_size: float, margin: float) -> list[str]:
    """Wrap text to fit within max_width, accounting for margins."""
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
                lines.append(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def calculate_notes_height(
    notes: str, page_width: float, font_size: float, margin: float
) -> tuple[list[str], float]:
    """Calculate the height needed for notes text."""
    lines = wrap_text(notes, page_width, font_size, margin)
    line_height = font_size * 1.2
    total_height = len(lines) * line_height + 2 * font_size
    return lines, total_height


def add_page_with_notes(
    combined_pdf: pikepdf.Pdf,
    source_page: pikepdf.Page,
    notes: str,
    font_size: float = 10.0,
    margin: float = 36.0,
) -> None:
    """Add a page to combined_pdf with notes at the bottom."""
    mediabox = pikepdf.Rectangle(source_page.mediabox)

    lines, notes_height = calculate_notes_height(
        notes, mediabox.width, font_size, margin
    )

    new_page_dict = pikepdf.Dictionary(
        Type=pikepdf.Name.Page,
        MediaBox=source_page.mediabox,
        Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary()),
    )
    new_page = pikepdf.Page(combined_pdf.make_indirect(new_page_dict))

    form_xobj = source_page.as_form_xobject()
    form_xobj_copy = combined_pdf.copy_foreign(form_xobj)
    xobj_name = new_page.add_resource(form_xobj_copy, pikepdf.Name.XObject)

    scale_factor = (mediabox.height - notes_height) / mediabox.height
    content = f"""
q
{scale_factor} 0 0 {scale_factor} 0 {notes_height} cm
{xobj_name} Do
Q
""".encode()
    new_page.contents_add(combined_pdf.make_stream(content))

    font_dict = pikepdf.Dictionary(
        Type=pikepdf.Name.Font,
        Subtype=pikepdf.Name.Type1,
        BaseFont=pikepdf.Name.Helvetica,
    )
    new_page.Resources.Font.F1 = combined_pdf.make_indirect(font_dict)

    line_height = font_size * 1.2
    total_text_height = len(lines) * line_height
    start_y = (notes_height + total_text_height) / 2 - font_size

    text_ops: list[str] = [
        "q",
        "BT",
        f"/F1 {font_size} Tf",
        f"{line_height} TL",
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
    """Copy pages from a source PDF to the combined PDF with notes."""
    with pikepdf.Pdf.open(input_pdf_path) as input_pdf:
        start_idx, end_idx = get_page_range(input_pdf, start_page, length)

        for page_idx in range(start_idx, end_idx):
            add_page_with_notes(combined_pdf, input_pdf.pages[page_idx], notes)

        return end_idx - start_idx


@dataclass
class _OutlineNode:
    title: str
    page: int | None = None
    children: list["_OutlineNode"] = field(default_factory=list)

    def first_page(self) -> int | None:
        if self.page is not None:
            return self.page
        for child in self.children:
            child_page = child.first_page()
            if child_page is not None:
                return child_page
        return None

    def to_outline_item(self) -> pikepdf.OutlineItem:
        item = pikepdf.OutlineItem(self.title, self.first_page() or 0)
        item.children.extend(child.to_outline_item() for child in self.children)
        return item


class Composer:
    def __init__(self, output_path: Path, *, autosave: bool = True) -> None:
        self._path = Path(output_path)
        self._pdf = pikepdf.Pdf.new()
        self._outline: list[_OutlineNode] = []
        self._section_stack: list[_OutlineNode] = []
        self._closed = False
        self._autosave = autosave

    @property
    def page_count(self) -> int:
        return len(self._pdf.pages)

    def add(
        self,
        title: str,
        source: Path,
        start: int | None = None,
        pages: int | None = None,
        notes: str | None = None,
    ) -> int:
        """Add a tune to the file"""
        self._ensure_open()

        page = len(self._pdf.pages)
        if notes:
            copy_pages_with_notes(self._pdf, str(source), notes, start, pages)
        else:
            copy_pages(self._pdf, str(source), start, pages)

        self._append_outline_node(_OutlineNode(title=title, page=page))
        self._save_if_needed()
        return page

    def start_section(self, title: str) -> None:
        self._ensure_open()
        section = _OutlineNode(title=title)
        self._append_outline_node(section)
        self._section_stack.append(section)
        self._save_if_needed()

    def end_section(self) -> None:
        self._ensure_open()
        if not self._section_stack:
            raise ValueError("No open section to end")
        self._section_stack.pop()
        self._save_if_needed()

    def save(self) -> None:
        """Write the current setbook to the output path."""
        self._ensure_open()
        if not self._pdf.pages:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._pdf.open_outline() as outline:
            del outline.root[:]
            outline.root.extend(node.to_outline_item() for node in self._outline)
        self._pdf.save(self._path)

    def close(self) -> None:
        if self._closed:
            return
        if self._pdf.pages:
            self.save()
        self._pdf.close()
        self._closed = True

    def __enter__(self) -> "Composer":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _append_outline_node(self, node: _OutlineNode) -> None:
        if self._section_stack:
            self._section_stack[-1].children.append(node)
        else:
            self._outline.append(node)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("Composer output is already closed")

    def _save_if_needed(self) -> None:
        if self._autosave and self._pdf.pages:
            self.save()
