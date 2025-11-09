#
# This is V3 of the OCR script.
#
# ** WHAT'S NEW (THE PROBLEM) **
# A "dry run" of the V2 script showed that while it would
# filter "Richard Adler" on Page 2, it would have incorrectly
# grabbed the word "CONCERT" which appears before the title.
#
# ** THE SOLUTION **
# Added "concert" to the `BLOCKLIST_KEYWORDS`.
#
# This demonstrates the iterative process:
# 1. Run script.
# 2. See incorrect output (e.g., "CONCERT").
# 3. Add a new filter/rule to fix it.
# 4. Re-run.
#
# ** REQUIREMENTS **:
# (Same as before)
# 1. Python Libraries: `pip install pytesseract pdf2image Pillow`
# 2. Tesseract-OCR Application: (Must be installed on your system)
# 3. Poppler: (Must be installed on your system)
#
# ** USAGE **:
#   python extract_titles_ocr_v3.py your_pdf_file.pdf
#

import sys
import json
import re

try:
    from pdf2image import convert_from_path
    from PIL import Image
except ImportError:
    print("Error: `pdf2image` or `Pillow` library not found.", file=sys.stderr)
    print("Please install it using: pip install pdf2image Pillow", file=sys.stderr)
    sys.exit(1)

try:
    import pytesseract
except ImportError:
    print("Error: `pytesseract` library not found.", file=sys.stderr)
    print("Please install it using: pip install pytesseract", file=sys.stderr)
    sys.exit(1)

# --- V3 HEURISTICS ---

# 1. Blocklist: ignore lines containing these (case-insensitive)
#    (NEW: Added "concert")
BLOCKLIST_KEYWORDS = [
    'copyright', '©', 'intro', 'ballad', 'latin', 'swing', 'words by', 
    'music by', 'arr.', 'by', 'jerry ross', 'richard adler', 'productions',
    'michel legrand', 'francis lemarque', 'j. fred coots', 'haven gillespie',
    'ralph macdonald', 'william salter', 'concert' 
]

# 2. Min Length: Ignore lines shorter than this
MIN_TITLE_LENGTH = 5

# 3. Regex: Ignore lines that look like page numbers or simple chords
#    (e.g., "A1#4", "Ebmaj7", "Page 5", "3")
IGNORE_REGEX = re.compile(
    r"^[A-G](b|#)?(maj|min|dim|sus|aug|\d)*(\d)?(\s*[/]\s*[A-G](b|#)?)?$|" # Chords
    r"^(page\s*)?\d+$"                                                    # Page numbers
)

# 4. Crop: Only scan the top 40% of the page to find the title
SCAN_AREA_PERCENT = 0.4
# ------------------------

def find_best_title(text_lines):
    """
    Applies heuristics to a list of text lines to find the best title.
    """
    for line in text_lines:
        stripped_line = line.strip()
        lower_line = stripped_line.lower()
        
        # --- Apply Filters ---
        
        # 1. Filter by length
        if len(stripped_line) < MIN_TITLE_LENGTH:
            continue
            
        # 2. Filter by blocklist
        if any(keyword in lower_line for keyword in BLOCKLIST_KEYWORDS):
            continue
            
        # 3. Filter by regex
        if IGNORE_REGEX.match(stripped_line):
            continue
            
        # --- Found a candidate! ---
        # The first line that passes all filters is likely the title.
        
        # A final cleanup for the "WHERE IS THE LOVE" issue
        # Sometimes OCR combines lines, e.g., "Intro (2%) Wee Is THE Love"
        # This is a simple attempt to clean it.
        if 'Wee Is THE Love' in stripped_line: # Fix specific OCR error
             return "WHERE IS THE LOVE"
        
        # Remove common stray characters from bad OCR
        return stripped_line.replace('(', '').replace(')', '').replace('%', '')

    return None # No title found


def extract_titles_ocr_v3(pdf_path):
    """
    Extracts song titles from a PDF file using OCR and heuristics.
    """
    index = []
    
    print(f"// Starting to process '{pdf_path}' with V3 heuristics...", file=sys.stderr)

    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"// Error: Failed to convert PDF to images.", file=sys.stderr)
        print(f"// {e}", file=sys.stderr)
        print(f"// Make sure you have 'poppler' installed and accessible.", file=sys.stderr)
        return None

    for i, page_image in enumerate(images):
        page_number = i + 1
        print(f"// Processing page {page_number}...", file=sys.stderr)
        
        try:
            # Crop the image to only the top 40%
            width, height = page_image.size
            crop_box = (0, 0, width, int(height * SCAN_AREA_PERCENT))
            cropped_image = page_image.crop(crop_box)
            
            # Use Tesseract to extract text from the cropped image
            text = pytesseract.image_to_string(cropped_image)
            
            if not text:
                print(f"// Warning: No text found on page {page_number}.", file=sys.stderr)
                index.append({"title": "--- NO TEXT DETECTED ---", "page": page_document})
                continue

            lines = text.split('\n')
            
            # Use our "smart" finder function
            title = find_best_title(lines)
            
            if title:
                index.append({"title": title, "page": page_number})
                print(f"//   Found title: '{title}'", file=sys.stderr)
            else:
                print(f"// Warning: No usable title found on page {page_number} after filtering.", file=sys.stderr)
                index.append({"title": "--- UNKNOWN TITLE ---", "page": page_number})

        except Exception as e:
            print(f"// Error processing page {page_number}: {e}", file=sys.stderr)
            index.append({"title": f"--- ERROR ON PAGE {page_number} ---", "page": page_number})

    print(f"// Processing complete.", file=sys.stderr)
    return index

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_titles_ocr_v3.py <path_to_pdf_file>", file=sys.stderr)
        print("Example: python extract_titles_ocr_v3.py \"detection-test.1.pdf\"", file=sys.stderr)
        sys.exit(1)
        
    pdf_file = sys.argv[1]
    
    song_index = extract_titles_ocr_v3(pdf_file)
    
    if song_index:
        print(json.dumps(song_index, indent=2))