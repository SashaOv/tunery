from pathlib import Path
import shutil
import sys
from typing import Optional
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import re


def check_poppler_installation():
    if not shutil.which("pdftotext") or not shutil.which("pdfimages"):
        if sys.platform == "darwin":
            platform_instructions = ": `brew install poppler`"
        elif sys.platform == "linux":
            platform_instructions = ": `apt-get install poppler-utils`"
        else:
            platform_instructions = ""
        print(
            "WARNING: pdftotext and pdfimages not found."
            + f"\n    Please install poppler{platform_instructions}."
            + "\n    See https://poppler.freedesktop.org/ for details.\n",
            file=sys.stderr,
        )


def extract_text_from_page(
    pdf_path: Path, page_num: int = 0, use_ocr: bool = False
) -> tuple[str, list[dict]]:
    """Extract text from a PDF page, using OCR if needed.
    
    Returns:
        Tuple of (text, words_with_positions) where words_with_positions is a list
        of dicts with 'text', 'top', 'left', 'conf' keys.
    """
    words_with_pos = []
    
    if use_ocr:
        try:
            images = convert_from_path(
                str(pdf_path), first_page=page_num + 1, last_page=page_num + 1
            )
            if images:
                img = images[0]
                # Try PSM 6 first (uniform block) for better layout detection
                text = pytesseract.image_to_string(img, config='--psm 6')
                
                # Get word positions for better title detection
                # Also try PSM 11 (sparse text) as fallback for pages with poor OCR
                data = pytesseract.image_to_data(img, config='--psm 6', output_type=pytesseract.Output.DICT)
                
                # If we got very little text with PSM 6, try PSM 11
                if len(text.strip()) < 50:
                    text_psm11 = pytesseract.image_to_string(img, config='--psm 11')
                    if len(text_psm11.strip()) > len(text.strip()):
                        text = text_psm11
                        data = pytesseract.image_to_data(img, config='--psm 11', output_type=pytesseract.Output.DICT)
                for i in range(len(data['text'])):
                    if data['text'][i].strip() and data['conf'][i] > 0:
                        words_with_pos.append({
                            'text': data['text'][i],
                            'top': data['top'][i],
                            'left': data['left'][i],
                            'conf': data['conf'][i]
                        })
                
                return text, words_with_pos
        except Exception as e:
            error_msg = str(e)
            explanation = "Poppler not found" if "poppler" in error_msg.lower() else ""
            print(
                f"WARNING: OCR failed for page {page_num + 1}: {explanation}\n    Cause: {error_msg}",
                file=sys.stderr,
            )
            return "", []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num < len(pdf.pages):
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                # Get word positions from pdfplumber
                try:
                    words = page.extract_words()
                    for word in words:
                        words_with_pos.append({
                            'text': word['text'],
                            'top': word.get('top', 0),
                            'left': word.get('left', 0),
                            'conf': 100  # Assume high confidence for text-based PDFs
                        })
                except (KeyError, AttributeError):
                    # Some PDFs might not have word position data
                    pass
                
                return text if text else "", words_with_pos
    except Exception as e:
        print(f"Warning: Text extraction failed for page {page_num + 1}: {e}")
        return "", []

    return "", []


def detect_piece_name(text: str, min_length: int = 3, words_with_pos: Optional[list[dict]] = None) -> Optional[str]:
    """Detect piece name from page text. Looks for titles at the top of pages.
    
    If words_with_pos is provided, prioritizes text at the top of the page.
    """
    # If we have positional information, use it to find the title at the top
    if words_with_pos:
        # Sort words by vertical position (top to bottom), then horizontal (left to right)
        sorted_words = sorted(words_with_pos, key=lambda w: (w['top'], w['left']))
        
        # Look for title-like text in the top portion of the page
        # Consider top 40% of the page for title detection (increased from 30%)
        if sorted_words:
            max_y = max(w['top'] for w in sorted_words)
            top_threshold = max_y * 0.4
            
            # Collect words from the top portion
            # For the very top (first 100 pixels), be more lenient with confidence
            # Bold/underlined titles often have lower OCR confidence
            very_top_threshold = 150  # First ~150 pixels
            very_top_words = [
                w for w in sorted_words 
                if w['top'] <= very_top_threshold and w['conf'] >= 10  # Very lenient for top
            ]
            
            # For rest of top portion, use normal confidence threshold
            min_conf = 20
            rest_top_words = [
                w for w in sorted_words 
                if very_top_threshold < w['top'] <= top_threshold and w['conf'] >= min_conf
            ]
            
            # Combine, prioritizing very top words
            top_words = very_top_words + rest_top_words
            
            # If we still didn't find many words, lower threshold further
            if len(top_words) < 3:
                min_conf = 5
                top_words = [
                    w for w in sorted_words 
                    if w['top'] <= top_threshold and w['conf'] >= min_conf
                ]
            
            # Group words by line (similar y positions)
            # Words on the same line should have y values within ~15 pixels
            lines = []
            current_line_words = []
            last_y = None
            
            for word in top_words[:40]:  # Check first 40 words
                if last_y is None or abs(word['top'] - last_y) <= 15:  # Same line
                    current_line_words.append(word)
                else:
                    # New line detected
                    if current_line_words:
                        lines.append(current_line_words)
                    current_line_words = [word]
                last_y = word['top']
            
            # Don't forget the last line
            if current_line_words:
                lines.append(current_line_words)
            
            # Try each line from top to bottom
            # Track the best candidate (topmost valid title)
            best_candidate = None
            best_candidate_y = None
            for line_words in lines:
                if not line_words:
                    continue
                
                # Form the line text
                line_text = ' '.join(w['text'] for w in line_words).strip()
                line_y = line_words[0]['top']  # Y position of this line
                
                # Skip if it looks like a composer name (2-4 capitalized words, all proper case)
                # But be careful - titles can also be 2-3 words, so we need to check context
                composer_pattern = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$")
                # Only skip if it matches composer pattern AND it's likely a name (not a title)
                # Composer names are usually just names, not phrases like "Whatever Lola Wants"
                if composer_pattern.match(line_text) and 2 <= len(line_words) <= 4:
                    # Check if it looks like a name (common name patterns) vs a title
                    # Names usually don't have words like "the", "is", "what", "where", etc.
                    common_title_words = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'where', 'when', 'why', 'how', 'whatever', 'who']
                    has_title_words = any(w['text'].lower() in common_title_words for w in line_words)
                    # If it has title words, it's probably a title, not a composer name
                    if not has_title_words:
                        continue
                
                # Skip lines that contain composer/credit keywords
                credit_keywords = ['legrand', 'dem', 'norman', 'gimbel', 'copyright', 'productions', 'music', 'corp', 'imb', 'sacriaes']
                if any(keyword in line_text.lower() for keyword in credit_keywords):
                    continue
                
                # Skip lines that are mostly symbols/punctuation (but allow some punctuation for titles)
                # Be more lenient for the very top where OCR might misread bold/underlined text
                symbol_ratio = len(re.sub(r'[\w\s]', '', line_text)) / len(line_text) if line_text else 0
                is_very_top = line_words and line_words[0]['top'] < 200
                max_symbol_ratio = 0.7 if is_very_top else 0.6  # More lenient for top
                if symbol_ratio > max_symbol_ratio:
                    # But if it's at the top and has some alphabetic content, don't skip
                    alpha_chars = len(re.sub(r'[^\w]', '', line_text))
                    if not (is_very_top and alpha_chars >= 5):
                        continue
                
                # Skip lines that are clearly garbage OCR (mostly single chars, symbols, or very low confidence)
                # But be more lenient for the very top of the page where titles might have lower confidence
                avg_conf = sum(w['conf'] for w in line_words) / len(line_words) if line_words else 0
                is_very_top = line_words and line_words[0]['top'] < 200
                min_conf_threshold = 15 if is_very_top else 25
                if avg_conf < min_conf_threshold and len(line_words) > 3:  # Low confidence and multiple words = likely garbage
                    # Exception: if it's at the very top and has some reasonable words, don't skip
                    reasonable_words = [w for w in line_words if len(w['text']) >= 3 and w['text'].isalpha()]
                    if not (is_very_top and len(reasonable_words) >= 2):
                        continue
                
                # Skip lines that look like garbage OCR misreads (lots of single chars, symbols, low confidence)
                # Check if line has too many single-character "words" which often indicates OCR failure
                single_char_words = sum(1 for w in line_words if len(w['text']) == 1 and not w['text'].isalnum())
                if single_char_words > len(line_words) * 0.5 and len(line_words) > 3:
                    # This looks like garbage OCR, skip it
                    continue
                
                # Skip lines that start with quotes/symbols and have mostly non-alphabetic content
                # This catches cases like '"4 NWP . : ' which are clearly OCR garbage
                if line_text and line_text[0] in ['"', "'", '.', ',', ':', ';']:
                    alpha_chars = len([c for c in line_text if c.isalnum()])
                    alpha_ratio = alpha_chars / len(line_text) if line_text else 0
                    # Also check if it has too many single-char words starting with symbols
                    symbol_start_words = sum(1 for w in line_words if len(w['text']) > 0 and w['text'][0] in ['"', "'", '.', ',', ':', ';', '|'])
                    if alpha_ratio < 0.4 or (symbol_start_words > 2 and alpha_ratio < 0.5):  # More aggressive filtering
                        continue
                
                # Skip lines that look like garbage OCR (very short words, low confidence, mixed with symbols)
                # Examples: 'maz Sd eee] A' - has very short words and brackets
                if len(line_words) >= 3:
                    short_words = sum(1 for w in line_words if len(w['text']) <= 2)
                    has_brackets = any('[' in w['text'] or ']' in w['text'] for w in line_words)
                    avg_conf = sum(w['conf'] for w in line_words) / len(line_words)
                    # Lines with brackets are often OCR garbage, especially with short words or low confidence
                    # Also skip if most words are very short and confidence is low
                    if (has_brackets and (avg_conf < 50 or short_words >= len(line_words) * 0.4)) or (short_words >= len(line_words) * 0.5 and avg_conf < 40):
                        continue
                
                # If we have a reasonable title candidate
                # Allow single words if they're substantial (5+ chars), or multi-word titles
                is_single_word = len(line_words) == 1
                min_chars = 5 if is_single_word else max(min_length, 10)
                
                if len(line_text) >= min_chars:
                    # For single words, check if it's a valid title (not notation, not too short)
                    if is_single_word:
                        word_text = line_words[0]['text'].rstrip(".,;: ")
                        # Skip if it's musical notation
                        if re.match(r"^[A-G][b#]?[-]?[0-9()]*$", word_text, re.IGNORECASE):
                            continue
                        # Skip very short words
                        if len(word_text) < 5:
                            continue
                        # Skip common non-title words
                        if word_text.lower() in ['the', 'and', 'or', 'a', 'an', 'ie', 'etc']:
                            continue
                        # If this is the first valid candidate, or it's higher on the page, use it
                        if best_candidate is None or (best_candidate_y is not None and line_y < best_candidate_y):
                            best_candidate = word_text
                            best_candidate_y = line_y
                        continue  # Don't process further for single words
                    
                    # For multi-word titles, require at least 2 words
                    if len(line_words) >= 2:
                        # Check if it looks like a title (not all notation)
                        alphanumeric = re.sub(r"[^\w\s]", "", line_text)
                        if len(alphanumeric) >= min_length:
                            # Skip if it's mostly musical notation
                            notation_count = sum(1 for w in line_words if re.match(r"^[A-G][b#]?[-]?[0-9()]*$", w['text'], re.IGNORECASE))
                            if notation_count < len(line_words) / 2:  # Less than half notation
                                # Clean up trailing punctuation
                                cleaned = line_text.rstrip(".,;: ")
                                # If this is the first valid candidate, or it's higher on the page, use it
                                if best_candidate is None or (best_candidate_y is not None and line_y < best_candidate_y):
                                    best_candidate = cleaned if cleaned else line_text
                                    best_candidate_y = line_y
            
            # If we found a valid candidate, return it (prioritizing topmost)
            if best_candidate:
                return best_candidate
            
            # Fall back to regular text-based detection
            # If we got here and didn't find anything, return None (will become "Unknown")
    
    if not text:
        return None

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        return None

    # Common patterns for piece names:
    # 1. First substantial line (not too short, not all caps unless it's a title)
    # 2. Lines that look like titles (capitalized words, not all caps)
    # 3. Skip common headers like page numbers, composer names in certain formats

    # Patterns that indicate composer/artist names (usually 2-4 capitalized words)
    composer_pattern = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$")
    
    # Check first 20 lines to catch titles that might be lower on the page
    for line in lines[:20]:
        # Skip very short lines
        if len(line) < min_length:
            continue

        # Skip lines that are just numbers (page numbers)
        if re.match(r"^\d+$", line):
            continue

        # Skip common headers/footers
        footer_keywords = [
            "page",
            "page of",
            "copyright",
            "all rights reserved",
            "composer",
            "arranged by",
        ]
        if any(keyword in line.lower() for keyword in footer_keywords):
            continue

        # Skip lines that are mostly numbers or special characters
        # But be more lenient - allow some special characters for titles like "No. 1" or "Op. 2"
        alphanumeric_chars = re.sub(r"[^\w\s]", "", line)
        if len(alphanumeric_chars) < min_length:
            continue

        # Skip lines that look like musical notation markers (too many special chars)
        special_char_ratio = len(re.sub(r"[\w\s]", "", line)) / len(line) if line else 0
        if special_char_ratio > 0.5:  # More than 50% special characters
            continue
        
        # Skip lines that look like musical notation (contain common notation patterns)
        # e.g., "Bb-", "C7", "F7", "A2", etc.
        notation_patterns = [
            r"[A-G][b#]?[0-9]?",  # Chord symbols like "Bb7", "C-7", "F7"
            r"[A-G][b#]?[-]?\(",  # Chords with parentheses
        ]
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in notation_patterns):
            # But allow if it also looks like it could be a title (has substantial text)
            # Check if it has more than 2 words that aren't notation
            words = line.split()
            non_notation_words = [w for w in words if not re.match(r"^[A-G][b#]?[-]?[0-9()]*$", w, re.IGNORECASE)]
            if len(non_notation_words) < 2:  # Mostly notation, skip it
                continue

        # Handle all-caps titles FIRST - be more lenient, allow longer titles
        # Many song titles are in all caps, especially in sheet music
        if line.isupper():
            # Allow all-caps if it looks like a title (has spaces, reasonable length)
            if " " in line and 5 <= len(line) <= 60:
                return line
            # Skip very long all-caps lines (likely not titles)
            if len(line) > 60:
                continue
        
        # Handle titles that start with all-caps but have other text (common in OCR)
        # e.g., "WHATEVER LO-LAWANTS, Lo-la gets..."
        # Extract the all-caps prefix if it looks like a title
        # This should be checked BEFORE composer names
        if line and line[0].isupper():
            # Find where the all-caps portion ends (before comma, period, or lowercase)
            all_caps_end = 0
            for i, char in enumerate(line):
                if char.isupper() or char in " -":
                    all_caps_end = i + 1
                elif char in ",.":
                    # Stop at punctuation if we have a reasonable title length
                    if all_caps_end >= 5:
                        break
                    all_caps_end = i + 1
                    break
                elif char.islower():
                    # Stop at first lowercase if we already have a good title
                    if all_caps_end >= 5:
                        break
                    all_caps_end = 0  # Reset if we hit lowercase too early
                    break
                else:
                    break
            
            if all_caps_end >= 5:
                all_caps_part = line[:all_caps_end].strip().rstrip(".,")
                # Check if it's a reasonable title (has spaces, not too long)
                if " " in all_caps_part and len(all_caps_part) <= 60:
                    return all_caps_part

        # Skip lines that look like composer/artist names (2-4 capitalized words)
        # This helps avoid picking "Richard Adler Jerry Ross" over the actual title
        # Check this AFTER checking for all-caps titles
        if composer_pattern.match(line) and 2 <= len(line.split()) <= 4:
            continue

        # Skip lines that are mostly lowercase (likely not titles)
        if line.islower() and len(line) > 20:
            continue

        # This looks like a potential piece name
        return line

    return None


def index_pdf(
    pdf_path: Path, use_ocr: bool = True, min_text_length: int = 3
) -> list[dict[str, int | str]]:
    """Create an index of pieces in a sheet music PDF.

    Args:
        pdf_path: Path to the PDF file to index
        use_ocr: Whether to use OCR for pages with no text
        min_text_length: Minimum text length to consider a piece name

    Returns:
        List of dictionaries with 'name' and 'start_page' keys
    """
    pieces = []
    current_piece = None
    current_piece_start = None

    # Get total number of pages
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
    except Exception as e:
        raise RuntimeError(f"Error opening PDF: {e}")

    for page_num in range(total_pages):
        # Try text extraction first
        text, words_pos = extract_text_from_page(pdf_path, page_num, use_ocr=False)

        # If no text found or very little text, and OCR is enabled, try OCR
        if use_ocr and (not text or len(text.strip()) < min_text_length):
            ocr_text, ocr_words_pos = extract_text_from_page(pdf_path, page_num, use_ocr=True)
            if ocr_text and len(ocr_text.strip()) >= min_text_length:
                text = ocr_text
                words_pos = ocr_words_pos

        # Detect piece name (pass positional info if available)
        piece_name = detect_piece_name(text, min_text_length, words_pos if words_pos else None)

        # Quality check: if the detected name looks like garbage OCR, treat it as unknown
        if piece_name:
            # Check if it looks like garbage (too many short words, brackets, low quality)
            words_in_name = piece_name.split()
            if len(words_in_name) >= 2:
                short_words = sum(1 for w in words_in_name if len(w) <= 2)
                has_brackets = '[' in piece_name or ']' in piece_name
                # Common short words that are valid in titles
                common_short_words = {'a', 'an', 'the', 'is', 'am', 'my', 'to', 'go', 'be', 'do', 'we', 'he', 'she', 'it', 'in', 'on', 'at', 'of', 'or', 'if', 'so', 'no', 'up', 'me', 'us', 'you', 'i'}
                # Count short words that are NOT common words
                uncommon_short_words = sum(1 for w in words_in_name if len(w) <= 2 and w.lower() not in common_short_words)
                # Check for numbers mixed with text (often OCR garbage)
                has_numbers = any(w.isdigit() for w in words_in_name)
                # Check for very short words that are just numbers or symbols
                has_garbage_words = any(len(w) <= 2 and not w.isalpha() and w.lower() not in common_short_words for w in words_in_name)
                # If most words are very short uncommon words, has brackets, or has numbers mixed with garbage, likely garbage
                # But allow if short words are common words (like "Head You Go My To")
                if (uncommon_short_words >= len(words_in_name) * 0.5) or (has_brackets and short_words >= len(words_in_name) * 0.4) or (has_numbers and has_garbage_words):
                    piece_name = None

        # If no piece name detected, use "(Unknown)"
        if not piece_name:
            piece_name = "(Unknown)"

        # If we found a piece name, check if it's a new piece
        if piece_name:
            # Check if this is a new piece (different name or first piece)
            if current_piece is None or piece_name != current_piece:
                # Save previous piece if exists
                if current_piece is not None:
                    pieces.append(
                        {"name": current_piece, "start_page": current_piece_start}
                    )

                # Start new piece
                current_piece = piece_name
                current_piece_start = page_num + 1  # 1-indexed for user
        # If no piece name detected but we have a current piece, it's likely a continuation page
        # Keep the current piece active (don't create a new entry)

    # Don't forget the last piece
    if current_piece is not None:
        pieces.append({"name": current_piece, "start_page": current_piece_start})

    return pieces
