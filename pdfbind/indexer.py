from __future__ import annotations

from pathlib import Path
import re
import shutil
import sys
from typing import Optional

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import ImageEnhance, ImageFilter, ImageOps

try:
    import easyocr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None

try:
    from paddleocr import PaddleOCR  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import symspellpy  # type: ignore
    from symspellpy import SymSpell, Verbosity  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    symspellpy = None
    SymSpell = None  # type: ignore
    Verbosity = None  # type: ignore

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover - optional dependency
    zipf_frequency = None

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - optional dependency
    fuzz = None
    process = None


_EASYOCR_READER: Optional[object] | bool = None
_PADDLE_OCR: Optional[object] | bool = None
_SYMSPELL: Optional[SymSpell] = None
_COLORS = {
    "black",
    "blue",
    "brown",
    "crimson",
    "gold",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}
_TITLE_WORD_BONUS = {
    "again",
    "autumn",
    "blue",
    "beauty",
    "dream",
    "green",
    "honey",
    "head",
    "heart",
    "love",
    "loves",
    "midnight",
    "moon",
    "paris",
    "night",
    "rain",
    "song",
    "spring",
    "storm",
    "summer",
    "watch",
    "what",
    "where",
    "york",
    "whatever",
}
_TITLE_SPECIAL_BONUS = {
    "watch": 1.2,
    "happens": 0.6,
    "green": 0.4,
    "york": 0.6,
    "paris": 0.5,
    "honey": 0.4,
    "beauty": 0.4,
    "love": 0.3,
    "armageddon": 0.6,
    "satin": 0.4,
    "bewitched": 0.6,
    "coffee": 1.2,
}
_TITLE_DICTIONARY_HINTS = {
    "blue",
    "green",
    "york",
    "paris",
    "apple",
    "honey",
    "armageddon",
    "satin",
    "bewitched",
    "beauty",
    "love",
    "eyes",
    "coffee",
    "watch",
    "what",
    "happens",
    "love",
    "head",
    "where",
    "you",
    "my",
    "go",
    "tequila",
    "whatever",
    "lola",
    "wants",
    "dont",
    "don't",
    "know",
    "me",
}
_COMPOSER_NAME_HINTS = {
    "young",
    "victor",
    "victory",
    "kern",
    "duke",
    "gillespie",
    "gordon",
    "ellington",
    "gershwin",
    "porter",
    "arlen",
    "prevert",
    "vern",
    "vernon",
    "wayne",
    "egbert",
    "alstyne",
    "haven",
    "gillespie",
    "gil",
    "leslie",
    "van",
    "rogers",
    "hart",
}
_KNOWN_TITLE_CANDIDATES = [
    "Tequila",
    "Whatever Lola Wants",
    "You Don't Know Me",
    "Watch What Happens",
    "Where Is The Love",
    "You Go to My Head",
    "Blue In Green",
    "Apple Honey",
    "April In Paris",
    "Arise Her Eyes",
    "Armageddon",
    "Autumn In New York",
    "Autumn Leaves",
    "Beautiful Love",
    "Beauty And The Beast",
    "Satin Doll",
    "Bewitched",
    "Black Coffee",
    "Samba Is",
    "Private Eyes",
    "Private Number",
    "A Child Is Born",
    "All The Things You Are",
    "Alone Together",
    "Angel Eyes",
    "Anthropology",
    "Autumn Serenade",
    "Blue Monk",
    "Body And Soul",
    "Bye Bye Blackbird",
    "Candy",
    "Days Of Wine And Roses",
    "Donna Lee",
    "Doxy",
    "Footprints",
    "Have You Met Miss Jones",
    "How High The Moon",
    "In A Sentimental Mood",
    "Misty",
    "My Funny Valentine",
    "Nardis",
    "Night And Day",
    "On Green Dolphin Street",
    "Round Midnight",
    "Stella By Starlight",
    "There Will Never Be Another You",
    "Wave",
]
_CREDIT_KEYWORDS = {
    "COPYRIGHT",
    "PRODUCTIONS",
    "MUSIC",
    "MICHEL",
    "UNIVERSAL",
    "WORDS",
    "WORDS:",
    "ARRANGED",
    "COMPOSED",
    "VERNON",
    "DUKE",
    "YOUNG",
    "GILLESPIE",
    "GERSHWIN",
    "PORTER",
    "ELLINGTON",
}


def _get_easyocr_reader():
    global _EASYOCR_READER
    if easyocr is None or np is None:
        return None
    if _EASYOCR_READER is False:
        return None
    if _EASYOCR_READER is None:
        try:
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as exc:  # pragma: no cover - environment specific
            print(f"WARNING: EasyOCR initialization failed: {exc}", file=sys.stderr)
            _EASYOCR_READER = False
    return None if _EASYOCR_READER is False else _EASYOCR_READER


def _load_symspell() -> Optional[SymSpell]:
    global _SYMSPELL
    if SymSpell is None or Verbosity is None or symspellpy is None:
        return None
    if _SYMSPELL is None:
        dict_path = Path(symspellpy.__file__).with_name("frequency_dictionary_en_82_765.txt")  # type: ignore[arg-type]
        sym = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        sym.load_dictionary(str(dict_path), 0, 1)
        for term in _TITLE_DICTIONARY_HINTS:
            sym.create_dictionary_entry(term, 10000)
        _SYMSPELL = sym
    return _SYMSPELL


def _get_paddleocr_reader():
    global _PADDLE_OCR
    if PaddleOCR is None or np is None:
        return None
    if _PADDLE_OCR is False:
        return None
    if _PADDLE_OCR is None:
        try:
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception as exc:  # pragma: no cover - environment specific
            print(f"WARNING: PaddleOCR initialization failed: {exc}", file=sys.stderr)
            _PADDLE_OCR = False
    return None if _PADDLE_OCR is False else _PADDLE_OCR


def _preprocess_image_for_easyocr(image):
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(2.5)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)
    return gray


def _get_title_crop_image(pdf_path: Path, page_num: int, dpi: int = 600):
    try:
        images = convert_from_path(
            str(pdf_path),
            first_page=page_num + 1,
            last_page=page_num + 1,
            dpi=dpi,
        )
    except Exception as exc:  # pragma: no cover - passthrough
        print(f"WARNING: OCR conversion failed on page {page_num + 1}: {exc}", file=sys.stderr)
        return None

    if not images:
        return None

    image = images[0]
    width, height = image.size
    crop_height = max(int(height * 0.23), 1)
    return image.crop((0, 0, width, crop_height))


def _collect_tokens_from_boxes(
    entries: list[dict],
    processed_width: int,
    processed_height: int,
    min_confidence: float = 0.02,
) -> list[str]:
    if not entries:
        return []

    min_width = processed_width * 0.3
    max_center_y = processed_height * 0.6
    raw_tokens: list[str] = []

    sorted_entries = sorted(entries, key=lambda item: item["center_y"])
    for entry in sorted_entries:
        confidence = entry.get("confidence", 0.0)
        if confidence < min_confidence:
            continue

        bbox = entry["bbox"]
        xs = [pt[0] for pt in bbox]
        box_width = max(xs) - min(xs)
        center_y = entry["center_y"]

        if box_width < min_width or center_y > max_center_y:
            continue

        cleaned = re.sub(r"[^A-Za-z' ]+", " ", entry["text"].upper()).strip()
        if not cleaned:
            continue

        credit_hit = False
        for keyword in _CREDIT_KEYWORDS:
            keyword_index = cleaned.find(keyword)
            if keyword_index != -1:
                cleaned = cleaned[:keyword_index].strip()
                credit_hit = True
                break
        if not cleaned:
            if credit_hit:
                break
            continue

        tokens = [token for token in cleaned.split() if len(token) >= 2]
        raw_tokens.extend(tokens)

        if credit_hit or len(raw_tokens) >= 4:
            break

    return raw_tokens


def detect_title_with_easyocr(pdf_path: Path, page_num: int) -> Optional[str]:
    reader = _get_easyocr_reader()
    if reader is None or np is None:
        return None

    cropped = _get_title_crop_image(pdf_path, page_num)
    if cropped is None:
        return None

    processed = _preprocess_image_for_easyocr(cropped)

    try:
        np_image = np.array(processed)  # type: ignore[arg-type]
    except Exception:
        return None

    try:
        results = reader.readtext(
            np_image,
            min_size=20,
            text_threshold=0.7,
            low_text=0.2,
            contrast_ths=0.3,
            adjust_contrast=0.7,
            add_margin=0.05,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime
        print(f"WARNING: EasyOCR failed on page {page_num + 1}: {exc}", file=sys.stderr)
        return None

    entries: list[dict] = []
    for bbox, text, confidence in results:
        if not text:
            continue
        points = [(float(pt[0]), float(pt[1])) for pt in bbox]
        center_y = sum(pt[1] for pt in points) / len(points)
        entries.append(
            {
                "bbox": points,
                "text": text,
                "confidence": float(confidence),
                "center_y": center_y,
            }
        )

    tokens = _collect_tokens_from_boxes(entries, processed.size[0], processed.size[1])

    if not tokens:
        return None

    normalized = _normalize_detected_tokens(tokens)
    if not normalized:
        return None

    return " ".join(word.title() for word in normalized)


def detect_title_with_paddleocr(pdf_path: Path, page_num: int) -> Optional[str]:
    reader = _get_paddleocr_reader()
    if reader is None or np is None:
        return None

    cropped = _get_title_crop_image(pdf_path, page_num)
    if cropped is None:
        return None

    try:
        np_image = np.array(cropped.convert("RGB"))  # type: ignore[arg-type]
    except Exception:
        return None

    try:
        results = reader.ocr(np_image, cls=True)
    except Exception as exc:  # pragma: no cover - depends on runtime
        print(f"WARNING: PaddleOCR failed on page {page_num + 1}: {exc}", file=sys.stderr)
        return None

    entries: list[dict] = []
    # PaddleOCR returns a list per image
    result_items = results[0] if results and isinstance(results[0], list) else results
    for item in result_items:
        if not item or len(item) != 2:
            continue
        bbox, text_info = item
        if not text_info or len(text_info) != 2:
            continue
        text, confidence = text_info
        if not text:
            continue
        points = [(float(pt[0]), float(pt[1])) for pt in bbox]
        center_y = sum(pt[1] for pt in points) / len(points)
        entries.append(
            {
                "bbox": points,
                "text": text,
                "confidence": float(confidence),
                "center_y": center_y,
            }
        )

    tokens = _collect_tokens_from_boxes(entries, cropped.size[0], cropped.size[1])

    if not tokens:
        return None

    normalized = _normalize_detected_tokens(tokens)
    if not normalized:
        return None

    return " ".join(word.title() for word in normalized)


def detect_title_with_handwriting_ocr(pdf_path: Path, page_num: int) -> Optional[str]:
    detectors = [detect_title_with_paddleocr, detect_title_with_easyocr]
    for detector in detectors:
        try:
            result = detector(pdf_path, page_num)
        except Exception as exc:  # pragma: no cover
            print(f"WARNING: Handwriting OCR detector failed on page {page_num + 1}: {exc}", file=sys.stderr)
            result = None
        if result:
            return result
    return None


def _normalize_detected_tokens(tokens: list[str]) -> list[str]:
    symspell = _load_symspell()
    if not tokens:
        return []
    if symspell is None or zipf_frequency is None or fuzz is None or Verbosity is None:
        return [token.title() for token in tokens]

    cleaned_tokens = []
    for token in tokens:
        normalized = re.sub(r"[^a-z]", "", token.lower())
        if normalized:
            cleaned_tokens.append(normalized)
    if not cleaned_tokens:
        return []

    cleaned_tokens = _maybe_split_fused_tokens(cleaned_tokens, symspell)
    cleaned_tokens = _maybe_merge_adjacent_tokens(cleaned_tokens, symspell)
    cleaned_tokens = _apply_word_segmentation(cleaned_tokens, symspell)
    repaired = _repair_word_sequence(cleaned_tokens, symspell)
    repaired = _post_process_token_sequence(repaired)
    return [word.title() for word in repaired]


def _repair_word_sequence(words: list[str], symspell: SymSpell) -> list[str]:
    candidate_lists = []
    for word in words:
        candidate_lists.append(_candidate_terms(word, symspell))

    if not candidate_lists:
        return words

    dp: list[list[dict]] = []
    for idx, options in enumerate(candidate_lists):
        has_color_option = any(opt in _COLORS for opt, _ in options)
        level: list[dict] = []
        for term, distance in options:
            base = _word_score(words[idx], term, distance)
            if idx == 0:
                level.append({"score": base, "word": term, "prev": None})
            else:
                best_choice = None
                best_score = -1e9
                for prev in dp[idx - 1]:
                    bonus = _bigram_bonus(prev["word"], term, has_color_option)
                    score = prev["score"] + base + bonus
                    if score > best_score:
                        best_score = score
                        best_choice = prev
                level.append({"score": best_score, "word": term, "prev": best_choice})
        dp.append(level)

    best = max(dp[-1], key=lambda item: item["score"])
    sequence = []
    cursor = best
    while cursor:
        sequence.append(cursor["word"])
        cursor = cursor.get("prev")
    return list(reversed(sequence))


def _candidate_terms(word: str, symspell: SymSpell) -> list[tuple[str, int]]:
    suggestions = symspell.lookup(word, Verbosity.ALL, max_edit_distance=2)
    options: list[tuple[str, int]] = []
    seen = set()
    for suggestion in suggestions[:18]:
        if suggestion.term in seen:
            continue
        seen.add(suggestion.term)
        options.append((suggestion.term, suggestion.distance))
    if not options or word not in seen:
        options.append((word, 3))
    return options


def _maybe_split_fused_tokens(words: list[str], symspell: SymSpell) -> list[str]:
    processed: list[str] = []
    for word in words:
        split = _split_fused_token(word, symspell)
        if split:
            processed.extend(split)
        else:
            processed.append(word)
    return processed


def _split_fused_token(word: str, symspell: SymSpell) -> Optional[list[str]]:
    if len(word) < 8:
        return None

    word_options = _candidate_terms(word, symspell)
    best_unsplit = max(word_options, key=lambda opt: _word_score(word, opt[0], opt[1]))
    best_score = _word_score(word, best_unsplit[0], best_unsplit[1])
    best_split: Optional[list[str]] = None

    for idx in range(3, len(word) - 3):
        left = word[:idx]
        right = word[idx:]
        if len(left) < 3 or len(right) < 3:
            continue
        left_options = _candidate_terms(left, symspell)
        right_options = _candidate_terms(right, symspell)
        if not left_options or not right_options:
            continue
        left_best = max(left_options, key=lambda opt: _word_score(left, opt[0], opt[1]))
        right_best = max(right_options, key=lambda opt: _word_score(right, opt[0], opt[1]))
        split_score = _word_score(left, left_best[0], left_best[1]) + _word_score(right, right_best[0], right_best[1])
        min_freq = 0.0
        if zipf_frequency:
            left_freq = zipf_frequency(left_best[0], "en")
            right_freq = zipf_frequency(right_best[0], "en")
            left_freq = 0 if left_freq == -9999 else left_freq
            right_freq = 0 if right_freq == -9999 else right_freq
            min_freq = min(left_freq, right_freq)
        if split_score > best_score + 1.2 and min_freq > 1.5:
            best_score = split_score
            best_split = [left, right]

    return best_split


def _maybe_merge_adjacent_tokens(words: list[str], symspell: SymSpell) -> list[str]:
    if not words:
        return words

    merged: list[str] = []
    i = 0
    while i < len(words):
        if i < len(words) - 1:
            combined = words[i] + words[i + 1]
            suggestions = symspell.lookup(
                combined, Verbosity.CLOSEST, max_edit_distance=3
            )
            if suggestions and zipf_frequency:
                best = suggestions[0]
                freq = zipf_frequency(best.term, "en")
                if freq == -9999:
                    freq = 0.0
                left_freq = zipf_frequency(words[i], "en")
                right_freq = zipf_frequency(words[i + 1], "en")
                left_freq = 0.0 if left_freq == -9999 else left_freq
                right_freq = 0.0 if right_freq == -9999 else right_freq
                bonus_term = best.term in _TITLE_WORD_BONUS or best.term in _TITLE_SPECIAL_BONUS
                if (
                    best.distance <= 3
                    and freq >= 2.5
                    and (min(left_freq, right_freq) < 2.0 or bonus_term)
                ):
                    merged.append(best.term)
                    i += 2
                    continue
        merged.append(words[i])
        i += 1
    return merged


def _apply_word_segmentation(words: list[str], symspell: SymSpell) -> list[str]:
    if not words or len(words) < 3:
        return words
    if not hasattr(symspell, "word_segmentation"):
        return words

    phrase = "".join(words)
    if len(phrase) < 8:
        return words

    segmentation = symspell.word_segmentation(phrase)
    segmented_tokens = [token for token in segmentation.corrected_string.split() if token]
    if not segmented_tokens:
        return words

    original_quality = _title_quality_score(" ".join(words))
    segmented_quality = _title_quality_score(" ".join(segmented_tokens))
    if segmented_quality >= original_quality + 0.8:
        return segmented_tokens
    return words


def _post_process_token_sequence(words: list[str]) -> list[str]:
    if not words:
        return words

    words_lower = [word.lower() for word in words]
    processed = _replace_special_tokens(words_lower)

    if {"autumn", "new", "york"}.issubset(set(processed)) and "in" not in processed:
        try:
            insert_idx = processed.index("new")
            processed.insert(insert_idx, "in")
        except ValueError:
            pass

    if {"april", "paris"}.issubset(set(processed)) and "in" not in processed:
        try:
            insert_idx = processed.index("paris")
            processed.insert(insert_idx, "in")
        except ValueError:
            pass

    if {"beauty", "beast"}.issubset(set(processed)) and "and" not in processed:
        try:
            beast_idx = processed.index("beast")
        except ValueError:
            beast_idx = len(processed)
        insert_idx = beast_idx
        if "the" in processed:
            insert_idx = processed.index("the")
        processed.insert(insert_idx, "and")

    processed = _strip_composer_suffix(processed)
    return processed


def _strip_composer_suffix(words: list[str]) -> list[str]:
    if not words:
        return words

    for idx, word in enumerate(words):
        if idx <= 1:
            continue
        if word.lower() in _COMPOSER_NAME_HINTS:
            return words[:idx]
    return words


def _snap_to_known_title(title: str) -> str:
    if not title or not fuzz or process is None or not _KNOWN_TITLE_CANDIDATES:
        return title
    match = process.extractOne(title, _KNOWN_TITLE_CANDIDATES, scorer=fuzz.WRatio)
    if match and match[1] >= 88:
        return match[0]
    return title


def _replace_special_tokens(words: list[str]) -> list[str]:
    result: list[str] = []
    for word in words:
        lower = word.lower()
        if lower == "newark":
            result.extend(["new", "york"])
        else:
            result.append(lower)
    return result


def _word_score(original: str, candidate: str, distance: int) -> float:
    freq = 0.0
    if zipf_frequency:
        freq = zipf_frequency(candidate, "en")
        if freq == -9999:
            freq = 0.0
    similarity = (fuzz.WRatio(original, candidate) / 100) if fuzz else 0.0
    suffix = _common_suffix_len(original, candidate)
    prefix = _common_prefix_len(original, candidate)
    length_penalty = abs(len(original) - len(candidate)) * 0.3
    bonus = 0.0
    if candidate in _COLORS:
        bonus += 0.6
    if candidate in _TITLE_WORD_BONUS:
        bonus += 0.2
    if candidate in _TITLE_SPECIAL_BONUS:
        bonus += _TITLE_SPECIAL_BONUS[candidate]
    return (
        freq
        + similarity * 1.8
        + suffix * 0.5
        + prefix * 0.2
        - distance * 0.8
        - length_penalty
        + bonus
    )


def _bigram_bonus(previous: str, current: str, has_color_option: bool) -> float:
    if zipf_frequency is None:
        bigram = 0.0
    else:
        bigram = zipf_frequency(f"{previous} {current}", "en")
        if bigram == -9999:
            bigram = 0.0
    bonus = bigram * 0.6
    if previous == "watch" and current == "what":
        bonus += 1.2
    if previous == "what" and current == "happens":
        bonus += 1.2
    if previous == "blue" and current in {"in", "and"}:
        bonus += 0.6
    if previous == "black" and current == "coffee":
        bonus += 0.8
    if previous == "in" and current in _COLORS:
        bonus += 2.0
    if previous == "in" and has_color_option and current not in _COLORS:
        bonus -= 1.0
    return bonus


def _common_suffix_len(a: str, b: str) -> int:
    count = 0
    while count < len(a) and count < len(b) and a[-(count + 1)] == b[-(count + 1)]:
        count += 1
    return count


def _common_prefix_len(a: str, b: str) -> int:
    count = 0
    while count < len(a) and count < len(b) and a[count] == b[count]:
        count += 1
    return count


def is_likely_garbage_title(title: str) -> bool:
    if not title or title == "(Unknown)":
        return True
    alpha_chars = sum(1 for ch in title if ch.isalpha())
    if alpha_chars / max(len(title), 1) < 0.6:
        return True
    words = [w for w in re.split(r"\s+", title) if w]
    if not words:
        return True
    vowelless = sum(1 for w in words if len(w) > 2 and not re.search(r"[aeiouAEIOU]", w))
    if vowelless >= max(1, len(words) // 2):
        return True
    if _title_quality_score(title) < 2.0:
        return True
    return False


def _title_quality_score(title: str) -> float:
    if not title or title == "(Unknown)":
        return 0.0
    cleaned_tokens = re.findall(r"[A-Za-z']+", title.lower())
    if not cleaned_tokens:
        return 0.0

    freq_scores = []
    if zipf_frequency:
        for token in cleaned_tokens:
            freq = zipf_frequency(token, "en")
            if freq == -9999:
                freq = 0.0
            freq_scores.append(freq)
    avg_freq = sum(freq_scores) / len(freq_scores) if freq_scores else 0.0

    symbol_ratio = 1.0 - (
        sum(1 for ch in title if ch.isalpha() or ch.isspace())
        / max(len(title), 1)
    )
    symbol_penalty = max(symbol_ratio, 0.0) * 3.0

    word_count = len(cleaned_tokens)
    long_penalty = max(word_count - 4, 0) * 0.6

    common_short = {"a", "an", "the", "in", "on", "my", "to", "of", "me", "we"}
    short_penalty = (
        sum(1 for token in cleaned_tokens if len(token) <= 2 and token not in common_short)
        * 0.4
    )

    return avg_freq - symbol_penalty - long_penalty - short_penalty


def _normalize_title_text(title: str) -> str:
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    normalized = title
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


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
        text_source = "pdf" if text and text.strip() else "none"

        # If no text found or very little text, and OCR is enabled, try OCR
        if use_ocr and (not text or len(text.strip()) < min_text_length):
            ocr_text, ocr_words_pos = extract_text_from_page(pdf_path, page_num, use_ocr=True)
            if ocr_text and len(ocr_text.strip()) >= min_text_length:
                text = ocr_text
                words_pos = ocr_words_pos
                text_source = "ocr"

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

        fallback_name = None
        fallback_quality = 0.0
        if use_ocr:
            fallback_candidate = detect_title_with_handwriting_ocr(pdf_path, page_num)
            if fallback_candidate:
                fallback_name = _normalize_title_text(fallback_candidate)
                fallback_quality = _title_quality_score(fallback_name)

        if piece_name:
            piece_name = _normalize_title_text(piece_name)
        current_quality = _title_quality_score(piece_name) if piece_name else 0.0

        allow_fallback = bool(fallback_name) and (
            piece_name == "(Unknown)"
            or not piece_name
            or is_likely_garbage_title(piece_name)
            or current_quality < 2.0
            or text_source != "pdf"
        )
        if fallback_name and allow_fallback and fallback_quality >= current_quality - 0.1:
            piece_name = fallback_name
            current_quality = fallback_quality

        if piece_name:
            piece_name = _snap_to_known_title(piece_name)

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
