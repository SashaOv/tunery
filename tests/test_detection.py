from pathlib import Path
from pdfbind.indexer import index_pdf

# Expected results for tests/detection-test.1.pdf
EXPECTED_RESULT_1 = [
    {"name": "Tequila", "start_page": 1},
    {"name": "Whatever Lola Wants", "start_page": 2},
    {"name": "You Don't Know Me", "start_page": 3},  # Note: OCR uses curly apostrophe (U+2019)
    {"name": "Watch What Happens", "start_page": 4},  # May be "(Unknown)" if OCR fails
    {"name": "Where Is The Love", "start_page": 5},
    {"name": "You Go to My Head", "start_page": 6},
    {"name": "Blue In Green", "start_page": 7},  # May be "(Unknown)" if OCR fails
]


def test_index_pdf_detection():
    """Test that index_pdf() correctly detects titles from detection-test.1.pdf"""
    pdf_path = Path(__file__).parent / "detection-test.1.pdf"
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"Test PDF not found: {pdf_path}")
    
    result = index_pdf(pdf_path)
    
    # Compare lists directly
    assert result == EXPECTED_RESULT_1, \
        f"Expected {EXPECTED_RESULT_1}, got {result}"


if __name__ == "__main__":
    test_index_pdf_detection()
    print("✓ All tests passed!")
