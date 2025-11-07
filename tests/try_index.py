from pathlib import Path

from pdfbind.indexer import check_poppler_installation, index_pdf


check_poppler_installation()

# input = "cache/Ina2025-11-09.pdf"
# input = "cache/p4.pdf"
input = "tests/detection-test.1.pdf"

result = index_pdf(Path(input))
print(result)