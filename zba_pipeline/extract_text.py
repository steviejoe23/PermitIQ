import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import sys

# Set path to Tesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


def ocr_page(page, dpi=300):
    """Convert PDF page to image and run OCR"""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes()))

    try:
        text = pytesseract.image_to_string(img)
    except Exception as e:
        print(f"OCR ERROR: {e}")
        text = ""

    return text


def extract_pdf_text(pdf_path, verbose=False):
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    full_text = ""

    # Detect if document is scanned
    sample_text = ""
    for i in range(min(5, num_pages)):
        sample_text += doc[i].get_text("text")

    is_scanned = len(sample_text.strip()) < 100
    mode = "OCR" if is_scanned else "HYBRID"

    print(f"  [{mode}] {num_pages} pages — {pdf_path}")
    sys.stdout.flush()

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        text_length = len(text.strip())

        if is_scanned:
            # Scanned doc → OCR every page at 300 DPI
            text = ocr_page(page, dpi=300)
        elif text_length < 50:
            # Mixed doc with weak text → OCR fallback
            ocr_text = ocr_page(page, dpi=300)
            if len(ocr_text.strip()) > text_length:
                text = ocr_text

        # Safety retry (if still empty)
        if len(text.strip()) == 0:
            print(f"    ⚠️ Empty page {page_num} → retrying OCR at 400 DPI")
            sys.stdout.flush()
            text = ocr_page(page, dpi=400)

        full_text += "\n" + text

        # Progress indicator every 25 pages
        if (page_num + 1) % 25 == 0 or page_num == num_pages - 1:
            print(f"    ... page {page_num + 1}/{num_pages}")
            sys.stdout.flush()

    doc.close()
    print(f"  → {len(full_text)} chars extracted")
    sys.stdout.flush()

    return full_text