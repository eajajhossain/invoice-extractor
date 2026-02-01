import pytesseract
from pdf2image import convert_from_path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Convert PDF → images → OCR text
    """
    pages = convert_from_path(pdf_path)
    text = []

    for page in pages:
        text.append(pytesseract.image_to_string(page))

    return "\n".join(text)
