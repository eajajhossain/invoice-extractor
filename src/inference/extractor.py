from src.models.ner_model import NERModel
from src.inference.postprocessor import InvoicePostProcessor
from src.inference.validator import InvoiceValidator
from src.utils.ocr import extract_text_from_pdf


class InvoiceExtractor:
    """
    End-to-end invoice processing pipeline.
    """

    def __init__(self):
        self.ner = NERModel()
        self.postprocessor = InvoicePostProcessor()
        self.validator = InvoiceValidator()

    def process(self, pdf_path: str) -> dict:
        text = extract_text_from_pdf(pdf_path)

        entities = self.ner.predict(text)
        structured = self.postprocessor.process(entities, text)

        structured["validation"] = self.validator.validate(structured)
        return structured
