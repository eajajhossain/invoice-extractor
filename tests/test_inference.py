from src.inference.extractor import InvoiceExtractor

def test_extraction_pipeline():
    extractor = InvoiceExtractor()
    result = extractor.process("tests/sample.pdf")
    assert "validation" in result
