import re


class InvoicePostProcessor:
    """
    Convert NER entities into structured invoice fields.
    """

    def process(self, entities: list[dict], raw_text: str) -> dict:
        invoice = {
            "invoice_number": None,
            "vendor": None,
            "total": None
        }

        for ent in entities:
            label = ent["label"]
            token = ent["token"]

            if label == "INVOICE_NUMBER" and not invoice["invoice_number"]:
                invoice["invoice_number"] = token

            elif label == "TOTAL" and invoice["total"] is None:
                invoice["total"] = self._parse_amount(token)

            elif label == "VENDOR" and not invoice["vendor"]:
                invoice["vendor"] = token

        return invoice

    @staticmethod
    def _parse_amount(value: str) -> float | None:
        value = re.sub(r"[^0-9.]", "", value)
        try:
            return float(value)
        except ValueError:
            return None
