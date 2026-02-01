class InvoiceValidator:
    """
    Business validation for extracted invoices.
    """

    def validate(self, invoice: dict) -> dict:
        errors = []

        if not invoice.get("invoice_number"):
            errors.append("Missing invoice number")

        total = invoice.get("total")
        if total is None or total <= 0:
            errors.append("Invalid total amount")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }
