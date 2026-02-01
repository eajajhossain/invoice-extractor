from pydantic import BaseModel
from typing import Optional

class InvoiceResponse(BaseModel):
    invoice_number: Optional[str]
    vendor: Optional[str]
    total: Optional[float]
    validation: dict
