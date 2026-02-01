import uuid
import shutil
from fastapi import APIRouter, UploadFile, File
from src.inference.extractor import InvoiceExtractor

router = APIRouter()
extractor = InvoiceExtractor()

@router.post("/extract")
async def extract_invoice(file: UploadFile = File(...)):
    temp_path = f"/tmp/{uuid.uuid4()}.pdf"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return extractor.process(temp_path)
