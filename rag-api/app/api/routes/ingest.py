from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ingest_service import ingest_pdf

router = APIRouter()


@router.post("/upload", summary="Upload a PDF document for ingestion")
def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF file to be parsed, chunked, embedded, and stored
    in the vector database for future querying.
    """
    allowed_extensions = (".pdf", ".docx", ".doc", ".txt")
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported formats: {', '.join(allowed_extensions)}"
        )

    file_bytes = file.file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = ingest_pdf(file_bytes=file_bytes, filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    return {"status": "success", "data": result}
