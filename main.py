from fastapi import FastAPI, UploadFile, File
import pdfplumber

app = FastAPI()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Le fichier doit être un PDF."}

    content = ""
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                content += text + "\n"

    return {"filename": file.filename, "text": content}
