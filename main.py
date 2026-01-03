from fastapi import FastAPI, UploadFile, File
import pdfplumber
import requests

app = FastAPI()

# Ton URL de Webhook n8n
N8N_WEBHOOK_URL = "https://hindh.app.n8n.cloud/webhook-test/341ec845-4197-49d6-bb15-b78785bc9e6f"

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    # Lire le PDF
    with pdfplumber.open(file.file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    # Préparer les données à envoyer à n8n
    payload = {
        "filename": file.filename,
        "text": text
    }

    # Envoyer à n8n via POST
    response = requests.post(N8N_WEBHOOK_URL, json=payload)

    return {"filename": file.filename, "status_sent_to_n8n": response.status_code}
