# api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from cv_pipeline import load_cv_text, run_pipeline
import os

app = FastAPI(title="CV Analyzer API")

@app.get("/")
def root():
    return {"message": "CV Analyzer API is running!"}

@app.post("/analyze_cv")
async def analyze_cv(file: UploadFile = File(...)):
    # Sauvegarder temporairement
    temp_path = f"temp_{file.filename}"
    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        cv_text = load_cv_text(temp_path)
        result = run_pipeline(cv_text)
    finally:
        os.remove(temp_path)  # nettoyage

    return JSONResponse(content=result)
