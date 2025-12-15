import shutil
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.helper import load_pdf, splitting, embeds
from src.pipeline import retrieval

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GLOBAL_VECTORSTORE = None

class ChatRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global GLOBAL_VECTORSTORE
    
    try:
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        documents = load_pdf(tmp_path)
        chunks = splitting(documents)
        
        GLOBAL_VECTORSTORE = embeds(chunks)

        os.remove(tmp_path)

        return {"status": "success", "message": "PDF processed successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat")
def chat(request: ChatRequest):
    global GLOBAL_VECTORSTORE

    if GLOBAL_VECTORSTORE is None:
        raise HTTPException(status_code=400, detail="No PDF loaded. Please upload a file first.")

    try:
        answer = retrieval(GLOBAL_VECTORSTORE, request.query)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")