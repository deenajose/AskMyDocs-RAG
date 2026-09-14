from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from pathlib import Path
import hashlib
import io

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.rag_pipeline import vector_store, ask_question


app = FastAPI(title="RAG Document Q&A API")


DATA_DIR = Path("api_data")
DATA_DIR.mkdir(exist_ok=True)


def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


@app.get("/")
def home():
    return {
        "message": "RAG API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        return {
            "error": "Only PDF files are supported."
        }

    file_bytes = await file.read()

    file_hash = get_file_hash(file_bytes)

    # Save PDF
    pdf_path = DATA_DIR / file.filename

    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    # Read PDF
    reader = PdfReader(io.BytesIO(file_bytes))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file.filename,
                        "page": page_number,
                        "file_hash": file_hash
                    }
                )
            )

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        return {
            "error": "No readable text was found in the PDF."
        }

    # Create unique IDs for the chunks
    ids = [
        f"{file_hash}_{i}"
        for i in range(len(chunks))
    ]

    # Store chunks + embeddings in ChromaDB
    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    return {
        "message": "PDF uploaded and indexed successfully!",
        "filename": file.filename,
        "pages": len(documents),
        "chunks": len(chunks),
        "file_hash": file_hash
    }


@app.get("/ask")
def ask(question: str):

    result = ask_question(question, k=3)

    return result