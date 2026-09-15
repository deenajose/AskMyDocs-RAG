# 📚 AskMyDocs — Document Q&A RAG System

AskMyDocs is an end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and ask questions about their contents.

Instead of relying only on an LLM's general knowledge, the system retrieves relevant information from the uploaded documents and uses **Google Gemini** to generate a grounded answer with document and page references.

## 🚀 Live Demo

**Streamlit Cloud:** Add your deployed URL here

## ✨ Features

- 📄 Upload one or multiple PDF documents
- 🔎 Semantic search across uploaded documents
- 🧩 Automatic document chunking
- 🧠 Hugging Face sentence-transformer embeddings
- 🗄️ ChromaDB vector database
- 🤖 Google Gemini-powered answer generation
- 📑 Source document and page references
- 💬 Conversational question-answering interface
- ⚡ Retrieval and generation timing information
- ♻️ Duplicate document detection using SHA-256 hashing
- 🌐 Streamlit web interface
- 🔌 FastAPI backend with REST API endpoints
- ☁️ Streamlit Cloud deployment support

## 🧠 How the RAG System Works

The project follows this pipeline:

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
ChromaDB Vector Storage
    ↓
User Question
    ↓
Semantic Similarity Search
    ↓
Relevant Document Chunks
    ↓
Context + Question
    ↓
Google Gemini
    ↓
Grounded Answer + Sources
```

### 1. Document Upload

The user uploads one or more PDF documents through the Streamlit interface.

### 2. Text Extraction

The application extracts readable text from each PDF using `pypdf`.

### 3. Chunking

Large documents are divided into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.

Current configuration:

```text
Chunk size: 800 characters
Chunk overlap: 120 characters
```

Chunk overlap helps preserve context between neighboring chunks.

### 4. Embeddings

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings allow the system to compare the semantic meaning of the user's question with the document content.

### 5. Vector Database

The generated embeddings are stored in **ChromaDB**.

ChromaDB allows the system to efficiently retrieve chunks that are semantically similar to a user's question.

### 6. Retrieval

When the user asks a question, the application performs a similarity search and retrieves the most relevant document chunks.

### 7. Generation

The retrieved chunks are provided to **Google Gemini** as context.

The model is instructed to answer using the retrieved document information and avoid inventing information that is not supported by the retrieved context.

### 8. Answer + Sources

The final response is displayed together with the relevant document name and page number whenever available.

---

## 🏗️ Project Architecture

```text
                    ┌──────────────────┐
                    │    PDF Upload    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Text Extraction │
                    │     (pypdf)      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Chunking     │
                    │    (LangChain)   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Embeddings    │
                    │  Hugging Face    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    ChromaDB      │
                    │  Vector Storage  │
                    └────────┬─────────┘
                             │
                      User Question
                             │
                             ▼
                    ┌──────────────────┐
                    │ Semantic Search  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Relevant Context │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Google Gemini  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Answer + Sources │
                    └──────────────────┘
```

## 🛠️ Technology Stack

| Technology            | Purpose                                |
| --------------------- | -------------------------------------- |
| Python                | Core programming language              |
| Streamlit             | Web application interface              |
| FastAPI               | REST API backend                       |
| LangChain             | Document processing and RAG components |
| ChromaDB              | Vector database                        |
| Hugging Face          | Text embeddings                        |
| Sentence Transformers | Semantic embedding model               |
| Google Gemini         | LLM for answer generation              |
| pypdf                 | PDF text extraction                    |
| Uvicorn               | FastAPI development server             |
| python-dotenv         | Environment variable management        |

## 📁 Project Structure

```text
AskMyDocs-RAG/
│
├── app.py                  # Streamlit web application
├── api.py                  # FastAPI backend
├── main.py                 # Project entry point / supporting script
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and folders
├── .env                    # Local API keys (not committed)
│
├── src/
│   ├── rag_pipeline.py     # Core RAG pipeline
│   └── gemini_test.py      # Gemini API testing
│
├── data/                   # Local PDF/document data
├── chroma_db/              # Local ChromaDB storage
├── api_data/               # Files handled by API
└── document_registry.json  # Local document registry
```

> `.env`, local document data, ChromaDB storage, and other runtime files should not be committed to GitHub.

## ⚙️ Local Installation

### 1. Clone the repository

```powershell
git clone https://github.com/deenajose/AskMyDocs-RAG.git
cd AskMyDocs-RAG
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Configure the Gemini API key

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

The API key should never be committed to GitHub.

### 6. Run the Streamlit application

```powershell
streamlit run app.py
```

The application will open in the browser.

## 🔌 FastAPI API

The project also provides a FastAPI interface.

Start the API with:

```powershell
uvicorn api:app --reload
```

Available endpoints include:

```text
GET  /
GET  /health
POST /upload
POST /ask
```

FastAPI automatically provides interactive API documentation at:

```text
/docs
```

## ☁️ Deployment

The Streamlit application is designed to run on **Streamlit Community Cloud**.

For cloud deployment:

1. Push the project to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Select `app.py` as the application entry point.
4. Add `GEMINI_API_KEY` through Streamlit's secrets configuration.
5. Deploy the application.

### Cloud Storage Note

The cloud version uses an in-memory ChromaDB configuration so that the application does not depend on the local persistent ChromaDB directory.

This means uploaded documents are not intended to be permanent across application restarts.

For a production-scale system, persistent cloud storage or a managed vector database would be preferable.

## 🔐 Security

Sensitive configuration is kept outside the source code.

The Gemini API key is stored using:

```text
.env
```

for local development and should be stored using the platform's secret-management system when deployed.

The `.env` file is excluded from Git using `.gitignore`.

## 🎯 Why RAG Instead of a Normal Chatbot?

A general-purpose chatbot may generate an answer from its pretrained knowledge.

AskMyDocs follows a different approach:

```text
User Question
      ↓
Search the user's documents
      ↓
Retrieve relevant information
      ↓
Give that information to the LLM
      ↓
Generate an answer grounded in the documents
```

This makes the system useful for **private, domain-specific documents** where the user expects answers based on their own content.

Examples include:

- Academic papers
- Research documents
- Company documentation
- Manuals
- Reports
- Notes
- Policies
- Study materials

## 📌 Current Limitations

- PDF text extraction works best with text-based PDFs.
- Scanned/image-only PDFs require OCR support.
- Cloud document storage is not persistent across application restarts.
- Retrieval quality depends on chunking and embedding quality.
- The current system does not include authentication or user accounts.

## 🔮 Future Improvements

Possible future improvements include:

- 🔐 User authentication
- 💾 Persistent cloud vector database
- 📦 Cloud object storage for uploaded documents
- 🖼️ OCR support for scanned PDFs
- 🎯 Retrieval reranking
- 📊 RAG evaluation metrics
- 🗑️ Individual document deletion
- 📈 Improved monitoring and logging
- ⚡ Further response-time optimization
- 👥 Multi-user document isolation

## 👩‍💻 Project Goal

The goal of AskMyDocs is to demonstrate a practical implementation of an end-to-end **Retrieval-Augmented Generation system**, combining document processing, vector search, embeddings, LLM generation, API development, and cloud deployment into a single application.

---

## ⭐ Key Learning Outcomes

Through this project, I implemented and worked with:

- Retrieval-Augmented Generation (RAG)
- Semantic search
- Vector embeddings
- Vector databases
- Document chunking
- Prompt engineering
- LLM integration
- PDF processing
- REST API development
- Streamlit application development
- Environment and secret management
- Git and GitHub
- Cloud deployment
