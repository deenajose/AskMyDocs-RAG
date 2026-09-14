from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Read the PDF
reader = PdfReader("data/my_document.pdf")


# 2. Split each page into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = []
metadatas = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if not text:
        continue

    page_chunks = text_splitter.split_text(text)

    for chunk in page_chunks:
        chunks.append(chunk)

        metadatas.append({
            "source": "my_document.pdf",
            "page": page_number
        })


print("Total chunks:", len(chunks))


# 3. Create the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Connect to ChromaDB
vector_store = Chroma(
    collection_name="my_documents",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


# 5. Give every chunk a unique ID
chunk_ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]


# 6. Store chunks + metadata in ChromaDB
vector_store.add_texts(
    texts=chunks,
    metadatas=metadatas,
    ids=chunk_ids
)


print("Chunks and page information successfully stored!")