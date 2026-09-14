from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Read PDF
reader = PdfReader("data/my_document.pdf")

text = ""
for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)

print("Total chunks:", len(chunks))

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create embeddings
vectors = embeddings.embed_documents(chunks)

print("Total vectors:", len(vectors))
print("Vector size:", len(vectors[0]))

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst 10 values of its vector:")
print(vectors[0][:10])