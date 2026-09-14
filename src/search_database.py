from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to our existing ChromaDB
vector_store = Chroma(
    collection_name="my_documents",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

# 3. Ask a question
question = "What is the population situation in Korea?"

# 4. Search for the 3 most relevant chunks
results = vector_store.similarity_search(
    question,
    k=3
)

# 5. Display the results
print("\nQuestion:", question)

for i, result in enumerate(results):
    print(f"\n--- Relevant Chunk {i + 1} ---")
    print(result.page_content)