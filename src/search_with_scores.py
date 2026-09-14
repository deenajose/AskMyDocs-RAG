from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to our existing database
vector_store = Chroma(
    collection_name="my_documents",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

# 3. User's question
question = "What is the population situation in Korea?"

# 4. Search with similarity scores
results = vector_store.similarity_search_with_score(
    question,
    k=3
)

# 5. Display results
print("\nQuestion:", question)

for i, (document, score) in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print("Similarity score:", score)
    print("Content:")
    print(document.page_content)