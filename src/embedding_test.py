from langchain_huggingface import HuggingFaceEmbeddings

# Create the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Convert a sentence into a vector
vector = embeddings.embed_query("What is machine learning?")

# Display the result
print("Vector length:", len(vector))
print("First 10 numbers:", vector[:10])