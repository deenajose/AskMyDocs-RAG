import os

from dotenv import load_dotenv
from google import genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")


# Connect to Gemini
client = genai.Client(api_key=api_key)


# Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to ChromaDB
vector_store = Chroma(
    collection_name="my_documents",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


def ask_question(question, k=3):
    """
    Retrieve relevant document chunks from ChromaDB
    and ask Gemini to answer using only those chunks.
    """

    # Search ChromaDB
    results = vector_store.similarity_search(
        question,
        k=k
    )

    # Combine retrieved chunks
    context = "\n\n".join(
        document.page_content
        for document in results
    )

    # Create prompt
    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the document."

Context:
{context}

Question:
{question}
"""

    # Ask Gemini
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    return interaction.output_text