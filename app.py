import hashlib
import io
import json
import os
import shutil
import time

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.1-flash-lite"


# ============================================================
# STORAGE
# ============================================================

CHROMA_PATH = "./chroma_db"
REGISTRY_PATH = "./document_registry.json"


# ============================================================
# TITLE
# ============================================================

st.title("📚 Document Q&A — RAG System")

st.write(
    "Upload one or more PDFs and ask questions about their contents."
)


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={
            "batch_size": 64,
            "normalize_embeddings": True
        }
    )


# ============================================================
# DOCUMENT REGISTRY
# ============================================================

def load_registry():

    if not os.path.exists(REGISTRY_PATH):
        return {}

    try:

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


def save_registry(registry):

    with open(
        REGISTRY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )


# ============================================================
# PDF PROCESSING
# ============================================================

@st.cache_data(show_spinner=False)
def process_pdf(file_bytes, filename):

    reader = PdfReader(
        io.BytesIO(file_bytes)
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = []
    metadatas = []

    pages_with_text = 0

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if not text or not text.strip():
            continue

        pages_with_text += 1

        page_chunks = splitter.split_text(text)

        for chunk in page_chunks:

            chunks.append(chunk)

            metadatas.append(
                {
                    "source": filename,
                    "page": page_number
                }
            )

    return (
        chunks,
        metadatas,
        pages_with_text,
        len(reader.pages)
    )


# ============================================================
# CHROMA VECTOR DATABASE
# ============================================================

@st.cache_resource
def get_vector_store():

    from langchain_chroma import Chroma

    embeddings = get_embeddings()

    return Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=2,
        max_value=8,
        value=4,
        help=(
            "Number of the most relevant chunks "
            "retrieved from all documents."
        )
    )

    show_timings = st.checkbox(
        "Show timing breakdown",
        value=True
    )

    show_retrieved_chunks = st.checkbox(
        "Show retrieved chunks",
        value=False,
        help=(
            "Useful for checking what information "
            "the RAG system retrieved."
        )
    )

    st.divider()

    # --------------------------------------------------------
    # CLEAR DATABASE
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear all documents & chat"
    ):

        if os.path.exists(CHROMA_PATH):

            try:
                shutil.rmtree(CHROMA_PATH)

            except Exception:
                pass

        if os.path.exists(REGISTRY_PATH):

            try:
                os.remove(REGISTRY_PATH)

            except Exception:
                pass

        st.cache_resource.clear()
        st.cache_data.clear()

        st.session_state.pop(
            "chat_history",
            None
        )

        st.success(
            "🗑️ All documents cleared."
        )

        st.rerun()


# ============================================================
# LOAD REGISTRY
# ============================================================

registry = load_registry()


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "📄 Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# INDEX DOCUMENTS
# ============================================================

if uploaded_files:

    all_new_chunks = []
    all_new_metadatas = []

    successful_files = []
    new_files = []

    # --------------------------------------------------------
    # FIND NEW FILES
    # --------------------------------------------------------

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        filename = uploaded_file.name

        file_hash = hashlib.sha256(
            file_bytes
        ).hexdigest()

        if file_hash in registry:

            continue

        new_files.append(
            (
                filename,
                file_bytes,
                file_hash
            )
        )


    # --------------------------------------------------------
    # PROCESS NEW FILES
    # --------------------------------------------------------

    if new_files:

        with st.spinner(
            "📖 Processing uploaded documents..."
        ):

            for (
                filename,
                file_bytes,
                file_hash
            ) in new_files:

                try:

                    (
                        chunks,
                        metadatas,
                        pages_with_text,
                        total_pages
                    ) = process_pdf(
                        file_bytes,
                        filename
                    )

                    if not chunks:

                        st.warning(
                            f"⚠️ No readable text found in "
                            f"{filename}"
                        )

                        continue

                    all_new_chunks.extend(
                        chunks
                    )

                    all_new_metadatas.extend(
                        metadatas
                    )

                    successful_files.append(
                        (
                            filename,
                            file_hash,
                            len(chunks)
                        )
                    )

                except Exception as e:

                    st.error(
                        f"❌ Error processing "
                        f"{filename}: {e}"
                    )


        # ----------------------------------------------------
        # CREATE EMBEDDINGS
        # ----------------------------------------------------

        if all_new_chunks:

            with st.spinner(
                "🧠 Creating document embeddings..."
            ):

                vector_store = get_vector_store()

                chunk_ids = []

                base_id = int(
                    time.time() * 1000000
                )

                for index in range(
                    len(all_new_chunks)
                ):

                    chunk_ids.append(
                        f"chunk_{base_id}_{index}"
                    )


                vector_store.add_texts(
                    texts=all_new_chunks,
                    metadatas=all_new_metadatas,
                    ids=chunk_ids
                )


            # ------------------------------------------------
            # SAVE REGISTRY
            # ------------------------------------------------

            for (
                filename,
                file_hash,
                chunk_count
            ) in successful_files:

                registry[file_hash] = {
                    "filename": filename,
                    "chunks": chunk_count
                }


            save_registry(
                registry
            )


            st.success(
                f"✅ Indexed "
                f"{len(successful_files)} "
                f"new document(s)."
            )


    else:

        if registry:

            st.info(
                "ℹ️ Uploaded documents are already indexed."
            )


# ============================================================
# SHOW INDEXED DOCUMENTS
# ============================================================

with st.sidebar:

    if registry:

        st.divider()

        st.subheader(
            "📚 Indexed Documents"
        )

        for document_info in registry.values():

            st.write(
                f"📄 {document_info['filename']}"
            )


# ============================================================
# CHAT HISTORY
# ============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for (
    past_question,
    past_answer,
    past_sources
) in st.session_state.chat_history:

    with st.chat_message("user"):

        st.write(
            past_question
        )

    with st.chat_message("assistant"):

        st.markdown(
            past_answer
        )

        if past_sources:

            st.caption(
                " · ".join(
                    past_sources
                )
            )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your document(s)..."
)


# ============================================================
# RAG PIPELINE
# ============================================================

if question:

    # --------------------------------------------------------
    # CHECK DOCUMENTS
    # --------------------------------------------------------

    if not registry:

        st.warning(
            "⚠️ Please upload at least one PDF first."
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY QUESTION
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(
            question
        )


    # --------------------------------------------------------
    # LOAD VECTOR STORE
    # --------------------------------------------------------

    vector_store = get_vector_store()


    # ========================================================
    # STEP 1 — RETRIEVAL
    # ========================================================

    search_start = time.time()

    # Search ALL indexed document chunks together.
    #
    # This is important.
    #
    # We do NOT search each PDF separately.
    #
    # Chroma ranks the chunks globally according to
    # their semantic similarity to the question.

    results = vector_store.similarity_search(
        question,
        k=top_k
    )

    search_time = (
        time.time() - search_start
    )


    # ========================================================
    # DEBUG — SHOW RETRIEVED CHUNKS
    # ========================================================

    if show_retrieved_chunks:

        with st.expander(
            "🔍 Retrieved chunks"
        ):

            if results:

                for (
                    index,
                    document
                ) in enumerate(
                    results,
                    start=1
                ):

                    source = document.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = document.metadata.get(
                        "page",
                        "Unknown"
                    )

                    st.markdown(
                        f"### Chunk {index}"
                    )

                    st.caption(
                        f"📄 {source} — p.{page}"
                    )

                    st.write(
                        document.page_content
                    )

                    st.divider()

            else:

                st.write(
                    "No chunks were retrieved."
                )


    # ========================================================
    # STEP 2 — BUILD CONTEXT
    # ========================================================

    context_parts = []

    for document in results:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "?"
        )

        content = document.page_content

        context_parts.append(
            f"[Source: {source}, Page {page}]\n"
            f"{content}"
        )


    context = "\n\n".join(
        context_parts
    )


    # ========================================================
    # STEP 3 — CONVERSATION HISTORY
    # ========================================================

    history_text = ""

    if st.session_state.chat_history:

        recent = (
            st.session_state.chat_history[-3:]
        )

        history_text = "\n".join(
            f"Q: {q}\nA: {a}"
            for q, a, _ in recent
        )


    # ========================================================
    # STEP 4 — RAG PROMPT
    # ========================================================

    prompt = f"""
You are a document question-answering assistant.

Your job is to answer the user's question using the
retrieved document context.

IMPORTANT RULES:

1. Use the retrieved document context as the factual
   source for your answer.

2. Do NOT invent information.

3. Do NOT use general world knowledge if the answer
   cannot be found in the retrieved context.

4. If the answer is clearly present in the retrieved
   context, answer it directly.

5. If the answer is not present in the retrieved context,
   say exactly:

I couldn't find the answer in the document.

6. When possible, mention the document name and page
   number containing the answer.

7. Previous conversation is ONLY for understanding
   follow-up questions. It is NOT evidence for factual
   answers.

PREVIOUS CONVERSATION:
{history_text or "(none)"}

RETRIEVED DOCUMENT CONTEXT:
{context or "(no relevant context retrieved)"}

USER QUESTION:
{question}
"""


    # ========================================================
    # STEP 5 — GEMINI GENERATION
    # ========================================================

    gemini_start = time.time()

    with st.chat_message("assistant"):

        answer_placeholder = st.empty()

        full_answer = ""

        try:

            stream = (
                client.models.generate_content_stream(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                        max_output_tokens=300
                    )
                )
            )


            # ------------------------------------------------
            # STREAM ANSWER
            # ------------------------------------------------

            for chunk in stream:

                if chunk.text:

                    full_answer += chunk.text

                    answer_placeholder.markdown(
                        full_answer + "▌"
                    )


            answer_placeholder.markdown(
                full_answer or "_(empty response)_"
            )


        except Exception as e:

            st.error(
                f"❌ Gemini error: {e}"
            )

            st.stop()


        gemini_time = (
            time.time() - gemini_start
        )


    # ========================================================
    # STEP 6 — SOURCE INFORMATION
    # ========================================================

    seen_sources = set()

    source_labels = []


    for document in results:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        source_key = (
            source,
            page
        )


        if source_key not in seen_sources:

            seen_sources.add(
                source_key
            )

            source_labels.append(
                f"📄 {source} — p.{page}"
            )


    # ========================================================
    # DISPLAY SOURCES
    # ========================================================

    if source_labels:

        st.caption(
            " · ".join(
                source_labels
            )
        )


    # ========================================================
    # DISPLAY TIMINGS
    # ========================================================

    if show_timings:

        st.caption(
            f"⏱️ search {search_time:.2f}s · "
            f"generation {gemini_time:.2f}s"
        )


    # ========================================================
    # SAVE CHAT
    # ========================================================

    st.session_state.chat_history.append(
        (
            question,
            full_answer,
            source_labels
        )
    )

