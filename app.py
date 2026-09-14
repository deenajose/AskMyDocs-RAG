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


# ==================================================
# STREAMLIT CONFIG
# ==================================================

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📚",
    layout="centered"
)


# ==================================================
# LOAD ENVIRONMENT
# ==================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()


# ==================================================
# GEMINI
# ==================================================

client = genai.Client(
    api_key=api_key
)

MODEL_NAME = "gemini-3.1-flash-lite"


# ==================================================
# STORAGE PATHS
# ==================================================

CHROMA_PATH = "./chroma_db"
REGISTRY_PATH = "./document_registry.json"


# ==================================================
# PAGE
# ==================================================

st.title("📚 Document Q&A — RAG System")

st.write(
    "Upload one or more PDFs and ask questions about their contents."
)


# ==================================================
# EMBEDDINGS
# ==================================================

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


# ==================================================
# FILE HASH
# ==================================================

def get_file_hash(file_bytes):

    return hashlib.sha256(
        file_bytes
    ).hexdigest()


# ==================================================
# DOCUMENT REGISTRY
# ==================================================

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


# ==================================================
# PDF → CHUNKS
# ==================================================

@st.cache_data(show_spinner=False)
def process_pdf(
    file_bytes,
    filename
):

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

        page_chunks = splitter.split_text(
            text
        )

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


# ==================================================
# PERSISTENT CHROMA DATABASE
# ==================================================

@st.cache_resource
def get_vector_store():

    from langchain_chroma import Chroma

    embeddings = get_embeddings()

    return Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=2,
        max_value=8,
        value=4,
        help="More chunks = more context but slower / noisier answers."
    )

    show_timings = st.checkbox(
        "Show timing breakdown",
        value=True
    )

    if st.button("🗑️ Clear all documents & chat"):

        # Remove Chroma database
        if os.path.exists(CHROMA_PATH):

            try:
                shutil.rmtree(CHROMA_PATH)
            except Exception:
                pass

        # Remove registry
        if os.path.exists(REGISTRY_PATH):

            try:
                os.remove(REGISTRY_PATH)
            except Exception:
                pass

        # Clear Streamlit caches
        st.cache_resource.clear()
        st.cache_data.clear()

        # Clear session state
        st.session_state.pop(
            "chat_history",
            None
        )

        st.success(
            "🗑️ All documents cleared."
        )

        st.rerun()


# ==================================================
# LOAD DOCUMENT REGISTRY
# ==================================================

registry = load_registry()


# ==================================================
# PDF UPLOAD
# ==================================================

uploaded_files = st.file_uploader(
    "📄 Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)


# ==================================================
# PROCESS UPLOADED FILES
# ==================================================

if uploaded_files:

    new_files = []

    # ----------------------------------------------
    # CHECK WHICH FILES ARE ALREADY INDEXED
    # ----------------------------------------------

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        filename = uploaded_file.name

        file_hash = get_file_hash(
            file_bytes
        )

        if file_hash not in registry:

            new_files.append(
                (
                    filename,
                    file_bytes,
                    file_hash
                )
            )


    # ==================================================
    # NEW FILES FOUND
    # ==================================================

    if new_files:

        all_new_chunks = []
        all_new_metadatas = []
        successful_files = []

        warnings = []

        extract_start = time.time()


        # ----------------------------------------------
        # EXTRACT + CHUNK ONLY NEW FILES
        # ----------------------------------------------

        for filename, file_bytes, file_hash in new_files:

            chunks, metadatas, pages_with_text, total_pages = process_pdf(
                file_bytes,
                filename
            )


            if pages_with_text == 0:

                warnings.append(
                    f"⚠️ No extractable text found in "
                    f"**{filename}** ({total_pages} pages). "
                    f"It may be a scanned/image-only PDF."
                )

                continue


            if pages_with_text < total_pages:

                warnings.append(
                    f"ℹ️ **{filename}**: only "
                    f"{pages_with_text}/{total_pages} pages "
                    f"had extractable text."
                )


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


        extract_time = (
            time.time()
            - extract_start
        )


        # ----------------------------------------------
        # SHOW WARNINGS
        # ----------------------------------------------

        for warning in warnings:

            st.warning(
                warning
            )


        # ==================================================
        # EMBED ONLY NEW CHUNKS
        # ==================================================

        if all_new_chunks:

            with st.spinner(
                f"🔄 Creating embeddings for "
                f"{len(all_new_chunks)} new chunks..."
            ):

                embed_start = time.time()

                vector_store = get_vector_store()


                # ------------------------------------------
                # UNIQUE CHUNK IDS
                # ------------------------------------------

                chunk_ids = []

                for index in range(
                    len(all_new_chunks)
                ):

                    chunk_ids.append(
                        f"chunk_{int(time.time() * 1000000)}_{index}"
                    )


                # ------------------------------------------
                # ADD NEW CHUNKS
                # ------------------------------------------

                vector_store.add_texts(
                    texts=all_new_chunks,
                    metadatas=all_new_metadatas,
                    ids=chunk_ids
                )


                embed_time = (
                    time.time()
                    - embed_start
                )


            # ------------------------------------------
            # SAVE ONLY SUCCESSFULLY INDEXED FILES
            # ------------------------------------------

            for filename, file_hash, chunk_count in successful_files:

                registry[file_hash] = {
                    "filename": filename,
                    "chunks": chunk_count
                }


            save_registry(
                registry
            )


            # ------------------------------------------
            # SUCCESS
            # ------------------------------------------

            st.success(
                f"✅ Indexed {len(all_new_chunks)} new chunks "
                f"from {len(successful_files)} new file(s). "
                f"Total time: "
                f"{extract_time + embed_time:.2f}s "
                f"(extract {extract_time:.2f}s · "
                f"embed {embed_time:.2f}s)"
            )


        elif warnings:

            st.warning(
                "⚠️ No new readable text was found."
            )


    # ==================================================
    # NO NEW FILES
    # ==================================================

    else:

        st.success(
            "⚡ All uploaded PDFs are already indexed. "
            "No new embeddings were created."
        )


# ==================================================
# INDEXED DOCUMENTS
# ==================================================

if registry:

    st.sidebar.subheader(
        "📚 Indexed Documents"
    )

    for document in registry.values():

        st.sidebar.write(
            f"📄 {document['filename']}"
        )


# ==================================================
# CHAT HISTORY
# ==================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


for past_question, past_answer, past_sources in st.session_state.chat_history:

    with st.chat_message("user"):

        st.write(
            past_question
        )

    with st.chat_message("assistant"):

        st.write(
            past_answer
        )

        if past_sources:

            st.caption(
                " · ".join(
                    past_sources
                )
            )


# ==================================================
# QUESTION
# ==================================================

question = st.chat_input(
    "Ask a question about your document(s)..."
)


# ==================================================
# RAG PIPELINE
# ==================================================

if question:

    # ----------------------------------------------
    # CHECK DOCUMENTS
    # ----------------------------------------------

    if not registry:

        st.warning(
            "⚠️ Please upload at least one PDF first."
        )

        st.stop()


    with st.chat_message("user"):

        st.write(
            question
        )


    # ----------------------------------------------
    # VECTOR STORE
    # ----------------------------------------------

    vector_store = get_vector_store()


    # ==================================================
    # RETRIEVAL
    # ==================================================

    search_start = time.time()

    results = vector_store.similarity_search(
        question,
        k=top_k
    )

    search_time = (
        time.time()
        - search_start
    )


    # ==================================================
    # CONTEXT
    # ==================================================

    context = "\n\n".join(

        f"[Source: {document.metadata.get('source', 'Unknown')}, "
        f"Page {document.metadata.get('page', '?')}]\n"
        f"{document.page_content}"

        for document in results
    )


    # ==================================================
    # CHAT HISTORY
    # ==================================================

    history_text = ""

    if st.session_state.chat_history:

        recent = (
            st.session_state.chat_history[-3:]
        )

        history_text = "\n".join(

            f"Q: {q}\nA: {a}"

            for q, a, _ in recent
        )


    # ==================================================
    # PROMPT
    # ==================================================

    prompt = f"""
You are a document question-answering assistant.

Use ONLY the information provided in the context below.

If the answer is not present in the context, say exactly:

I couldn't find the answer in the document.

Answer concisely and directly.

When useful, mention the source and page where the information came from.

PREVIOUS CONVERSATION:
{history_text or "(none)"}

CONTEXT:
{context}

QUESTION:
{question}
"""


    # ==================================================
    # GEMINI STREAMING
    # ==================================================

    gemini_start = time.time()


    with st.chat_message("assistant"):

        answer_placeholder = st.empty()

        full_answer = ""


        try:

            stream = client.models.generate_content_stream(

                model=MODEL_NAME,

                contents=prompt,

                config=types.GenerateContentConfig(

                    thinking_config=types.ThinkingConfig(
                        thinking_level="minimal"
                    ),

                    max_output_tokens=300
                )
            )


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
            time.time()
            - gemini_start
        )


        # ==================================================
        # SOURCES
        # ==================================================

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


        if source_labels:

            st.caption(
                " · ".join(
                    source_labels
                )
            )


        # ==================================================
        # TIMING
        # ==================================================

        if show_timings:

            st.caption(
                f"⏱️ search {search_time:.2f}s · "
                f"generation {gemini_time:.2f}s"
            )


    # ==================================================
    # SAVE CHAT
    # ==================================================

    st.session_state.chat_history.append(
        (
            question,
            full_answer,
            source_labels
        )
    )