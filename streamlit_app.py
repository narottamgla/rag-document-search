import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
import wikipedia
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import WikipediaAPIWrapper

from src.document_injestion.document_processing import DocumentProcessing


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_document_search")
WIKIPEDIA_FALLBACK_RELEVANCE_THRESHOLD = 0.4

load_dotenv()
os.environ.setdefault("USER_AGENT", "rag-document-search/1.0")
wikipedia.set_user_agent(os.environ["USER_AGENT"])

no_proxy_hosts = {
    host.strip()
    for value in (os.environ.get("no_proxy", ""), os.environ.get("NO_PROXY", ""))
    for host in value.split(",")
    if host.strip()
}
no_proxy_hosts.update({"en.wikipedia.org", ".wikipedia.org"})
os.environ["NO_PROXY"] = ",".join(sorted(no_proxy_hosts))
os.environ["no_proxy"] = os.environ["NO_PROXY"]

st.set_page_config(
    page_title="RAG Document Search",

    page_icon="🧪",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    [data-testid="stSidebar"] {background-color: #f7f8fa;}
    .source-card {
        background-color: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vector_store(documents: List[Document], chunk_size: int, chunk_overlap: int) -> Tuple[Optional[Chroma], List[Document]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    logger.info("Split %d source documents into %d chunks", len(documents), len(chunks))
    add_activity(f"Split documents into {len(chunks)} chunks")

    if not chunks:
        logger.warning("No chunks were created; vector store was not built")
        return None, []

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    logger.info("Vector store created successfully with %d chunks", len(chunks))
    add_activity(f"Vector store ready with {len(chunks)} chunks")
    return vector_store, chunks


def load_uploaded_files(uploaded_files, processor):
    documents = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = Path(temp_dir) / uploaded_file.name
            file_path.write_bytes(uploaded_file.getvalue())
            logger.info("Uploaded file saved for processing: %s", uploaded_file.name)

            if file_path.suffix.lower() == ".pdf":
                loaded_documents = processor.load_from_pdf(file_path)
            elif file_path.suffix.lower() == ".txt":
                loaded_documents = processor.load_from_txt(file_path)
            else:
                continue

            for document in loaded_documents:
                document.metadata["source"] = uploaded_file.name

            documents.extend(loaded_documents)
            logger.info(
                "Loaded %s: %d document section(s)",
                uploaded_file.name,
                len(loaded_documents),
            )

    logger.info("Document upload and loading completed: %d file(s), %d section(s)", len(uploaded_files), len(documents))
    add_activity(f"Loaded {len(documents)} section(s) from {len(uploaded_files)} file(s)")
    return documents


def answer_question(question, vector_store, api_key, model_name, result_count):
    logger.info(
        "Fetching up to %d document chunk(s) from vector store for query: %s",
        result_count,
        question,
    )
    retrieval_started = time.perf_counter()
    try:
        scored_docs = vector_store.similarity_search_with_relevance_scores(
            question,
            k=result_count,
        )
    except Exception:
        logger.exception("Vector-store retrieval failed")
        raise
    retrieval_duration = time.perf_counter() - retrieval_started
    retrieved_docs = [document for document, _ in scored_docs]
    best_relevance = max((score for _, score in scored_docs), default=0.0)

    sources = [
        document.metadata.get("source", "Unknown source")
        for document in retrieved_docs
    ]
    logger.info(
        "Vector-store retrieval completed in %.3f seconds: %d chunk(s), best relevance %.3f; sources: %s",
        retrieval_duration,
        len(retrieved_docs),
        best_relevance,
        ", ".join(sources),
    )
    add_activity(f"Retrieved {len(retrieved_docs)} chunk(s) for the query")

    for index, document in enumerate(retrieved_docs, start=1):
        logger.info(
            "Retrieved chunk %d: source=%s, metadata=%s, content_length=%d",
            index,
            sources[index - 1],
            document.metadata,
            len(document.page_content),
        )

    wikipedia_docs = []
    wikipedia_error = None
    if best_relevance < WIKIPEDIA_FALLBACK_RELEVANCE_THRESHOLD:
        logger.info(
            "Best vector-store match is below %.2f; searching Wikipedia",
            WIKIPEDIA_FALLBACK_RELEVANCE_THRESHOLD,
        )
        try:
            wikipedia_docs = WikipediaAPIWrapper(
                top_k_results=3,
                lang="en",
                doc_content_chars_max=4000,
            ).load(question)
            logger.info("Wikipedia returned %d document(s)", len(wikipedia_docs))
            add_activity(f"Wikipedia returned {len(wikipedia_docs)} result(s)")
        except Exception as exc:
            logger.exception("Wikipedia fallback search failed")
            wikipedia_error = str(exc)
            add_activity(f"Wikipedia fallback failed: {exc}")

    all_context_docs = retrieved_docs + wikipedia_docs
    context = "\n\n".join(
        f"Source {index} ({doc.metadata.get('source', 'Unknown source')}): "
        f"{doc.page_content}"
        for index, doc in enumerate(all_context_docs, start=1)
    )

    prompt = f"""You are a test-case generation assistant.
Use the uploaded knowledge-base sources as the primary source for answers.
If they do not contain enough information, use Wikipedia sources when provided.
If the available context is insufficient, say so clearly. Do not invent facts.

Context:
{context}

Question:
{question}

Provide a clear, practical answer."""

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )
    answer = llm.invoke(prompt)

    if wikipedia_error:
        answer_text = (
            f"{answer.content}\n\n"
            f"*Wikipedia lookup failed; this answer uses uploaded documents only "
            f"({wikipedia_error}).*"
        )
    elif wikipedia_docs:
        answer_text = f"{answer.content}\n\n*Additional context retrieved from Wikipedia.*"
    else:
        answer_text = answer.content

    return answer_text, all_context_docs


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "activity_logs" not in st.session_state:
    st.session_state.activity_logs = []


def add_activity(message: str):
    st.session_state.activity_logs.insert(0, message)
    st.session_state.activity_logs = st.session_state.activity_logs[:20]


with st.sidebar:
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="The key is used only for this session.",
    )

    model_name = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    st.divider()
    st.subheader("Document settings")

    chunk_size = st.slider("Chunk size", 200, 1500, 500, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 50, 25)
    result_count = st.slider("Retrieved sources", 1, 8, 4)

    st.divider()
    st.subheader("Knowledge base")

    if st.session_state.indexed_files:
        st.caption("Indexed files")
        for file_name in st.session_state.indexed_files:
            st.write(f"• {file_name}")


st.title("📚 RAG Document Search")
st.caption("Search, analyze, and generate test cases from uploaded documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT documents",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="document_uploader",
)

if st.button("Build knowledge base", type="primary"):
    if not uploaded_files:
        st.warning("Choose at least one PDF or TXT document to continue.")
    else:
        with st.spinner("Processing documents and creating embeddings..."):
            try:
                processor = DocumentProcessing(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                documents = load_uploaded_files(uploaded_files, processor)

                if not documents:
                    st.error("No text could be extracted from the uploaded files.")
                else:
                    vector_store, chunks = build_vector_store(
                        documents,
                        chunk_size,
                        chunk_overlap,
                    )

                    if vector_store is None:
                        st.error("The documents did not produce any text chunks.")
                    else:
                        st.session_state.vector_store = vector_store
                        st.session_state.indexed_files = [
                            getattr(file, "name", str(file))
                            for file in uploaded_files
                        ]
                        st.success(
                            f"Indexed {len(chunks)} chunks from "
                            f"{len(documents)} document sections."
                        )
                        add_activity(f"Indexing complete: {len(chunks)} chunks")
                        logger.info("Indexing completed successfully")

            except Exception as exc:
                logger.exception("Document processing failed")
                add_activity(f"Document processing failed: {exc}")
                st.error(f"Document processing failed: {exc}")

if not st.session_state.indexed_files:
    st.info("Upload one or more documents above, then build the knowledge base.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved sources"):
                for index, source in enumerate(message["sources"], start=1):
                    title = source.metadata.get("source", f"Source {index}")
                    st.markdown(
                        f"<div class='source-card'><strong>{title}</strong><br>"
                        f"{source.page_content[:1200]}</div>",
                        unsafe_allow_html=True,
                    )

if question := st.chat_input("Ask about requirements or request test cases..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    sources = []
    if st.session_state.vector_store is None:
        answer = "Please upload and index documents before asking a question."
        logger.warning("Question received before a vector store was built")
    elif not api_key:
        answer = "Enter an OpenAI API key in the sidebar to generate an answer."
        logger.warning("Question received without an OpenAI API key")
    else:
        with st.spinner("Searching requirements and generating a response..."):
            try:
                answer, sources = answer_question(
                    question,
                    st.session_state.vector_store,
                    api_key,
                    model_name,
                    result_count,
                )
            except Exception as exc:
                logger.exception("Could not generate a response")
                answer = f"Could not generate a response: {exc}"

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("View retrieved sources"):
                for index, source in enumerate(sources, start=1):
                    title = source.metadata.get("source", f"Source {index}")
                    st.markdown(
                        f"<div class='source-card'><strong>{title}</strong><br>"
                        f"{source.page_content[:1200]}</div>",
                        unsafe_allow_html=True,
                    )

if st.session_state.activity_logs:
    st.sidebar.subheader("Activity logs")
    for log in st.session_state.activity_logs:
        st.sidebar.markdown(f"• {log}")
