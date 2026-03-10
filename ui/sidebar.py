import uuid

import streamlit as st

from ingestion.loader import load_pdf, load_url
from ingestion.pipeline import delete_source, get_ingested_sources, ingest_documents


def render_sidebar():
    with st.sidebar:
        st.title("Research Assistant")
        st.markdown("---")

        # New conversation
        if st.button("New Conversation", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.trace_events = []
            st.rerun()

        st.markdown("---")

        # Document upload
        st.subheader("Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("Ingest PDFs"):
                with st.spinner("Processing documents..."):
                    total_chunks = 0
                    for file in uploaded_files:
                        docs = load_pdf(file, file.name)
                        count = ingest_documents(docs)
                        total_chunks += count
                    st.success(f"Ingested {total_chunks} chunks from {len(uploaded_files)} file(s)")

        # URL input
        url_input = st.text_input("Or enter a URL")
        if url_input and st.button("Ingest URL"):
            with st.spinner("Loading URL..."):
                docs = load_url(url_input)
                count = ingest_documents(docs)
                st.success(f"Ingested {count} chunks from URL")

        st.markdown("---")

        # Ingested documents
        st.subheader("Knowledge Base")
        sources = get_ingested_sources()

        if sources:
            for source in sources:
                col1, col2 = st.columns([3, 1])
                with col1:
                    name = source["source"]
                    display = name[:30] + "..." if len(name) > 30 else name
                    st.text(f"{display} ({source['chunks']} chunks)")
                with col2:
                    if st.button("🗑️", key=f"del_{source['source']}"):
                        delete_source(source["source"])
                        st.rerun()
        else:
            st.caption("No documents ingested yet.")

        st.markdown("---")

        # Settings
        st.subheader("Settings")
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        st.session_state.max_iterations = st.slider("Max Agent Iterations", 1, 5, 3)
