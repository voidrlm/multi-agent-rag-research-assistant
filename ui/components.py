import streamlit as st


def render_sources(sources: list[dict]):
    if not sources:
        return

    st.markdown("---")
    st.markdown("**Sources**")

    for i, source in enumerate(sources, 1):
        source_type = source.get("type", "unknown")
        icon = "🌐" if source_type == "web_search" else "📄"
        query = source.get("query", "")
        summary = source.get("summary", "")

        with st.expander(f"{icon} [{i}] {source_type}: {query[:60]}"):
            st.markdown(summary)


def render_document_card(doc: dict):
    source = doc.get("source", "Unknown")
    chunks = doc.get("chunks", 0)
    st.markdown(f"**{source}** — {chunks} chunks")
