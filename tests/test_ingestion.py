from langchain_core.documents import Document

from ingestion.chunker import chunk_documents


def test_chunk_documents_splits_long_text():
    doc = Document(
        page_content="word " * 500,
        metadata={"source": "test.pdf"},
    )
    chunks = chunk_documents([doc])
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"


def test_chunk_documents_preserves_short_text():
    doc = Document(
        page_content="Short text.",
        metadata={"source": "test.pdf"},
    )
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


def test_chunk_documents_empty_list():
    chunks = chunk_documents([])
    assert chunks == []
