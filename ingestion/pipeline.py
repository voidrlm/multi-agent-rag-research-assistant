from langchain_core.documents import Document

from core.vectorstore import get_vectorstore
from ingestion.chunker import chunk_documents


def ingest_documents(docs: list[Document]) -> int:
    chunks = chunk_documents(docs)
    if not chunks:
        return 0
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)


def get_ingested_sources() -> list[dict]:
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    result = collection.get(include=["metadatas"])

    sources: dict[str, int] = {}
    for meta in result["metadatas"]:
        source = meta.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    return [{"source": s, "chunks": c} for s, c in sources.items()]


def delete_source(source_name: str) -> None:
    vectorstore = get_vectorstore()
    collection = vectorstore._collection
    collection.delete(where={"source": source_name})
