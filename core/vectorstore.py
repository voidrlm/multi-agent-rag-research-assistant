from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import get_settings
from core.embeddings import get_embeddings


def get_vectorstore(collection_name: str = "research_docs") -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def get_retriever(k: int = 5) -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
