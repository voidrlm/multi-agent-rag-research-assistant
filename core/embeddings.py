from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
