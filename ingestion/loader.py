import tempfile
from datetime import datetime, timezone
from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document


def load_pdf(file: BytesIO, filename: str) -> list[Document]:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file.read())
        tmp.flush()
        loader = PyPDFLoader(tmp.name)
        docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source": filename,
            "filename": filename,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
    return docs


def load_url(url: str) -> list[Document]:
    loader = WebBaseLoader(url)
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source": url,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
    return docs
