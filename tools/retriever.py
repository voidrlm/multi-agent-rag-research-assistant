from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import BaseTool

from core.vectorstore import get_retriever


def get_retriever_tool() -> BaseTool:
    retriever = get_retriever()
    return create_retriever_tool(
        retriever,
        name="document_search",
        description=(
            "Search through uploaded documents in the knowledge base. "
            "Use this to find relevant passages, facts, and information "
            "from PDFs and web pages that have been ingested. "
            "Input should be a search query string."
        ),
    )
