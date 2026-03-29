import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents.state import AgentState
from core.llm import get_llm
from tools.retriever import get_retriever_tool

logger = logging.getLogger(__name__)

FACT_CHECKER_PROMPT = """You are a fact-checking specialist. Your job is to search through uploaded documents in the knowledge base to find relevant information.

Instructions:
1. Use the document_search tool to retrieve relevant passages.
2. Assess the relevance and reliability of each retrieved passage.
3. Always cite the source document name and page number when available.
4. If the knowledge base doesn't contain relevant information, clearly state that.
5. Highlight any contradictions or inconsistencies you find."""


def fact_checker_node(state: AgentState) -> dict:
    try:
        return _fact_checker_node(state)
    except Exception as e:
        logger.error("Fact-checker node failed: %s", e, exc_info=True)
        error_msg = f"Document retrieval encountered an error: {e}"
        return {
            "retrieved_context": state.get("retrieved_context", []) + [
                {"query": state.get("research_query", ""), "findings": error_msg}
            ],
            "sources": state.get("sources", []),
            "messages": [AIMessage(content=f"[Fact-Checker] {error_msg}", name="fact_checker")],
        }


def _fact_checker_node(state: AgentState) -> dict:
    llm = get_llm(streaming=False, temperature=0.1)
    retriever_tool = get_retriever_tool()
    llm_with_tools = llm.bind_tools([retriever_tool])

    query = state.get("research_query", "")
    if not query:
        msgs = state.get("messages", [])
        query = msgs[-1].content if msgs else ""

    logger.info("Fact-checker searching knowledge base for: %s", query[:80])

    messages = [
        SystemMessage(content=FACT_CHECKER_PROMPT),
        HumanMessage(content=f"Search the knowledge base for information about: {query}"),
    ]

    # Agent loop
    response = None
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_result = retriever_tool.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            ))

    # Safely extract content — if the loop ended on a tool-call round,
    # response.content may be empty; fall back gracefully.
    content = (getattr(response, "content", "") or "") if response else ""
    if not content:
        logger.warning("Fact-checker produced no content for query: %s", query[:80])
        content = "Document retrieval completed but no relevant passages were found."

    retrieved_context = state.get("retrieved_context", []) + [{"query": query, "findings": content}]
    summary = (content[:200] + "...") if len(content) > 200 else content
    sources = state.get("sources", []) + [{
        "type": "document_retrieval",
        "query": query,
        "summary": summary,
    }]

    logger.info("Fact-checker completed, findings length: %d chars", len(content))

    return {
        "retrieved_context": retrieved_context,
        "sources": sources,
        "messages": [AIMessage(content=f"[Fact-Checker] {content}", name="fact_checker")],
    }
