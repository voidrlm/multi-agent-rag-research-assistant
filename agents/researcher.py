import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents.state import AgentState
from core.llm import get_llm
from tools.search import get_search_tool

logger = logging.getLogger(__name__)

RESEARCHER_PROMPT = """You are a research specialist. Your job is to search the web for relevant, current information about the user's query.

Instructions:
1. Use the search tool to find relevant information.
2. Analyze the search results and extract key findings.
3. Always note the source URL for each piece of information.
4. Focus on factual, verifiable information.
5. Summarize your findings clearly and concisely."""


def researcher_node(state: AgentState) -> dict:
    try:
        return _researcher_node(state)
    except Exception as e:
        logger.error("Researcher node failed: %s", e, exc_info=True)
        error_msg = f"Web research encountered an error: {e}"
        return {
            "search_results": state.get("search_results", []) + [
                {"query": state.get("research_query", ""), "findings": error_msg}
            ],
            "sources": state.get("sources", []),
            "messages": [AIMessage(content=f"[Researcher] {error_msg}", name="researcher")],
        }


def _researcher_node(state: AgentState) -> dict:
    llm = get_llm(streaming=False, temperature=0.2)
    search_tool = get_search_tool()
    llm_with_tools = llm.bind_tools([search_tool])

    query = state.get("research_query", "")
    if not query:
        msgs = state.get("messages", [])
        query = msgs[-1].content if msgs else ""

    logger.info("Researcher searching for: %s", query[:80])

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"Research the following topic: {query}"),
    ]

    # Agent loop: let the LLM decide when to call tools
    response = None
    for _ in range(3):  # max 3 tool-call rounds
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_result = search_tool.invoke(tool_call["args"])
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            ))

    # Safely extract content — if the loop ended on a tool-call round,
    # response.content may be empty; fall back gracefully.
    content = (getattr(response, "content", "") or "") if response else ""
    if not content:
        logger.warning("Researcher produced no content for query: %s", query[:80])
        content = "Research completed but no summary was generated."

    search_results = state.get("search_results", []) + [{"query": query, "findings": content}]
    summary = (content[:200] + "...") if len(content) > 200 else content
    sources = state.get("sources", []) + [{
        "type": "web_search",
        "query": query,
        "summary": summary,
    }]

    logger.info("Researcher completed, findings length: %d chars", len(content))

    return {
        "search_results": search_results,
        "sources": sources,
        "messages": [AIMessage(content=f"[Researcher] {content}", name="researcher")],
    }
