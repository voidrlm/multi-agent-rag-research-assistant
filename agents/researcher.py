from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from core.llm import get_llm
from tools.search import get_search_tool

RESEARCHER_PROMPT = """You are a research specialist. Your job is to search the web for relevant, current information about the user's query.

Instructions:
1. Use the search tool to find relevant information.
2. Analyze the search results and extract key findings.
3. Always note the source URL for each piece of information.
4. Focus on factual, verifiable information.
5. Summarize your findings clearly and concisely."""


def researcher_node(state: AgentState) -> dict:
    llm = get_llm(streaming=False, temperature=0.2)
    search_tool = get_search_tool()
    llm_with_tools = llm.bind_tools([search_tool])

    query = state.get("research_query", "")
    if not query and state.get("messages"):
        query = state["messages"][-1].content

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"Research the following topic: {query}"),
    ]

    # Agent loop: let the LLM decide when to call tools
    for _ in range(3):  # max 3 tool-call rounds
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_result = search_tool.invoke(tool_call["args"])
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            ))

    # Extract results
    search_results = state.get("search_results", [])
    sources = state.get("sources", [])

    search_results.append({
        "query": query,
        "findings": response.content,
    })

    sources.append({
        "type": "web_search",
        "query": query,
        "summary": response.content[:200],
    })

    return {
        "search_results": search_results,
        "sources": sources,
        "messages": [AIMessage(
            content=f"[Researcher] {response.content}",
            name="researcher",
        )],
    }
