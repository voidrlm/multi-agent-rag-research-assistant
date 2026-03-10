from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from core.llm import get_llm
from tools.retriever import get_retriever_tool

FACT_CHECKER_PROMPT = """You are a fact-checking specialist. Your job is to search through uploaded documents in the knowledge base to find relevant information.

Instructions:
1. Use the document_search tool to retrieve relevant passages.
2. Assess the relevance and reliability of each retrieved passage.
3. Always cite the source document name and page number when available.
4. If the knowledge base doesn't contain relevant information, clearly state that.
5. Highlight any contradictions or inconsistencies you find."""


def fact_checker_node(state: AgentState) -> dict:
    llm = get_llm(streaming=False, temperature=0.1)
    retriever_tool = get_retriever_tool()
    llm_with_tools = llm.bind_tools([retriever_tool])

    query = state.get("research_query", "")
    if not query and state.get("messages"):
        query = state["messages"][-1].content

    messages = [
        SystemMessage(content=FACT_CHECKER_PROMPT),
        HumanMessage(content=f"Search the knowledge base for information about: {query}"),
    ]

    # Agent loop
    for _ in range(3):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_result = retriever_tool.invoke(tool_call["args"])
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            ))

    # Extract results
    retrieved_context = state.get("retrieved_context", [])
    sources = state.get("sources", [])

    retrieved_context.append({
        "query": query,
        "findings": response.content,
    })

    sources.append({
        "type": "document_retrieval",
        "query": query,
        "summary": response.content[:200],
    })

    return {
        "retrieved_context": retrieved_context,
        "sources": sources,
        "messages": [AIMessage(
            content=f"[Fact-Checker] {response.content}",
            name="fact_checker",
        )],
    }
