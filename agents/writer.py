import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.state import AgentState
from core.llm import get_llm

logger = logging.getLogger(__name__)

WRITER_PROMPT = """You are a report writer. Synthesize all gathered research findings and fact-checked information into a clear, well-structured report.

Instructions:
1. Write a comprehensive yet concise report addressing the user's query.
2. Use inline citations like [1], [2], etc. for claims backed by sources.
3. Structure the report with clear headings using markdown.
4. Include a "Sources" section at the end listing all referenced sources.
5. If research and document findings conflict, note the discrepancy.
6. Be objective and balanced in your analysis.
7. If no research or documents were found, answer based on your knowledge but note the limitation."""


def writer_node(state: AgentState) -> dict:
    temperature = state.get("temperature", 0.3)
    llm = get_llm(streaming=True, temperature=temperature)

    # Build context for the writer
    context_parts = []

    if state.get("search_results"):
        context_parts.append("## Web Research Findings")
        for i, result in enumerate(state["search_results"], 1):
            context_parts.append(f"[{i}] Query: {result['query']}\nFindings: {result['findings']}")

    if state.get("retrieved_context"):
        context_parts.append("\n## Document Knowledge Base Findings")
        offset = len(state.get("search_results", [])) + 1
        for i, ctx in enumerate(state["retrieved_context"], offset):
            context_parts.append(f"[{i}] Query: {ctx['query']}\nFindings: {ctx['findings']}")

    context = "\n\n".join(context_parts) if context_parts else "No research data was gathered."

    query = state.get("research_query", "")
    if not query:
        for msg in state.get("messages", []):
            if hasattr(msg, "type") and msg.type == "human":
                query = msg.content

    messages = [
        SystemMessage(content=WRITER_PROMPT),
        HumanMessage(content=f"User's question: {query}\n\nGathered research:\n{context}\n\nWrite a comprehensive report."),
    ]

    try:
        response = llm.invoke(messages)
        report = response.content or "The report could not be generated."
    except Exception as e:
        logger.error("Writer node LLM call failed: %s", e, exc_info=True)
        report = f"An error occurred while generating the report: {e}"

    logger.info("Writer completed report, length: %d chars", len(report))

    return {
        "report": report,
        "messages": [AIMessage(content=report, name="writer")],
    }
