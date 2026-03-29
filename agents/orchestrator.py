import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.state import AgentState
from core.llm import get_llm

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    next_agent: str = Field(
        description="The next agent to route to: 'researcher', 'fact_checker', 'writer', or 'FINISH'"
    )
    reasoning: str = Field(description="Brief explanation of routing decision")


ORCHESTRATOR_PROMPT = """You are a supervisor orchestrating a research assistant team.
Your job is to analyze the user's query and the current state, then decide which agent should act next.

Available agents:
- **researcher**: Searches the web for current, external information. Use when the query needs fresh data, news, or information not in uploaded documents.
- **fact_checker**: Searches through uploaded documents in the knowledge base. Use when the query can be answered from ingested PDFs/URLs.
- **writer**: Synthesizes all gathered research and context into a final, well-structured report with citations. Route here when sufficient information has been gathered.

Rules:
1. If the user's question needs external/web information, route to 'researcher' first.
2. If uploaded documents are relevant, route to 'fact_checker'.
3. You can route to both researcher and fact_checker across iterations to gather comprehensive info.
4. Once you have enough context (search results AND/OR retrieved documents), route to 'writer'.
5. If a final report has already been written, return 'FINISH'.
6. For simple conversational messages (greetings, thanks), route directly to 'writer'.

Current iteration: {iteration} (max {max_iterations} — if iteration >= {max_iterations}, you MUST route to 'writer')"""


def orchestrator_node(state: AgentState) -> dict:
    iteration = state.get("iteration", 0) + 1
    max_iterations = state.get("max_iterations", 3)

    if iteration > max_iterations:
        logger.info("Max iterations reached (%d), forcing writer", max_iterations)
        return {"next_agent": "writer", "iteration": iteration}

    # Check if report is already written
    if state.get("report"):
        return {"next_agent": "FINISH", "iteration": iteration}

    # Extract the original human query — walk backwards to find the last HumanMessage
    # so later AIMessage additions don't overwrite the research query.
    research_query = state.get("research_query", "")
    if not research_query:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                research_query = msg.content
                break
        else:
            msgs = state.get("messages", [])
            research_query = msgs[-1].content if msgs else ""

    llm = get_llm(streaming=False, temperature=0)
    structured_llm = llm.with_structured_output(RouteDecision)

    messages = [
        SystemMessage(content=ORCHESTRATOR_PROMPT.format(
            iteration=iteration,
            max_iterations=max_iterations,
        )),
        *state["messages"],
    ]

    # Add context summary if available
    context_parts = []
    if state.get("search_results"):
        context_parts.append(f"Web search has returned {len(state['search_results'])} result(s).")
    if state.get("retrieved_context"):
        context_parts.append(f"Document retrieval found {len(state['retrieved_context'])} relevant chunk(s).")
    if context_parts:
        messages.append(AIMessage(content="Current state: " + " ".join(context_parts)))

    try:
        decision = structured_llm.invoke(messages)
        next_agent = decision.next_agent
        logger.info("Orchestrator routing to '%s' (iter %d/%d): %s",
                    next_agent, iteration, max_iterations, decision.reasoning)
    except Exception as e:
        logger.warning("Orchestrator LLM call failed (%s), falling back to writer", e)
        next_agent = "writer"

    return {
        "next_agent": next_agent,
        "iteration": iteration,
        "research_query": research_query,
    }
