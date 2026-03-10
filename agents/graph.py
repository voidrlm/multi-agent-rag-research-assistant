from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.fact_checker import fact_checker_node
from agents.orchestrator import orchestrator_node
from agents.researcher import researcher_node
from agents.state import AgentState
from agents.writer import writer_node
from core.memory import get_checkpointer


def _route_next(state: AgentState) -> str:
    return state.get("next_agent", "writer")


def build_graph() -> CompiledStateGraph:
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("writer", writer_node)

    # Entry point
    workflow.add_edge(START, "orchestrator")

    # Orchestrator routes conditionally
    workflow.add_conditional_edges(
        "orchestrator",
        _route_next,
        {
            "researcher": "researcher",
            "fact_checker": "fact_checker",
            "writer": "writer",
            "FINISH": END,
        },
    )

    # Sub-agents loop back to orchestrator
    workflow.add_edge("researcher", "orchestrator")
    workflow.add_edge("fact_checker", "orchestrator")
    workflow.add_edge("writer", END)

    return workflow.compile(checkpointer=get_checkpointer())


@lru_cache
def get_graph() -> CompiledStateGraph:
    return build_graph()
