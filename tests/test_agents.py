from agents.graph import build_graph
from agents.state import AgentState


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_agent_state_has_required_fields():
    state: AgentState = {
        "messages": [],
        "research_query": "",
        "search_results": [],
        "retrieved_context": [],
        "report": "",
        "sources": [],
        "next_agent": "",
        "iteration": 0,
    }
    assert "messages" in state
    assert "next_agent" in state
    assert state["iteration"] == 0
