from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    research_query: str
    search_results: list[dict]
    retrieved_context: list[dict]
    report: str
    sources: list[dict]
    next_agent: str
    iteration: int
    max_iterations: int
    temperature: float
