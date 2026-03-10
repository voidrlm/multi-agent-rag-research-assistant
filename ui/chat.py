import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from agents.graph import get_graph
from ui.components import render_sources
from ui.trace_viewer import render_trace


def _init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "trace_events" not in st.session_state:
        st.session_state.trace_events = []


def render_chat():
    _init_session()

    st.header("Multi-Agent Research Assistant")
    st.caption("Ask a question — agents will research, verify, and write a report.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])
            if msg.get("trace"):
                render_trace(msg["trace"])

    # Chat input
    user_input = st.chat_input("Ask anything...")

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run the agent graph
        with st.chat_message("assistant"):
            trace_events = []
            status_placeholder = st.empty()
            response_placeholder = st.empty()

            graph = get_graph()
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            final_report = ""
            sources = []

            try:
                for event in graph.stream(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "search_results": [],
                        "retrieved_context": [],
                        "sources": [],
                        "report": "",
                        "next_agent": "",
                        "research_query": "",
                        "iteration": 0,
                    },
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, node_output in event.items():
                        if node_name == "__end__":
                            continue

                        # Track trace
                        summary = ""
                        if node_name == "orchestrator":
                            next_agent = node_output.get("next_agent", "")
                            summary = f"Routing to: {next_agent}"
                        elif node_name == "researcher":
                            summary = "Searched the web for information"
                        elif node_name == "fact_checker":
                            summary = "Checked uploaded documents"
                        elif node_name == "writer":
                            summary = "Writing final report"

                        trace_events.append({"node": node_name, "summary": summary})
                        status_placeholder.info(f"🔄 **{node_name.replace('_', ' ').title()}**: {summary}")

                        # Capture final output
                        if node_name == "writer" and node_output.get("report"):
                            final_report = node_output["report"]

                        if node_output.get("sources"):
                            sources = node_output["sources"]

            except Exception as e:
                st.error(f"Error: {e}")
                final_report = f"An error occurred while processing your query: {e}"

            status_placeholder.empty()
            response_placeholder.markdown(final_report)

            if sources:
                render_sources(sources)

            render_trace(trace_events)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_report,
                "sources": sources,
                "trace": trace_events,
            })
