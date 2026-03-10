import streamlit as st

AGENT_STYLES = {
    "orchestrator": {"icon": "🎯", "color": "#FF6B6B"},
    "researcher": {"icon": "🔍", "color": "#4ECDC4"},
    "fact_checker": {"icon": "✅", "color": "#45B7D1"},
    "writer": {"icon": "✍️", "color": "#96CEB4"},
}


def render_trace(events: list[dict]):
    if not events:
        return

    with st.expander("Agent Trace", expanded=False):
        for i, event in enumerate(events):
            node_name = event.get("node", "unknown")
            style = AGENT_STYLES.get(node_name, {"icon": "⚙️", "color": "#999"})

            col1, col2 = st.columns([1, 8])
            with col1:
                st.markdown(f"### {style['icon']}")
            with col2:
                st.markdown(f"**Step {i + 1}: {node_name.replace('_', ' ').title()}**")
                if event.get("summary"):
                    st.caption(event["summary"])

            if i < len(events) - 1:
                st.markdown("<div style='border-left: 2px solid #ccc; height: 20px; margin-left: 24px;'></div>",
                            unsafe_allow_html=True)
