import streamlit as st

from ui.chat import render_chat
from ui.sidebar import render_sidebar

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()
render_chat()
