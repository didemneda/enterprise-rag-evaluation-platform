import streamlit as st

from core.config import Settings
from core.pipeline import RAGPipeline


@st.cache_resource
def get_pipeline(settings: Settings) -> RAGPipeline:
    """Return the single pipeline shared by every Streamlit page."""
    return RAGPipeline(settings)
