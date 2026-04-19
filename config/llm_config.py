"""
LLM provider configuration — Groq (primary) with Google Gemini (fallback).
Both are free-tier. Temperature kept low for factual output.
"""

import os
from config.settings import (
    LLM_TEMPERATURE,
    LLM_PRIMARY_MODEL,
    LLM_FALLBACK_MODEL,
    LLM_REQUEST_TIMEOUT_SECONDS,
)

_llm_instance = None


def _with_timeout(factory, kwargs: dict):
    """Instantiate a LangChain chat model with timeout when supported."""
    for timeout_key in ("request_timeout", "timeout"):
        try:
            return factory(**kwargs, **{timeout_key: LLM_REQUEST_TIMEOUT_SECONDS})
        except TypeError:
            continue
    return factory(**kwargs)


def get_llm():
    """Returns the best available free-tier LLM with automatic fallback."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    errors = []

    # Try Groq first (Llama 3.1 8B — extremely fast, 30 RPM free)
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        try:
            import streamlit as st
            groq_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass

    if groq_key:
        try:
            from langchain_groq import ChatGroq
            _llm_instance = _with_timeout(
                ChatGroq,
                {
                    "model": LLM_PRIMARY_MODEL,
                    "temperature": LLM_TEMPERATURE,
                    "api_key": groq_key,
                },
            )
            return _llm_instance
        except Exception as e:
            errors.append(f"Groq init failed: {e}")

    # Fallback: Google Gemini Flash (15 RPM free)
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    if not google_key:
        try:
            import streamlit as st
            google_key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            pass

    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            _llm_instance = _with_timeout(
                ChatGoogleGenerativeAI,
                {
                    "model": LLM_FALLBACK_MODEL,
                    "temperature": LLM_TEMPERATURE,
                    "google_api_key": google_key,
                },
            )
            return _llm_instance
        except Exception as e:
            errors.append(f"Gemini init failed: {e}")

    error_details = "; ".join(errors) if errors else "No API keys provided"
    raise RuntimeError(
        f"No LLM available. {error_details}. "
        "Set GROQ_API_KEY or GOOGLE_API_KEY in environment variables or .streamlit/secrets.toml"
    )
