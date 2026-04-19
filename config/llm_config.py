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


def _clean_key(key):
    """Clean key of whitespace and surrounding quotes."""
    if key and isinstance(key, str):
        return key.strip().strip('"').strip("'")
    return key


def get_llm():
    """Returns the best available free-tier LLM with automatic fallback."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    errors = []

    # Try to load secrets from Streamlit (Cloud) or Environment
    groq_key = _clean_key(os.environ.get("GROQ_API_KEY"))
    google_key = _clean_key(os.environ.get("GOOGLE_API_KEY"))

    try:
        import streamlit as st
        # Log available keys (safely) for cloud debugging
        if hasattr(st, "secrets") and st.secrets:
            # Only print key names, never values
            print(f"DEBUG: st.secrets detected with keys: {list(st.secrets.keys())}")
            if not groq_key:
                groq_key = _clean_key(st.secrets.get("GROQ_API_KEY"))
            if not google_key:
                google_key = _clean_key(st.secrets.get("GOOGLE_API_KEY"))
    except Exception as e:
        # Not running in Streamlit or st.secrets unavailable
        pass

    # Try Groq first
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

    # Fallback to Gemini
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
        "Verified GROQ_API_KEY and GOOGLE_API_KEY are missing from environment and st.secrets. "
        "If deployed on Streamlit Cloud, add them in Settings -> Secrets."
    )
