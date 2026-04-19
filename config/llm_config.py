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

# --- Loud Cloud Diagnostics ---
def _cloud_diag():
    print("\n" + "="*50)
    print("LOUD DEBUG: Starting LLM Configuration Discovery")
    # Check Environment
    env_keys = [k for k in os.environ.keys() if "GROQ" in k or "GOOGLE" in k or "API_KEY" in k]
    print(f"LOUD DEBUG: Relevant Environment Keys: {env_keys}")
    
    # Check Streamlit Secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets:
            sec_keys = list(st.secrets.keys())
            print(f"LOUD DEBUG: st.secrets detected with keys: {sec_keys}")
            # Check for common variants
            for k in ["groq_api_key", "google_api_key", "Groq_Api_Key"]:
                if k in sec_keys:
                    print(f"LOUD DEBUG: FOUND VARIANT KEY: {k}")
        else:
            print("LOUD DEBUG: st.secrets is EMPTY or UNAVAILABLE")
    except Exception as e:
        print(f"LOUD DEBUG: st.secrets error: {e}")
    print("="*50 + "\n")

_cloud_diag()


def _get_secret(name):
    """Robustly retrieve a secret from environment or Streamlit."""
    val = os.environ.get(name)
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(name)
            # Try lowercase variant if uppercase fails
            if not val:
                val = st.secrets.get(name.lower())
        except:
            pass
    return _clean_key(val)


def get_llm():
    """Returns the best available free-tier LLM with automatic fallback."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    errors = []

    # Try to load secrets
    groq_key = _get_secret("GROQ_API_KEY")
    google_key = _get_secret("GOOGLE_API_KEY")

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
