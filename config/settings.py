"""
Application-wide constants and configuration.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "kc_house_data.csv")
# Use local path if writable, otherwise fall back to /tmp for cloud deployments
_local_chroma = os.path.join(BASE_DIR, "rag", "chroma_db")
_tmp_chroma = os.path.join("/tmp", "chroma_db")
CHROMA_DB_PATH = _local_chroma if os.access(os.path.dirname(_local_chroma), os.W_OK) else _tmp_chroma
KNOWLEDGE_SOURCES_PATH = os.path.join(BASE_DIR, "rag", "knowledge_sources")

# --- LLM ---
LLM_TEMPERATURE = 0.1  # Low temperature for factual, grounded output
LLM_PRIMARY_MODEL = "llama-3.1-8b-instant"
LLM_FALLBACK_MODEL = "gemini-2.0-flash"
LLM_REQUEST_TIMEOUT_SECONDS = 18

# --- RAG ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "real_estate_knowledge"
RAG_TOP_K = 5
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# --- Agent ---
MAX_SYNTHESIZER_RETRIES = 2
CONFIDENCE_THRESHOLD = 50.0  # Below this, skip market_analyst
VERY_HIGH_RISK_THRESHOLD = 75  # Above this, trigger human review
CHAT_HISTORY_WINDOW = 8

# --- Guardrails / Search ---
MAX_CHAT_QUERY_CHARS = 500
MAX_LISTING_NOTES_CHARS = 1600
WEB_SEARCH_TIMEOUT_SECONDS = 8
WEB_SEARCH_MAX_RESULTS = 6
WEB_SEARCH_CACHE_SIZE = 48
SEARCH_RESULT_CHAR_LIMIT = 3500

# --- Investment Labels ---
INVESTMENT_LABELS = {
    (75, 101): ("STRONG_BUY", "Strong Buy", "#10b981"),
    (60, 75): ("BUY", "Good Buy", "#34d399"),
    (45, 60): ("HOLD", "Moderate", "#f59e0b"),
    (30, 45): ("CAUTION", "Caution", "#f97316"),
    (0, 30): ("AVOID", "Avoid", "#ef4444"),
}


def get_investment_label(score: int) -> tuple[str, str, str]:
    for (low, high), (code, label, color) in INVESTMENT_LABELS.items():
        if low <= score < high:
            return code, label, color
    return "HOLD", "Moderate", "#f59e0b"
