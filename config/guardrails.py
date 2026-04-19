"""
Guardrail System — 3-layer defense for the property chatbot.
Layer 1: Topic classifier (is it real estate?)
Layer 2: Prompt injection detector
Layer 3: Output validator
"""

import re
import unicodedata
from config.settings import (
    MAX_CHAT_QUERY_CHARS,
    MAX_LISTING_NOTES_CHARS,
    SEARCH_RESULT_CHAR_LIMIT,
)

# ─── Layer 1: Topic Classifier ───

REAL_ESTATE_KEYWORDS = {
    # Property types
    "property", "house", "home", "apartment", "condo", "condominium", "townhouse",
    "duplex", "mansion", "bungalow", "villa", "cottage", "studio", "loft", "flat",
    "residential", "commercial", "land", "lot", "plot", "estate", "dwelling",
    # Transactions
    "buy", "sell", "rent", "lease", "mortgage", "loan", "refinance", "closing",
    "listing", "offer", "bid", "appraisal", "inspection", "escrow", "deed",
    # Pricing
    "price", "cost", "value", "worth", "valuation", "assessment", "estimate",
    "affordable", "expensive", "cheap", "premium", "budget", "median",
    # Features
    "bedroom", "bathroom", "sqft", "square feet", "square foot", "living area",
    "lot size", "garage", "basement", "attic", "pool", "patio", "deck",
    "waterfront", "view", "grade", "condition", "renovation", "remodel",
    "kitchen", "floor", "story", "stories",
    # Location
    "neighborhood", "zipcode", "zip code", "county", "city", "suburb", "downtown",
    "district", "school district", "walkability", "transit", "commute",
    # Market
    "market", "real estate", "housing", "investment", "roi", "appreciation",
    "depreciation", "equity", "foreclosure", "hoa", "property tax",
    "comparable", "comps", "arv", "cap rate",
    # Agents
    "realtor", "agent", "broker", "mls",
}

# Greeting patterns that should be allowed through
GREETING_PATTERNS = [
    r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy|greetings)\b",
    r"^(thanks|thank you|ok|okay|sure|yes|no|got it|understood)\b",
    r"^(help|what can you do|how do you work|what do you know)\b",
]


def is_real_estate_query(query: str) -> tuple[bool, str]:
    """Check if a query is related to real estate topics.
    Returns (is_allowed, rejection_reason).
    """
    query_lower = query.lower().strip()

    # Allow greetings and meta-questions
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, query_lower):
            return True, ""

    # Check for real estate keywords
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    # Also check 2-word phrases
    query_bigrams = set()
    words = query_lower.split()
    for i in range(len(words) - 1):
        query_bigrams.add(f"{words[i]} {words[i+1]}")

    all_terms = query_words | query_bigrams

    matches = all_terms & REAL_ESTATE_KEYWORDS
    if matches:
        return True, ""

    return False, (
        "I can only help with real estate and property price questions. "
        "Please ask about property valuations, housing markets, price estimates, "
        "or investment analysis."
    )


# ─── Layer 2: Prompt Injection Detector ───

INJECTION_PATTERNS = [
    # Role manipulation
    r"ignore\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions|rules|prompts)",
    r"disregard\s+(your|all|the)\s+(instructions|rules|guidelines|prompts)",
    r"forget\s+(your|all|the)\s+(instructions|rules|guidelines|training)",
    r"you\s+are\s+now\s+(a|an|my)",
    r"act\s+as\s+(a|an|if)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"role\s*play\s+as",
    r"switch\s+to\s+.+\s+mode",
    r"enter\s+.+\s+mode",
    # Prompt extraction
    r"(show|reveal|display|print|output|repeat|tell\s+me)\s+(your|the)\s+(system\s+)?prompt",
    r"what\s+(are|is)\s+your\s+(instructions|rules|system\s+prompt|guidelines)",
    r"(show|display|print)\s+your\s+(instructions|configuration)",
    # Jailbreak patterns
    r"\bjailbreak\b",
    r"\bDAN\s+(mode|prompt|jailbreak)",  # avoid false positive on name "Dan"
    r"developer\s+mode",
    r"unrestricted\s+mode",
    r"bypass\s+(safety|filter|restriction|guardrail)",
    r"override\s+(safety|rules|restrictions|instructions)",
    r"ignore\s+all\s+rules",
    r"no\s+restrictions",
    r"without\s+(any\s+)?restrictions",
    # Code execution attempts
    r"(execute|run|eval)\s+(this|the|my)\s+(code|script|command|program)",
    r"write\s+(me\s+)?(a|some|the)\s+(code|script|program|function)",
    r"```\s*(python|javascript|bash|sql|sh)",
    # Data exfiltration
    r"(send|email|post|upload)\s+(this|the|my)\s+(data|information|conversation)",
    r"(what|list)\s+(other\s+)?users?\s+(asked|said|data)",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


_HOMOGLYPH_MAP = str.maketrans({
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0443': 'y', '\u0445': 'x', '\u0456': 'i',
    '\u0410': 'A', '\u0415': 'E', '\u041e': 'O', '\u0420': 'P',
    '\u0421': 'C', '\u0423': 'Y', '\u0425': 'X', '\u0406': 'I',
    '\u0222': 'O', '\u2010': '-', '\u2011': '-', '\u2012': '-',
    '\u2013': '-', '\u2014': '-',
})


def _normalize_text(text: str) -> str:
    """Unicode normalize, replace homoglyphs, strip zero-width chars."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff\u00ad]', '', text)
    text = text.translate(_HOMOGLYPH_MAP)
    return text


def detect_injection(query: str) -> tuple[bool, str]:
    """Detect prompt injection attempts.
    Returns (is_injection, detected_pattern_description).
    """
    # Normalize Unicode before checking patterns (prevents homoglyph bypass)
    query = _normalize_text(query)

    for pattern in _compiled_patterns:
        if pattern.search(query):
            return True, (
                "Your message was flagged by our safety system. "
                "I'm designed to help with real estate questions only. "
                "Please rephrase your question about property prices or markets."
            )

    # Check for excessive special characters (encoding attacks)
    special_ratio = sum(1 for c in query if not c.isalnum() and c not in " .,?!'-$%/") / max(len(query), 1)
    if special_ratio > 0.3:
        return True, (
            "Your message contains unusual formatting. "
            "Please ask your real estate question in plain text."
        )

    return False, ""


# ─── Layer 3: Output Validator ───

OUTPUT_BLOCK_PATTERNS = [
    r"```\s*(python|javascript|bash|sql|sh|java|cpp|ruby)",  # Code blocks
    r"(password|api[_\s]?key|secret[_\s]?key|token)\s*[:=]",  # Credentials
    r"(DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO|UPDATE\s+.+\s+SET)",  # SQL injection
    r"<script[^>]*>",  # XSS
    r"(system|developer)\s+(prompt|message|instructions)",
    r"(hidden|private)\s+(chain\s+of\s+thought|reasoning|instructions)",
]

_compiled_output_patterns = [re.compile(p, re.IGNORECASE) for p in OUTPUT_BLOCK_PATTERNS]

SENSITIVE_DATA_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[redacted-ssn]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[redacted-card]"),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[redacted-email]"),
    (re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?)\d{3}[ -]?\d{4}\b"), "[redacted-phone]"),
]

ADVISORY_TEXT_FIELDS = ("objective", "raw_notes")


def redact_sensitive_text(text: str) -> str:
    """Redact common sensitive values before text is shown to an LLM."""
    cleaned = _normalize_text(str(text or ""))
    for pattern, replacement in SENSITIVE_DATA_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def sanitize_plaintext(text: str, max_chars: int = 1000) -> str:
    """Normalize and strip unsafe formatting while preserving readable text."""
    cleaned = redact_sensitive_text(text)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    cleaned = re.sub(r"[<>{}`]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def validate_response(response: str) -> tuple[bool, str]:
    """Validate LLM response is safe and on-topic.
    Returns (is_valid, sanitized_response_or_reason).
    """
    # Check response length
    if len(response) > 3000:
        response = response[:3000] + "\n\n[Response truncated for safety]"

    response = redact_sensitive_text(response)

    # Check for blocked patterns
    for pattern in _compiled_output_patterns:
        if pattern.search(response):
            return False, (
                "I encountered an issue generating a safe response. "
                "Please try rephrasing your question about property prices."
            )

    return True, response


# ─── Layer 4: Web Search Guardrails ───

def sanitize_search_query(query: str) -> str:
    """Sanitize a user query before it's used for web search.
    Strips injection patterns, URLs, and non-RE content."""
    # Normalize unicode first
    query = sanitize_plaintext(query, max_chars=240)
    # Strip anything that looks like code or injection
    query = re.sub(r'[<>{}()\[\]|;`\\"\']', '', query)
    query = re.sub(r'https?://\S+', '', query)
    query = re.sub(r'\b(SELECT|INSERT|DELETE|DROP|UPDATE|EXEC|UNION)\b', '', query, flags=re.IGNORECASE)
    query = re.sub(
        r"\b(ignore|disregard|override|bypass|jailbreak|developer\s+mode|system\s+prompt)\b",
        "",
        query,
        flags=re.IGNORECASE,
    )
    # Collapse whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    return query[:200]


def validate_web_results(results_text: str) -> tuple[bool, str]:
    """Validate web search results before passing to LLM.
    Returns (is_safe, cleaned_text)."""
    if not results_text or results_text.startswith("["):
        return True, results_text  # Empty or status messages pass through

    # Check for suspicious patterns in results
    for pattern in _compiled_output_patterns:
        if pattern.search(results_text):
            return False, "[Web results filtered for safety.]"

    # Truncate if too long
    results_text = redact_sensitive_text(results_text)

    if len(results_text) > SEARCH_RESULT_CHAR_LIMIT:
        results_text = results_text[:SEARCH_RESULT_CHAR_LIMIT] + "\n[Results truncated]"

    return True, results_text


# ─── Combined Guardrail Check ───

def run_input_guardrails(query: str, message_count: int = 0) -> tuple[bool, str]:
    """Run all input guardrails. Returns (is_allowed, error_message).
    If is_allowed is True, error_message is empty.
    """
    query_stripped = query.strip()

    # Allow greetings and short conversational phrases through before length check
    # so that "hi", "ok", "thanks", etc. are not rejected for being too short.
    query_lower = query_stripped.lower()
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, query_lower):
            return True, ""

    # Length check (only for non-greeting messages)
    if len(query_stripped) < 3:
        return False, "Please enter a longer question about property prices."

    if len(query) > MAX_CHAT_QUERY_CHARS:
        return False, f"Please keep your question under {MAX_CHAT_QUERY_CHARS} characters."

    # Rate limit
    if message_count >= 30:
        return False, (
            "You've reached the message limit for this session. "
            "Please refresh the page to start a new conversation."
        )

    # Injection check (before topic check — catch attacks first)
    is_injection, inject_msg = detect_injection(query)
    if is_injection:
        return False, inject_msg

    # Topic check
    is_relevant, topic_msg = is_real_estate_query(query)
    if not is_relevant:
        return False, topic_msg

    return True, ""


def validate_advisory_inputs(property_input: dict, consultation_context: dict) -> tuple[bool, list[str], list[str]]:
    """Validate advisory-mode inputs before the graph receives them."""
    errors: list[str] = []
    warnings: list[str] = []

    notes = str(consultation_context.get("raw_notes", "") or "")
    if len(notes) > MAX_LISTING_NOTES_CHARS:
        errors.append(f"Listing notes must be under {MAX_LISTING_NOTES_CHARS} characters.")

    for field in ADVISORY_TEXT_FIELDS:
        value = str(consultation_context.get(field, "") or "")
        if not value:
            continue
        is_injection, message = detect_injection(value)
        if is_injection:
            errors.append(f"{field.replace('_', ' ').title()} was blocked by guardrails: {message}")

    sqft_living = int(property_input.get("sqft_living", 0) or 0)
    sqft_above = int(property_input.get("sqft_above", 0) or 0)
    sqft_lot = int(property_input.get("sqft_lot", 0) or 0)
    bedrooms = int(property_input.get("bedrooms", 0) or 0)
    bathrooms = float(property_input.get("bathrooms", 0) or 0)
    zipcode = int(property_input.get("zipcode", 0) or 0)

    if sqft_above > sqft_living:
        errors.append("Above-ground sqft cannot exceed total living sqft.")
    if bedrooms > 0 and sqft_living / max(bedrooms, 1) < 120:
        warnings.append("Living area looks very small for the bedroom count; verify the listing details.")
    if bathrooms > bedrooms + 3:
        warnings.append("Bathroom count looks unusually high for the bedroom count.")
    if sqft_lot < sqft_living * 0.25:
        warnings.append("Lot size is unusually small relative to living area; verify the input.")
    if not (98001 <= zipcode <= 98199):
        errors.append("Advisory report mode only supports King County zipcodes 98001-98199.")

    mode = consultation_context.get("client_mode", "BUYER")
    budget = float(consultation_context.get("budget", 0) or 0)
    asking_price = float(consultation_context.get("asking_price", 0) or 0)
    if mode in ("BUYER", "INVESTOR") and budget and asking_price and budget < asking_price * 0.85:
        warnings.append("Budget is materially below the asking/target price; recommendation may lean conservative.")
    if mode == "SELLER" and asking_price == 0:
        warnings.append("Seller mode works best when an asking or target list price is provided.")

    return len(errors) == 0, errors, warnings
