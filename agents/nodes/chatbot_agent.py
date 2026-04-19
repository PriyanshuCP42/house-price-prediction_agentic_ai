"""
Chatbot Agent — LangGraph ReAct Agent with autonomous tool-calling.
The LLM decides when to search the web, query RAG, or use the ML model.
This is a true agentic AI, not a scripted chatbot.
"""

import re
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config.llm_config import get_llm
from config.prompts import CHATBOT_SYSTEM_PROMPT
from config.guardrails import (
    run_input_guardrails, validate_response,
    sanitize_search_query, validate_web_results,
)
from config.settings import CHAT_HISTORY_WINDOW

# King County zipcode range + city names
KC_ZIPCODES = set(range(98001, 98200))
KC_CITIES = {
    "seattle", "bellevue", "redmond", "kirkland", "renton", "kent", "auburn",
    "federal way", "burien", "tukwila", "seatac", "shoreline", "kenmore",
    "bothell", "woodinville", "sammamish", "issaquah", "mercer island",
    "medina", "clyde hill", "yarrow point", "king county",
}

KC_ZIPCODE_COORDS = {
    98103: (47.6715, -122.3425), 98115: (47.6850, -122.2956),
    98118: (47.5412, -122.2646), 98052: (47.6694, -122.1186),
    98034: (47.7116, -122.2091), 98006: (47.5506, -122.1603),
    98033: (47.6744, -122.1876), 98004: (47.6162, -122.2044),
}


# ═══════════════════════════════════════════════════
# TOOLS — The LLM agent autonomously decides which to call
# ═══════════════════════════════════════════════════

@tool
def search_web_real_estate(query: str) -> str:
    """Search the internet for current real estate market data, property prices,
    and housing trends GLOBALLY. Use this when you need live/current data about
    any housing market worldwide — US cities, Indian pincodes, UK postcodes, etc.
    Include location details like city name, pincode, or zipcode in the query.
    Works for India (Magicbricks, 99acres, Housing.com), US (Zillow, Redfin),
    UK (Rightmove, Zoopla), and more."""

    from agents.tools.web_search import search_real_estate
    clean_q = sanitize_search_query(query)
    if len(clean_q) < 5:
        return "Query too short. Please provide more details about the location or property."

    raw_results = search_real_estate(clean_q, max_results=5)
    is_safe, validated = validate_web_results(raw_results)
    return validated if is_safe else "[Web results filtered for safety.]"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the local real estate knowledge base for market insights,
    zipcode profiles, investment guidelines, and property valuation methods.
    Use this for general real estate principles, King County zipcode data,
    and investment analysis frameworks."""

    try:
        from agents.tools.rag_retriever import rag_search_raw
        clean_q = sanitize_search_query(query)
        if len(clean_q) < 3:
            return "Please provide a more specific real estate question for the knowledge base."
        results = rag_search_raw(clean_q, n_results=4, max_distance=1.3)
        if results["documents"]:
            passages = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                source = meta.get("source", "Knowledge Base")
                section = meta.get("section", "")
                passages.append(f"[{source} — {section}]\n{doc[:300]}")
            return "\n\n".join(passages[:3])
        return "No relevant information found in knowledge base."
    except Exception:
        return "Knowledge base temporarily unavailable."


@tool
def predict_king_county_price(
    bedrooms: int = 3,
    bathrooms: float = 2.0,
    sqft_living: int = 1800,
    grade: int = 7,
    condition: int = 3,
    zipcode: int = 98103,
    yr_built: int = 1990,
) -> str:
    """Predict property price using the trained ML model. ONLY works for
    King County, WA (zipcodes 98001-98199). Provide as many property details
    as possible for the most accurate prediction. Returns predicted price,
    confidence interval, and investment score."""

    try:
        zipcode = int(zipcode)
    except Exception:
        return "Invalid zipcode. Please provide a King County zipcode between 98001 and 98199."

    if zipcode not in KC_ZIPCODES:
        return (
            f"Zipcode {zipcode} is outside King County, WA. "
            "The ML model only works for KC zipcodes 98001-98199. "
            "Use search_web_real_estate or search_knowledge_base for other locations."
        )

    import datetime
    from agents.tools.ml_predictor import predict_price_raw, compute_investment_score, compute_market_status, get_zipcode_mean

    bedrooms = max(1, min(10, int(bedrooms)))
    bathrooms = max(0.0, min(8.0, float(bathrooms)))
    sqft_living = max(300, min(13000, int(sqft_living)))
    grade = max(1, min(13, int(grade)))
    condition = max(1, min(5, int(condition)))
    yr_built = max(1900, min(datetime.datetime.now().year, int(yr_built)))

    house_age = datetime.datetime.now().year - yr_built
    lat, lng = KC_ZIPCODE_COORDS.get(zipcode, (47.55, -122.20))

    zipcode_mean = get_zipcode_mean()
    if hasattr(zipcode_mean, 'mean'):
        zenc = float(zipcode_mean.get(zipcode, zipcode_mean.mean()))
    elif isinstance(zipcode_mean, dict) and zipcode_mean:
        zenc = float(zipcode_mean.get(zipcode, sum(zipcode_mean.values()) / len(zipcode_mean)))
    else:
        zenc = 500000.0

    input_dict = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
        "sqft_lot": 7500, "floors": 1.0, "waterfront": 0, "view": 0,
        "condition": condition, "grade": grade, "sqft_above": sqft_living,
        "sqft_basement": 0, "lat": lat, "long": lng,
        "sqft_living15": sqft_living, "sqft_lot15": 7500,
        "house_age": house_age, "renovated": 0,
        "amenity_score": condition + grade, "zipcode_encoded": zenc,
    }

    try:
        price, low, high, conf, std = predict_price_raw(input_dict)
        inv_score = compute_investment_score(price, sqft_living, grade, condition, house_age, 0, 0, 0)
        market_stat, ppsf = compute_market_status(price, sqft_living)

        return (
            f"ML Model Prediction (King County, trained Random Forest R²=0.88):\n"
            f"  Predicted Price: ${price:,.0f}\n"
            f"  Price Range: ${low:,.0f} – ${high:,.0f}\n"
            f"  Model Confidence: {conf:.1f}%\n"
            f"  Price/Sqft: ${ppsf:,.0f}\n"
            f"  Market Status: {market_stat}\n"
            f"  Investment Score: {inv_score}/100\n"
            f"  Property: {bedrooms}BR/{bathrooms}BA, {sqft_living}sqft, "
            f"Grade {grade}, Condition {condition}, Zipcode {zipcode}, Built {yr_built}"
        )
    except Exception as e:
        return f"ML prediction error. Try using search_web_real_estate instead."


@tool
def get_comparable_sales(zipcode: int, bedrooms: int = 3, sqft_living: int = 1800) -> str:
    """Find comparable recently-sold properties in King County, WA.
    ONLY works for KC zipcodes 98001-98199. Shows similar properties
    that sold recently with their prices."""

    try:
        zipcode = int(zipcode)
    except Exception:
        return "Invalid zipcode. Please provide a King County zipcode between 98001 and 98199."

    if zipcode not in KC_ZIPCODES:
        return f"Zipcode {zipcode} is outside King County. Use search_web_real_estate for other areas."

    try:
        from agents.tools.comparables_finder import find_comparables_raw
        from agents.tools.ml_predictor import predict_price_raw, get_zipcode_mean
        import datetime

        bedrooms = max(1, min(10, int(bedrooms)))
        sqft_living = max(300, min(13000, int(sqft_living)))

        zipcode_mean = get_zipcode_mean()
        if hasattr(zipcode_mean, 'mean'):
            zenc = float(zipcode_mean.get(zipcode, zipcode_mean.mean()))
        else:
            zenc = 500000.0

        house_age = datetime.datetime.now().year - 1990
        lat, lng = KC_ZIPCODE_COORDS.get(zipcode, (47.55, -122.20))
        input_dict = {
            "bedrooms": bedrooms, "bathrooms": 2.0, "sqft_living": sqft_living,
            "sqft_lot": 7500, "floors": 1.0, "waterfront": 0, "view": 0,
            "condition": 3, "grade": 7, "sqft_above": sqft_living,
            "sqft_basement": 0, "lat": lat, "long": lng,
            "sqft_living15": sqft_living, "sqft_lot15": 7500,
            "house_age": house_age, "renovated": 0,
            "amenity_score": 10, "zipcode_encoded": zenc,
        }
        price, *_ = predict_price_raw(input_dict)

        comps = find_comparables_raw(zipcode, bedrooms, sqft_living, price)
        if comps.empty:
            return f"No comparable properties found in zipcode {zipcode}."

        lines = [f"Comparable Sales in Zipcode {zipcode} ({len(comps)} found):"]
        for i, row in comps.iterrows():
            lines.append(
                f"  #{i+1}: ${row['price']:,.0f} | {int(row['bedrooms'])}BR/{row['bathrooms']}BA | "
                f"{int(row['sqft_living'])}sqft | Grade {int(row['grade'])} | "
                f"Age {int(row['house_age'])}yr | ${row['price_per_sqft']:,.0f}/sqft"
            )
        lines.append(f"\nAverage: ${comps['price'].mean():,.0f} | Median: ${comps['price'].median():,.0f}")
        return "\n".join(lines)

    except Exception:
        return "Comparable search error. Try search_web_real_estate instead."


# All tools the agent can use
AGENT_TOOLS = [
    search_web_real_estate,
    search_knowledge_base,
    predict_king_county_price,
    get_comparable_sales,
]


# ═══════════════════════════════════════════════════
# AGENT BUILD — LangGraph ReAct Agent
# ═══════════════════════════════════════════════════

_agent = None


def _build_agent():
    """Build a LangGraph ReAct agent with tools."""
    global _agent
    if _agent is None:
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver

        llm = get_llm()
        checkpointer = MemorySaver()

        _agent = create_react_agent(
            model=llm,
            tools=AGENT_TOOLS,
            prompt=CHATBOT_SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
    return _agent


def chat_with_advisor(query: str, chat_history: list[dict], thread_id: str | None = None) -> str:
    """Process a chat query through the LangGraph ReAct agent.

    The agent autonomously decides which tools to call:
    - search_web_real_estate: for live market data from the internet
    - search_knowledge_base: for static knowledge (RAG)
    - predict_king_county_price: for ML model predictions (KC only)
    - get_comparable_sales: for finding similar sold properties (KC only)

    Includes defense-in-depth guardrails at input and output.
    """

    # Defense-in-depth: guardrails inside the agent
    is_allowed, guard_msg = run_input_guardrails(query, len(chat_history))
    if not is_allowed:
        return guard_msg

    try:
        agent = _build_agent()

        # Build messages from history (exclude blocked queries)
        messages = []
        clean_history = [
            m for m in chat_history[-CHAT_HISTORY_WINDOW:]
            if not m.get("_blocked") and m.get("content") != "[blocked query]" and m.get("role") in ("user", "assistant")
        ]
        for msg in clean_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=query))

        # Run the agent
        config = {
            "configurable": {"thread_id": thread_id or "chat-default"},
            "recursion_limit": 8,
        }
        result = agent.invoke({"messages": messages}, config)

        # Extract the final AI response
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
        if ai_messages:
            response_text = ai_messages[-1].content
        else:
            response_text = "I couldn't generate a response. Please try rephrasing your question."

    except Exception as e:
        # Fallback: direct LLM call without tools
        try:
            llm = get_llm()
            fallback_messages = [
                SystemMessage(content=CHATBOT_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
            response = llm.invoke(fallback_messages)
            response_text = response.content + "\n\n*Note: Advanced tools were temporarily unavailable.*"
        except Exception:
            return "I'm having trouble connecting to the AI service right now. Please try again in a moment."

    # Defense-in-depth: validate output
    is_valid, validated = validate_response(response_text)
    return validated if is_valid else validated
