"""
Market Analyst Agent — RAG-powered agent that retrieves relevant market
context from the ChromaDB knowledge base and generates a market narrative.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AdvisoryState
from config.llm_config import get_llm
from config.prompts import MARKET_ANALYST_PROMPT


def market_analyst(state: AdvisoryState) -> dict:
    """Retrieve market context via RAG and generate narrative."""

    prop = state["property_input"]
    valuation = state.get("valuation", {})
    errors = list(state.get("error_log", []))

    price = valuation.get("predicted_price", 0)
    price_tier = (
        "luxury" if price > 800000
        else "premium" if price > 500000
        else "mid-market" if price > 300000
        else "affordable"
    )

    # Step 1: RAG retrieval with targeted queries
    queries = [
        f"zipcode {prop['zipcode']} King County market statistics",
        f"{price_tier} properties grade {prop['grade']} King County",
        f"King County housing market trends investment",
    ]

    all_docs = []
    all_sources = []
    try:
        from agents.tools.rag_retriever import rag_search_raw

        for query in queries:
            results = rag_search_raw(query, n_results=3)
            for doc, meta in zip(results["documents"], results["metadatas"]):
                if doc not in all_docs:  # dedup
                    all_docs.append(doc)
                    source = f"{meta.get('source', 'Unknown')} — {meta.get('section', '')}"
                    all_sources.append(source)
    except Exception as e:
        errors.append(f"RAG retrieval failed: {e}")
        return {
            "market_context": "Market context unavailable due to knowledge base error.",
            "rag_sources": [],
            "current_phase": "market_analysis_failed",
            "error_log": errors,
            "messages": [SystemMessage(content=f"Market analysis error: {e}")],
        }

    if not all_docs:
        return {
            "market_context": "No relevant market data found in the knowledge base for this property.",
            "rag_sources": [],
            "current_phase": "market_analysis_complete",
            "error_log": errors,
            "messages": [SystemMessage(content="Market analysis: no relevant data found")],
        }

    # Step 2: LLM synthesis
    retrieved_text = "\n\n---\n\n".join(all_docs[:7])  # limit context size
    try:
        llm = get_llm()
        context = (
            f"Subject Property: {prop['bedrooms']}BR/{prop['bathrooms']}BA, "
            f"{prop['sqft_living']}sqft, Grade {prop['grade']}, "
            f"Zipcode {prop['zipcode']}, Predicted Price ${price:,.0f}\n\n"
            f"Retrieved Market Documents:\n{retrieved_text}"
        )

        response = llm.invoke([
            SystemMessage(content=MARKET_ANALYST_PROMPT),
            HumanMessage(content=f"Analyze the market context for this property:\n\n{context}"),
        ])
        market_context = response.content
    except Exception as e:
        # Fallback: use raw RAG content
        market_context = f"Market data for zipcode {prop['zipcode']}:\n" + all_docs[0][:500]
        errors.append(f"LLM market synthesis failed (using raw RAG): {e}")

    unique_sources = list(dict.fromkeys(all_sources))  # dedup preserving order

    return {
        "market_context": market_context,
        "rag_sources": unique_sources[:5],
        "current_phase": "market_analysis_complete",
        "error_log": errors,
        "messages": [SystemMessage(content=f"Market analysis complete ({len(unique_sources)} sources referenced)")],
    }
