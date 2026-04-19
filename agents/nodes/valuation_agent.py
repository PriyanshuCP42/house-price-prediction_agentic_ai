"""
Valuation Agent — Runs the ML model prediction and uses LLM to explain results.
Uses tools: predict_property_price, get_zipcode_market_stats
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AdvisoryState, ValuationResult
from agents.tools.ml_predictor import (
    predict_price_raw,
    compute_investment_score,
    compute_market_status,
)
from agents.tools.market_stats import get_zipcode_stats_raw
from config.llm_config import get_llm
from config.prompts import VALUATION_AGENT_PROMPT
from config.settings import get_investment_label


def valuation_agent(state: AdvisoryState) -> dict:
    """Run ML prediction, compute scores, generate LLM explanation."""

    prop = state["property_input"]
    errors = list(state.get("error_log", []))

    # Step 1: ML prediction
    try:
        price, price_low, price_high, confidence, std_dev = predict_price_raw(prop)
    except Exception as e:
        errors.append(f"ML prediction failed: {e}")
        return {
            "error_log": errors,
            "current_phase": "valuation_failed",
            "messages": [SystemMessage(content=f"Valuation error: {e}")],
        }

    # Step 2: Investment score & market status
    inv_score = compute_investment_score(
        price, prop["sqft_living"], prop["grade"], prop["condition"],
        prop["house_age"], prop["waterfront"], prop["view"], prop["renovated"],
    )
    market_stat, ppsf = compute_market_status(price, prop["sqft_living"])
    inv_code, inv_label, inv_color = get_investment_label(inv_score)

    # Step 3: Zipcode stats for context
    zip_stats = get_zipcode_stats_raw(prop["zipcode"])

    # Step 4: LLM explanation
    explanation = ""
    try:
        llm = get_llm()
        context = (
            f"Property: {prop['bedrooms']}BR/{prop['bathrooms']}BA, {prop['sqft_living']}sqft, "
            f"Grade {prop['grade']}, Condition {prop['condition']}, Zipcode {prop['zipcode']}\n"
            f"ML Prediction: ${price:,.0f} (confidence: {confidence:.1f}%)\n"
            f"Price Range: ${price_low:,.0f} – ${price_high:,.0f}\n"
            f"Price Per Sqft: ${ppsf:,.0f}\n"
            f"Market Status: {market_stat}\n"
            f"Investment Score: {inv_score}/100 ({inv_label})\n"
        )
        if zip_stats:
            context += (
                f"\nZipcode {prop['zipcode']} Stats:\n"
                f"  Median Price: ${zip_stats['median_price']:,.0f}\n"
                f"  Mean Price: ${zip_stats['mean_price']:,.0f}\n"
                f"  Avg Price/Sqft: ${zip_stats['avg_price_per_sqft']:,.0f}\n"
                f"  Sales Volume: {zip_stats['sales_count']}\n"
                f"  Avg Grade: {zip_stats['avg_grade']:.1f}"
            )

        response = llm.invoke([
            SystemMessage(content=VALUATION_AGENT_PROMPT),
            HumanMessage(content=f"Explain this property valuation:\n\n{context}"),
        ])
        explanation = response.content
    except Exception as e:
        explanation = (
            f"The model predicts this property at ${price:,.0f} with {confidence:.1f}% confidence. "
            f"The price per square foot is ${ppsf:,.0f}, classified as '{market_stat}'. "
            f"Investment score: {inv_score}/100 ({inv_label})."
        )
        errors.append(f"LLM explanation failed (using fallback): {e}")

    valuation: ValuationResult = {
        "predicted_price": price,
        "price_low": price_low,
        "price_high": price_high,
        "confidence": confidence,
        "std_dev": std_dev,
        "investment_score": inv_score,
        "investment_label": inv_label,
        "market_status": market_stat,
        "price_per_sqft": ppsf,
        "valuation_explanation": explanation,
    }

    return {
        "valuation": valuation,
        "current_phase": "valuation_complete",
        "error_log": errors,
        "messages": [SystemMessage(content=f"Valuation complete: ${price:,.0f} ({confidence:.1f}% confidence)")],
    }
