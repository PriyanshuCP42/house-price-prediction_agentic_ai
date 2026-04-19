"""
Neighborhood Analyst Agent — Builds a scorecard for neighborhood strength,
liquidity, livability, and upside potential using local market statistics.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AdvisoryState, NeighborhoodAnalysis
from agents.tools.market_stats import get_zipcode_stats_raw, _load_dataset
from config.llm_config import get_llm
from config.prompts import NEIGHBORHOOD_AGENT_PROMPT


def _clip(value: float, low: int = 0, high: int = 100) -> int:
    return int(max(low, min(high, round(value))))


def neighborhood_analyst(state: AdvisoryState) -> dict:
    """Score the target neighborhood and summarize its market position."""

    prop = state["property_input"]
    valuation = state.get("valuation", {})
    consultation = state.get("consultation_context", {})
    errors = list(state.get("error_log", []))

    try:
        df = _load_dataset()
        zip_stats = get_zipcode_stats_raw(prop["zipcode"])
        if not zip_stats:
            raise ValueError(f"No zipcode data found for {prop['zipcode']}")

        zip_medians = df.groupby("zipcode")["price"].median().sort_values()
        sales_counts = df.groupby("zipcode")["price"].count()
        county_median = float(df["price"].median())
        county_sales_median = float(sales_counts.median())

        zip_median = zip_stats["median_price"]
        zip_avg_ppsf = zip_stats["avg_price_per_sqft"]
        property_ppsf = valuation.get("price_per_sqft", zip_avg_ppsf)

        price_percentile = (
            float((zip_medians < zip_median).sum()) / max(len(zip_medians), 1)
        ) * 100
        liquidity_percentile = (
            float((sales_counts < zip_stats["sales_count"]).sum()) / max(len(sales_counts), 1)
        ) * 100

        livability_score = _clip(
            prop["grade"] * 5.5
            + prop["condition"] * 9
            + prop["amenity_score"] * 3.5
            + (8 if prop["waterfront"] else 0)
            - max(prop["house_age"] - 25, 0) * 0.35
        )
        liquidity_score = _clip(35 + liquidity_percentile * 0.65)
        upside_score = _clip(
            50
            + ((zip_avg_ppsf - property_ppsf) / max(zip_avg_ppsf, 1)) * 120
            + max(valuation.get("confidence", 50) - 50, 0) * 0.3
        )
        rental_demand_score = _clip(
            40
            + min(prop["bedrooms"], 5) * 7
            + liquidity_percentile * 0.25
            + (8 if consultation.get("monthly_rent_estimate", 0) > 0 else 0)
        )
        pricing_power_score = _clip(
            30
            + price_percentile * 0.45
            + (12 if zip_median > county_median else 0)
            + (8 if zip_stats["sales_count"] >= county_sales_median else 0)
        )
        overall_score = _clip(
            livability_score * 0.24
            + liquidity_score * 0.2
            + upside_score * 0.2
            + rental_demand_score * 0.16
            + pricing_power_score * 0.2
        )

        if zip_median > county_median * 1.15 and zip_stats["sales_count"] >= county_sales_median:
            market_heat = "HOT"
        elif zip_median < county_median * 0.9 and zip_stats["sales_count"] < county_sales_median:
            market_heat = "COOL"
        else:
            market_heat = "BALANCED"

        highlights = [
            f"Zipcode {prop['zipcode']} median price is ${zip_median:,.0f}, around {price_percentile:.0f}th percentile in King County.",
            f"Neighborhood liquidity is supported by {zip_stats['sales_count']} recorded sales in the dataset.",
            f"Subject property price-per-sqft is ${property_ppsf:,.0f} versus zipcode average ${zip_avg_ppsf:,.0f}.",
        ]

        prompt_context = (
            f"Client mode: {consultation.get('client_mode', 'BUYER')}\n"
            f"Objective: {consultation.get('objective', 'General decision support')}\n"
            f"Overall score: {overall_score}/100\n"
            f"Market heat: {market_heat}\n"
            f"Livability: {livability_score}/100\n"
            f"Liquidity: {liquidity_score}/100\n"
            f"Upside: {upside_score}/100\n"
            f"Rental demand: {rental_demand_score}/100\n"
            f"Pricing power: {pricing_power_score}/100\n"
            f"Highlights:\n- " + "\n- ".join(highlights)
        )

        try:
            llm = get_llm()
            response = llm.invoke([
                SystemMessage(content=NEIGHBORHOOD_AGENT_PROMPT),
                HumanMessage(content=prompt_context),
            ])
            narrative = response.content
        except Exception as llm_error:
            narrative = (
                f"From a neighborhood perspective, zipcode {prop['zipcode']} screens as a "
                f"{market_heat.lower()} market with an overall score of {overall_score}/100. Liquidity is "
                f"{'strong' if liquidity_score >= 65 else 'adequate' if liquidity_score >= 45 else 'thin'}, "
                f"while upside potential is {'attractive' if upside_score >= 60 else 'moderate' if upside_score >= 45 else 'limited'}. "
                f"I would position this setting for "
                f"{'owner-occupiers and balanced buyers' if consultation.get('client_mode') == 'BUYER' else 'operators who can execute a plan'}."
            )
            errors.append(f"Neighborhood narrative fallback used: {llm_error}")

        analysis: NeighborhoodAnalysis = {
            "overall_score": overall_score,
            "livability_score": livability_score,
            "liquidity_score": liquidity_score,
            "upside_score": upside_score,
            "rental_demand_score": rental_demand_score,
            "pricing_power_score": pricing_power_score,
            "market_heat": market_heat,
            "highlights": highlights,
            "narrative": narrative,
        }

        return {
            "neighborhood_analysis": analysis,
            "current_phase": "neighborhood_analysis_complete",
            "error_log": errors,
            "messages": [
                SystemMessage(
                    content=(
                        f"Neighborhood analysis complete: {overall_score}/100 overall, "
                        f"{market_heat} market, liquidity {liquidity_score}/100."
                    )
                )
            ],
        }
    except Exception as error:
        errors.append(f"Neighborhood analysis failed: {error}")
        return {
            "neighborhood_analysis": {
                "overall_score": 50,
                "livability_score": 50,
                "liquidity_score": 50,
                "upside_score": 50,
                "rental_demand_score": 50,
                "pricing_power_score": 50,
                "market_heat": "BALANCED",
                "highlights": ["Neighborhood analysis unavailable."],
                "narrative": "Neighborhood analysis unavailable due to data processing error.",
            },
            "current_phase": "neighborhood_analysis_failed",
            "error_log": errors,
            "messages": [SystemMessage(content=f"Neighborhood analysis error: {error}")],
        }
