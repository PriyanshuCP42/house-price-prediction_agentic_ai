"""
Comparables Agent — Finds comparable properties and generates a narrative
comparison using LLM. Reuses Milestone 1 comparables matching logic.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AdvisoryState
from agents.tools.comparables_finder import find_comparables_raw
from config.llm_config import get_llm
from config.prompts import COMPARABLES_AGENT_PROMPT


def comparables_agent(state: AdvisoryState) -> dict:
    """Find comparable properties and generate narrative."""

    prop = state["property_input"]
    valuation = state.get("valuation", {})
    errors = list(state.get("error_log", []))
    predicted_price = valuation.get("predicted_price", 0)

    # Step 1: Find comparables (reuses Milestone 1 logic)
    try:
        comp_df = find_comparables_raw(
            prop["zipcode"], prop["bedrooms"], prop["sqft_living"], predicted_price
        )
    except Exception as e:
        errors.append(f"Comparables search failed: {e}")
        return {
            "comparables": [],
            "comparables_narrative": "Unable to find comparable properties.",
            "current_phase": "comparables_failed",
            "error_log": errors,
            "messages": [SystemMessage(content=f"Comparables error: {e}")],
        }

    if comp_df.empty:
        return {
            "comparables": [],
            "comparables_narrative": f"No comparable properties found in zipcode {prop['zipcode']}.",
            "current_phase": "comparables_complete",
            "error_log": errors,
            "messages": [SystemMessage(content="No comparables found")],
        }

    # Convert to list of dicts for state
    comps_list = comp_df.to_dict("records")

    # Step 2: Build comparison context
    avg_price = comp_df["price"].mean()
    median_price = comp_df["price"].median()
    price_range = f"${comp_df['price'].min():,.0f} – ${comp_df['price'].max():,.0f}"
    diff_from_avg = ((predicted_price - avg_price) / max(avg_price, 1)) * 100

    comp_summary_lines = [
        f"Subject Property: {prop['bedrooms']}BR, {prop['sqft_living']}sqft, "
        f"Grade {prop['grade']}, Age {prop['house_age']}yr, Predicted ${predicted_price:,.0f}\n",
        f"Found {len(comp_df)} comparable properties in zipcode {prop['zipcode']}:",
    ]
    for i, row in comp_df.iterrows():
        diff_pct = ((row["price"] - predicted_price) / max(predicted_price, 1)) * 100
        comp_summary_lines.append(
            f"  Comp {i+1}: ${row['price']:,.0f} | {int(row['bedrooms'])}BR | "
            f"{int(row['sqft_living'])}sqft | Grade {int(row['grade'])} | "
            f"Age {int(row['house_age'])}yr | {'+' if diff_pct > 0 else ''}{diff_pct:.1f}%"
        )
    comp_summary_lines.append(f"\nAverage: ${avg_price:,.0f} | Median: ${median_price:,.0f}")
    comp_summary_lines.append(f"Subject vs Average: {'+' if diff_from_avg > 0 else ''}{diff_from_avg:.1f}%")

    # Step 3: LLM narrative
    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=COMPARABLES_AGENT_PROMPT),
            HumanMessage(content="Generate comparison narrative:\n\n" + "\n".join(comp_summary_lines)),
        ])
        narrative = response.content
    except Exception as e:
        narrative = (
            f"Found {len(comp_df)} comparable properties in zipcode {prop['zipcode']} "
            f"with prices ranging from {price_range}. The average comparable price is "
            f"${avg_price:,.0f}. The subject property's predicted price of ${predicted_price:,.0f} "
            f"is {'+' if diff_from_avg > 0 else ''}{diff_from_avg:.1f}% vs the comparable average."
        )
        errors.append(f"LLM comparables narrative failed (using fallback): {e}")

    return {
        "comparables": comps_list,
        "comparables_narrative": narrative,
        "current_phase": "comparables_complete",
        "error_log": errors,
        "messages": [SystemMessage(content=f"Comparables analysis complete: {len(comps_list)} properties found")],
    }
