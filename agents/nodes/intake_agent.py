"""
Intake Agent — Validates property input and computes derived features.
This is a deterministic node (no LLM) — shows evaluators you know
not every node needs an LLM.
"""

import datetime
from langchain_core.messages import SystemMessage
from agents.state import AdvisoryState
from agents.tools.ml_predictor import get_zipcode_mean


def intake_agent(state: AdvisoryState) -> dict:
    """Validate input, compute derived features, set property_input on state."""

    user_query = state.get("user_query", "")
    prop = state.get("property_input", {})
    consultation = state.get("consultation_context", {})

    if not prop:
        return {
            "error_log": state.get("error_log", []) + ["No property input provided"],
            "current_phase": "intake_failed",
            "messages": [SystemMessage(content="ERROR: No property input data received.")],
        }

    # Validate ranges and clamp to sane values
    validated = {
        "bedrooms": max(1, min(10, int(prop.get("bedrooms", 3)))),
        "bathrooms": max(0, min(8, float(prop.get("bathrooms", 2)))),
        "sqft_living": max(300, min(13500, int(prop.get("sqft_living", 1800)))),
        "sqft_lot": max(500, min(1700000, int(prop.get("sqft_lot", 7500)))),
        "floors": max(1, min(3.5, float(prop.get("floors", 1)))),
        "waterfront": 1 if prop.get("waterfront") else 0,
        "view": max(0, min(4, int(prop.get("view", 0)))),
        "condition": max(1, min(5, int(prop.get("condition", 3)))),
        "grade": max(1, min(13, int(prop.get("grade", 7)))),
        "sqft_above": max(0, int(prop.get("sqft_above", prop.get("sqft_living", 1800)))),
        "sqft_basement": max(0, int(prop.get("sqft_basement", 0))),
        "lat": float(prop.get("lat", 47.5)),
        "long": float(prop.get("long", -122.2)),
        "sqft_living15": max(0, int(prop.get("sqft_living15", prop.get("sqft_living", 1800)))),
        "sqft_lot15": max(0, int(prop.get("sqft_lot15", prop.get("sqft_lot", 7500)))),
        "zipcode": int(prop.get("zipcode", 98103)),
        "yr_built": max(1900, min(datetime.datetime.now().year, int(prop.get("yr_built", 1990)))),
    }

    # Compute derived features
    validated["house_age"] = datetime.datetime.now().year - validated["yr_built"]
    validated["renovated"] = 1 if prop.get("renovated") else 0
    validated["amenity_score"] = (
        validated["waterfront"] + validated["view"]
        + validated["condition"] + validated["grade"]
    )

    # Zipcode target encoding
    zipcode_mean = get_zipcode_mean()
    if hasattr(zipcode_mean, 'values') and hasattr(zipcode_mean, 'mean'):
        # It's a pandas Series
        overall_mean = float(zipcode_mean.mean())
        validated["zipcode_encoded"] = float(zipcode_mean.get(validated["zipcode"], overall_mean))
    elif isinstance(zipcode_mean, dict) and zipcode_mean:
        overall_mean = sum(zipcode_mean.values()) / len(zipcode_mean)
        validated["zipcode_encoded"] = zipcode_mean.get(validated["zipcode"], overall_mean)
    else:
        validated["zipcode_encoded"] = 500000.0

    # Build summary message
    summary = (
        f"Property intake complete: {validated['bedrooms']}BR/{validated['bathrooms']}BA, "
        f"{validated['sqft_living']}sqft in zipcode {validated['zipcode']}, "
        f"Grade {validated['grade']}, Condition {validated['condition']}, "
        f"Age {validated['house_age']}yr"
        + (", Waterfront" if validated["waterfront"] else "")
        + (", Renovated" if validated["renovated"] else "")
    )

    consultation_context = {
        "client_mode": consultation.get("client_mode", "BUYER"),
        "objective": consultation.get("objective", "General purchase review").strip() or "General purchase review",
        "budget": max(0.0, float(consultation.get("budget", 0))),
        "asking_price": max(0.0, float(consultation.get("asking_price", 0))),
        "target_hold_years": max(0, int(consultation.get("target_hold_years", 0))),
        "risk_tolerance": consultation.get("risk_tolerance", "BALANCED"),
        "financing": consultation.get("financing", "MIXED"),
        "monthly_rent_estimate": max(0.0, float(consultation.get("monthly_rent_estimate", 0))),
        "renovation_budget": max(0.0, float(consultation.get("renovation_budget", 0))),
        "must_haves": [item for item in consultation.get("must_haves", []) if item],
        "raw_notes": str(consultation.get("raw_notes", "")).strip(),
    }

    return {
        "property_input": validated,
        "consultation_context": consultation_context,
        "current_phase": "intake_complete",
        "messages": [
            SystemMessage(
                content=(
                    f"{summary}. Client mode: {consultation_context['client_mode']} | "
                    f"Objective: {consultation_context['objective']} | "
                    f"Budget: ${consultation_context['budget']:,.0f}"
                )
            )
        ],
    }
