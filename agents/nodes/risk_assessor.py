"""
Risk Assessor Agent — Computes 8-dimensional risk profile and generates
a risk narrative using LLM. Uses computed risk factors from the risk calculator.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AdvisoryState, RiskAssessment
from agents.tools.risk_calculator import compute_risk_factors_raw
from config.llm_config import get_llm
from config.prompts import RISK_ASSESSOR_PROMPT
from config.settings import VERY_HIGH_RISK_THRESHOLD


def risk_assessor(state: AdvisoryState) -> dict:
    """Compute risk profile and generate narrative."""

    prop = state["property_input"]
    valuation = state.get("valuation", {})
    comps = state.get("comparables", [])
    errors = list(state.get("error_log", []))

    predicted_price = valuation.get("predicted_price", 0)
    confidence = valuation.get("confidence", 50)

    # Compute comparable average
    comp_avg = 0
    if comps:
        comp_avg = sum(c.get("price", 0) for c in comps) / len(comps)

    # Step 1: Compute risk factors
    try:
        risk_data = compute_risk_factors_raw(
            predicted_price=predicted_price,
            confidence=confidence,
            house_age=prop["house_age"],
            grade=prop["grade"],
            condition=prop["condition"],
            sqft_living=prop["sqft_living"],
            sqft_lot=prop["sqft_lot"],
            renovated=prop["renovated"],
            waterfront=prop["waterfront"],
            zipcode=prop["zipcode"],
            comp_avg_price=comp_avg,
        )
    except Exception as e:
        errors.append(f"Risk calculation failed: {e}")
        return {
            "risk_assessment": {
                "overall_risk": "MODERATE",
                "risk_score": 50.0,
                "risk_factors": [],
                "risk_narrative": "Risk assessment unavailable due to calculation error.",
            },
            "current_phase": "risk_assessment_failed",
            "error_log": errors,
            "messages": [SystemMessage(content=f"Risk assessment error: {e}")],
        }

    # Step 2: LLM narrative
    risk_summary_lines = [
        f"Overall Risk: {risk_data['overall_risk']} (Score: {risk_data['total_score']:.1f}/100)\n",
    ]
    for f in risk_data["factors"]:
        risk_summary_lines.append(
            f"[{f['severity']}] {f['factor']}: {f['score']:.1f}/12.5 — {f['explanation']}"
        )

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=RISK_ASSESSOR_PROMPT),
            HumanMessage(content="Generate risk narrative:\n\n" + "\n".join(risk_summary_lines)),
        ])
        risk_narrative = response.content
    except Exception as e:
        # Fallback: structured text from computed factors
        top_risks = sorted(risk_data["factors"], key=lambda x: -x["score"])[:3]
        risk_narrative = (
            f"Overall risk level: {risk_data['overall_risk']} "
            f"(score: {risk_data['total_score']:.1f}/100). "
            f"Top risk factors: "
            + "; ".join(f"{r['factor']} ({r['severity']})" for r in top_risks)
            + "."
        )
        errors.append(f"LLM risk narrative failed (using fallback): {e}")

    risk_assessment: RiskAssessment = {
        "overall_risk": risk_data["overall_risk"],
        "risk_score": risk_data["total_score"],
        "risk_factors": risk_data["factors"],
        "risk_narrative": risk_narrative,
    }

    # Check if human review is needed
    needs_review = risk_data["total_score"] >= VERY_HIGH_RISK_THRESHOLD

    return {
        "risk_assessment": risk_assessment,
        "needs_human_review": needs_review,
        "current_phase": "risk_assessment_complete",
        "error_log": errors,
        "messages": [
            SystemMessage(
                content=f"Risk assessment complete: {risk_data['overall_risk']} "
                f"(score {risk_data['total_score']:.1f}/100)"
                + (" — HUMAN REVIEW TRIGGERED" if needs_review else "")
            )
        ],
    }
