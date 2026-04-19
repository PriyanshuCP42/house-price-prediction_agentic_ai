"""
Decision Agent — Produces bull/base/risk viewpoints and a concise executive memo.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AdvisoryState, DecisionSummary
from config.llm_config import get_llm
from config.prompts import DECISION_AGENT_PROMPT


def decision_agent(state: AdvisoryState) -> dict:
    """Create decision lenses and next-step recommendations."""

    consultation = state.get("consultation_context", {})
    valuation = state.get("valuation", {})
    risk = state.get("risk_assessment", {})
    neighborhood = state.get("neighborhood_analysis", {})
    negotiation = state.get("negotiation_strategy", {})
    errors = list(state.get("error_log", []))

    client_mode = consultation.get("client_mode", "BUYER")
    risk_level = risk.get("overall_risk", "MODERATE")
    investment_score = valuation.get("investment_score", 50)
    neighborhood_score = neighborhood.get("overall_score", 50)

    recommendation = "HOLD"
    if risk_level in ("HIGH", "VERY_HIGH"):
        recommendation = "CAUTION" if investment_score >= 35 else "AVOID"
    elif client_mode == "INVESTOR" and investment_score >= 70 and neighborhood_score >= 55:
        recommendation = "BUY"
    elif investment_score >= 75 and risk_level == "LOW":
        recommendation = "STRONG_BUY"
    elif investment_score >= 60:
        recommendation = "BUY"
    elif investment_score >= 45:
        recommendation = "HOLD"
    elif investment_score >= 30:
        recommendation = "CAUTION"
    else:
        recommendation = "AVOID"

    bull_case = (
        f"If execution is clean, the property benefits from neighborhood strength "
        f"({neighborhood_score}/100), a valuation center at ${valuation.get('predicted_price', 0):,.0f}, "
        f"and a negotiation target of ${negotiation.get('target_price', 0):,.0f}."
    )
    base_case = (
        f"On a balanced underwriting view, the deal works best when held for "
        f"{consultation.get('target_hold_years', 0)} years with discipline around the walk-away price "
        f"of ${negotiation.get('walk_away_price', 0):,.0f}."
    )
    risk_case = (
        f"The downside case is driven by {risk_level.lower()} risk, confidence of "
        f"{valuation.get('confidence', 0):.1f}%, and any gap between asking and intrinsic value."
    )

    prompt_context = (
        f"Client mode: {client_mode}\n"
        f"Recommendation seed: {recommendation}\n"
        f"Investment score: {investment_score}\n"
        f"Risk level: {risk_level}\n"
        f"Neighborhood score: {neighborhood_score}\n"
        f"Bull case: {bull_case}\n"
        f"Base case: {base_case}\n"
        f"Risk case: {risk_case}"
    )

    next_steps = [
        "Validate the negotiation corridor against a current listing sheet or broker comps.",
        "Pressure-test renovation, financing, and hold assumptions before committing.",
        "Use the risk heatmap to decide whether to proceed, renegotiate, or pass.",
    ]

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=DECISION_AGENT_PROMPT),
            HumanMessage(content=prompt_context),
        ])
        executive_summary = response.content
    except Exception as llm_error:
        executive_summary = (
            f"My read is {recommendation.replace('_', ' ')} for this {client_mode.lower()} profile. "
            f"The deal has to be viewed through three lenses: investment score ({investment_score}/100), "
            f"neighborhood quality ({neighborhood_score}/100), and risk level ({risk_level}). "
            "I would use the next step to either tighten the offer logic or walk away if the numbers stop supporting the story."
        )
        errors.append(f"Decision memo fallback used: {llm_error}")

    decision_summary: DecisionSummary = {
        "headline": f"{recommendation.replace('_', ' ')} for this {client_mode.lower()} scenario",
        "executive_summary": executive_summary,
        "next_steps": next_steps,
    }

    decision_lenses = [
        {"lens": "Bull Case", "stance": "Upside", "summary": bull_case},
        {"lens": "Base Case", "stance": "Balanced", "summary": base_case},
        {"lens": "Risk Case", "stance": "Protection", "summary": risk_case},
    ]

    return {
        "decision_lenses": decision_lenses,
        "decision_summary": decision_summary,
        "recommendation": recommendation,
        "current_phase": "decision_complete",
        "error_log": errors,
        "messages": [
            SystemMessage(
                content=f"Decision analysis complete: {recommendation.replace('_', ' ')}."
            )
        ],
    }
