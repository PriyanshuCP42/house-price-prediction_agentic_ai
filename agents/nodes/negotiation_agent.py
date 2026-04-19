"""
Negotiation Agent — Builds a persona-aware pricing corridor and action plan.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AdvisoryState, NegotiationStrategy
from config.llm_config import get_llm
from config.prompts import NEGOTIATION_AGENT_PROMPT


def _bounded_price(value: float) -> float:
    return float(max(0.0, round(value, 2)))


def negotiation_agent(state: AdvisoryState) -> dict:
    """Create a negotiation playbook for buyer, seller, or investor scenarios."""

    consultation = state.get("consultation_context", {})
    valuation = state.get("valuation", {})
    risk = state.get("risk_assessment", {})
    comps = state.get("comparables", [])
    neighborhood = state.get("neighborhood_analysis", {})
    errors = list(state.get("error_log", []))

    mode = consultation.get("client_mode", "BUYER")
    asking_price = consultation.get("asking_price", 0) or valuation.get("predicted_price", 0)
    predicted_price = valuation.get("predicted_price", asking_price)
    confidence = valuation.get("confidence", 50)
    risk_score = risk.get("risk_score", 50)
    comp_avg = (
        sum(comp.get("price", 0) for comp in comps) / len(comps)
        if comps else predicted_price
    )

    buyer_discount = 0.03 + risk_score / 500 + max(0, 65 - confidence) / 500
    investor_discount = 0.05 + risk_score / 420 + max(0, 70 - confidence) / 450

    if mode == "SELLER":
        anchor_price = max(asking_price, predicted_price * 1.02)
        target_price = max(predicted_price, comp_avg, asking_price * 0.99)
        walk_away_price = max(valuation.get("price_low", predicted_price * 0.94), target_price * 0.94)
        leverage_points = [
            f"Support pricing with predicted value of ${predicted_price:,.0f} and comparable average of ${comp_avg:,.0f}.",
            f"Neighborhood score of {neighborhood.get('overall_score', 50)}/100 supports confidence when qualifying serious buyers.",
            "Lead negotiations by anchoring on condition, grade, and liquidity rather than starting with discounts.",
        ]
        caution_points = [
            "Do not defend an asking price materially above the model high range without renovation or scarcity evidence.",
            "If buyers repeatedly cite deferred maintenance, adjust terms before cutting headline price aggressively.",
        ]
        action_plan = [
            "Open at the anchor price but prepare concessions on closing costs or timelines first.",
            "Share comparable support before responding to low offers.",
            "Treat offers below the walk-away price as fallback or relist triggers.",
        ]
    else:
        discount = investor_discount if mode == "INVESTOR" else buyer_discount
        anchor_price = min(asking_price * (1 - discount * 1.35), predicted_price * (0.94 if mode == "INVESTOR" else 0.97))
        target_price = min(asking_price * (1 - discount), predicted_price * (0.98 if mode == "INVESTOR" else 1.0))
        walk_away_price = min(
            asking_price,
            valuation.get("price_high", predicted_price * 1.06),
            predicted_price * (1.0 if mode == "BUYER" else 0.97),
        )
        leverage_points = [
            f"Comparable average is ${comp_avg:,.0f} against asking ${asking_price:,.0f}.",
            f"Risk score of {risk_score:.1f}/100 creates room to negotiate on certainty and due diligence.",
            f"Model confidence is {confidence:.1f}% with predicted value centered at ${predicted_price:,.0f}.",
        ]
        caution_points = [
            "Avoid stretching above the walk-away price unless a must-have requirement is uniquely met.",
            "Use inspection, age, and renovation budget as negotiation anchors rather than generic discount requests.",
        ]
        if mode == "INVESTOR":
            leverage_points.append(
                f"Monthly rent assumption of ${consultation.get('monthly_rent_estimate', 0):,.0f} should still clear your yield hurdle."
            )
            action_plan = [
                "Open with the anchor price and justify it with risk, yield, and renovation math.",
                "Negotiate for price reduction before requesting cosmetic credits.",
                "Exit if the deal crosses the walk-away threshold or kills target yield.",
            ]
        else:
            action_plan = [
                "Open with the anchor price and present comparables immediately.",
                "Use inspection contingency and timing flexibility as tradeable concessions.",
                "Only move toward the target price when the seller responds with documented support.",
            ]

    prompt_context = (
        f"Client mode: {mode}\n"
        f"Asking price: ${asking_price:,.0f}\n"
        f"Predicted price: ${predicted_price:,.0f}\n"
        f"Comparable average: ${comp_avg:,.0f}\n"
        f"Risk score: {risk_score:.1f}/100\n"
        f"Anchor: ${anchor_price:,.0f}\n"
        f"Target: ${target_price:,.0f}\n"
        f"Walk-away: ${walk_away_price:,.0f}\n"
        f"Leverage points:\n- " + "\n- ".join(leverage_points)
    )

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=NEGOTIATION_AGENT_PROMPT),
            HumanMessage(content=prompt_context),
        ])
        strategy_summary = response.content
    except Exception as llm_error:
        strategy_summary = (
            f"If I were coaching a {mode.lower()} on this deal, I would frame the corridor as "
            f"anchor ${anchor_price:,.0f}, target ${target_price:,.0f}, and walk-away ${walk_away_price:,.0f}. "
            f"The strongest leverage comes from the comps, the model range, and the risk profile, so I would lead with those facts rather than a generic discount ask."
        )
        errors.append(f"Negotiation narrative fallback used: {llm_error}")

    strategy: NegotiationStrategy = {
        "anchor_price": _bounded_price(anchor_price),
        "target_price": _bounded_price(target_price),
        "walk_away_price": _bounded_price(walk_away_price),
        "leverage_points": leverage_points,
        "caution_points": caution_points,
        "action_plan": action_plan,
        "strategy_summary": strategy_summary,
    }

    return {
        "negotiation_strategy": strategy,
        "current_phase": "negotiation_strategy_complete",
        "error_log": errors,
        "messages": [
            SystemMessage(
                content=(
                    f"Negotiation strategy complete: anchor ${anchor_price:,.0f}, "
                    f"target ${target_price:,.0f}, walk-away ${walk_away_price:,.0f}."
                )
            )
        ],
    }
