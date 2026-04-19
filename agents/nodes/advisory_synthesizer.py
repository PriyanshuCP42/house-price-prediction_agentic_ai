"""
Advisory Synthesizer Agent — Assembles all prior analysis into a final
structured advisory report. Uses LLM with robust deterministic fallback.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import AdvisoryState
from config.llm_config import get_llm
from config.prompts import ADVISORY_SYNTHESIZER_PROMPT


def advisory_synthesizer(state: AdvisoryState) -> dict:
    """Synthesize all analysis into a final advisory report."""

    consultation = state.get("consultation_context", {})
    valuation = state.get("valuation", {})
    market_context = state.get("market_context", "Market context not available.")
    comparables_narrative = state.get("comparables_narrative", "No comparable data available.")
    risk_assessment = state.get("risk_assessment", {})
    neighborhood = state.get("neighborhood_analysis", {})
    negotiation = state.get("negotiation_strategy", {})
    decision_lenses = state.get("decision_lenses", [])
    decision_summary = state.get("decision_summary", {})
    errors = list(state.get("error_log", []))
    iteration = state.get("iteration_count", 0) + 1

    sections = [
        "=== CLIENT BRIEF ===\n"
        f"Client Mode: {consultation.get('client_mode', 'BUYER')}\n"
        f"Objective: {consultation.get('objective', 'General decision support')}\n"
        f"Budget: ${consultation.get('budget', 0):,.0f}\n"
        f"Asking Price: ${consultation.get('asking_price', 0):,.0f}\n"
        f"Risk Tolerance: {consultation.get('risk_tolerance', 'BALANCED')}\n"
        f"Financing: {consultation.get('financing', 'MIXED')}\n"
        f"Hold Horizon: {consultation.get('target_hold_years', 0)} years\n"
        f"Monthly Rent Estimate: ${consultation.get('monthly_rent_estimate', 0):,.0f}\n"
        f"Renovation Budget: ${consultation.get('renovation_budget', 0):,.0f}\n"
        f"Must Haves: {', '.join(consultation.get('must_haves', [])) or 'Not specified'}"
    ]

    if valuation:
        sections.append(
            "=== VALUATION DATA ===\n"
            f"Predicted Price: ${valuation.get('predicted_price', 0):,.0f}\n"
            f"Price Range: ${valuation.get('price_low', 0):,.0f} – ${valuation.get('price_high', 0):,.0f}\n"
            f"Confidence: {valuation.get('confidence', 0):.1f}%\n"
            f"Price/Sqft: ${valuation.get('price_per_sqft', 0):,.0f}\n"
            f"Market Status: {valuation.get('market_status', 'Unknown')}\n"
            f"Investment Score: {valuation.get('investment_score', 0)}/100 ({valuation.get('investment_label', 'N/A')})\n"
            f"Explanation: {valuation.get('valuation_explanation', 'N/A')}"
        )

    sections.append(f"=== MARKET CONTEXT (from RAG) ===\n{market_context}")
    sections.append(f"=== COMPARABLE SALES ANALYSIS ===\n{comparables_narrative}")

    if risk_assessment:
        risk_text = (
            f"Overall Risk: {risk_assessment.get('overall_risk', 'Unknown')} "
            f"(Score: {risk_assessment.get('risk_score', 0):.1f}/100)\n"
        )
        for factor in risk_assessment.get("risk_factors", []):
            risk_text += (
                f"  [{factor.get('severity', '?')}] {factor.get('factor', '?')}: "
                f"{factor.get('explanation', '')}\n"
                f"    Mitigation: {factor.get('mitigation', '')}\n"
            )
        risk_text += f"\nRisk Narrative: {risk_assessment.get('risk_narrative', '')}"
        sections.append(f"=== RISK ASSESSMENT ===\n{risk_text}")

    if neighborhood:
        sections.append(
            "=== NEIGHBORHOOD INTELLIGENCE ===\n"
            f"Overall Score: {neighborhood.get('overall_score', 0)}/100\n"
            f"Market Heat: {neighborhood.get('market_heat', 'BALANCED')}\n"
            f"Livability: {neighborhood.get('livability_score', 0)}/100\n"
            f"Liquidity: {neighborhood.get('liquidity_score', 0)}/100\n"
            f"Upside: {neighborhood.get('upside_score', 0)}/100\n"
            f"Rental Demand: {neighborhood.get('rental_demand_score', 0)}/100\n"
            f"Pricing Power: {neighborhood.get('pricing_power_score', 0)}/100\n"
            f"Narrative: {neighborhood.get('narrative', 'N/A')}\n"
            + "\n".join(f"  Highlight: {item}" for item in neighborhood.get("highlights", []))
        )

    if negotiation:
        sections.append(
            "=== NEGOTIATION PLAYBOOK ===\n"
            f"Anchor Price: ${negotiation.get('anchor_price', 0):,.0f}\n"
            f"Target Price: ${negotiation.get('target_price', 0):,.0f}\n"
            f"Walk-away Price: ${negotiation.get('walk_away_price', 0):,.0f}\n"
            f"Summary: {negotiation.get('strategy_summary', 'N/A')}\n"
            + "\n".join(f"  Leverage: {item}" for item in negotiation.get("leverage_points", []))
            + ("\n" if negotiation.get("leverage_points") else "")
            + "\n".join(f"  Caution: {item}" for item in negotiation.get("caution_points", []))
            + ("\n" if negotiation.get("caution_points") else "")
            + "\n".join(f"  Action: {item}" for item in negotiation.get("action_plan", []))
        )

    if decision_lenses or decision_summary:
        lens_lines = [
            f"{lens.get('lens', 'Lens')} ({lens.get('stance', 'Balanced')}): {lens.get('summary', '')}"
            for lens in decision_lenses
        ]
        sections.append(
            "=== DECISION MEMO ===\n"
            f"Headline: {decision_summary.get('headline', 'N/A')}\n"
            f"Executive Summary: {decision_summary.get('executive_summary', 'N/A')}\n"
            + "\n".join(f"  Next Step: {step}" for step in decision_summary.get("next_steps", []))
            + ("\n" if decision_summary.get("next_steps") else "")
            + "\n".join(f"  Lens: {item}" for item in lens_lines)
        )

    full_context = "\n\n".join(sections)

    risk_level = risk_assessment.get("overall_risk", "MODERATE")
    inv_score = valuation.get("investment_score", 50)
    confidence = valuation.get("confidence", 50)
    recommendation_seed = state.get("recommendation")

    recommendation_hint = ""
    if recommendation_seed:
        recommendation_hint = f"Recommendation seed from decision agent: {recommendation_seed}."
    elif risk_level in ("HIGH", "VERY_HIGH"):
        recommendation_hint = "Note: Risk is HIGH or VERY_HIGH, so recommendation should be CAUTION or AVOID."
    elif inv_score >= 75 and confidence >= 60:
        recommendation_hint = "Note: Strong investment score and good confidence suggest STRONG BUY or BUY."
    elif inv_score >= 60:
        recommendation_hint = "Note: Good investment score suggests BUY or HOLD."

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=ADVISORY_SYNTHESIZER_PROMPT),
            HumanMessage(
                content=(
                    "Synthesize this analysis into a structured advisory report:\n\n"
                    f"{full_context}\n\n{recommendation_hint}"
                )
            ),
        ])
        report = response.content
    except Exception as error:
        report = _build_fallback_report(
            consultation,
            valuation,
            market_context,
            comparables_narrative,
            risk_assessment,
            neighborhood,
            negotiation,
            decision_lenses,
            decision_summary,
        )
        errors.append(f"LLM synthesis failed (using fallback): {error}")

    recommendation = _extract_recommendation(report, risk_level, inv_score, recommendation_seed)

    disclaimers = (
        "This report is AI-generated for educational purposes only. "
        "It does not constitute financial, legal, or professional real estate advice. "
        "The ML model was trained on King County, WA data from 2014-2015 and may not "
        "reflect current market conditions. Always consult qualified professionals "
        "before making investment decisions."
    )

    return {
        "advisory_report": report,
        "recommendation": recommendation,
        "disclaimers": disclaimers,
        "iteration_count": iteration,
        "current_phase": "synthesis_complete",
        "error_log": errors,
        "messages": [
            SystemMessage(
                content=(
                    f"Advisory report synthesized (iteration {iteration}). "
                    f"Recommendation: {recommendation}."
                )
            )
        ],
    }


def _extract_recommendation(
    report: str,
    risk_level: str,
    inv_score: int,
    recommendation_seed: str | None = None,
) -> str:
    """Extract recommendation from report, with validation."""
    report_upper = report.upper()

    if recommendation_seed in ("STRONG_BUY", "BUY", "HOLD", "CAUTION", "AVOID"):
        if risk_level in ("HIGH", "VERY_HIGH") and recommendation_seed in ("STRONG_BUY", "BUY"):
            return "CAUTION"
        return recommendation_seed

    for rec in ["STRONG BUY", "STRONG_BUY"]:
        if rec in report_upper:
            if risk_level in ("HIGH", "VERY_HIGH"):
                return "CAUTION"
            return "STRONG_BUY"

    if "AVOID" in report_upper and "BUY" not in report_upper.split("AVOID")[0][-20:]:
        return "AVOID"
    if "CAUTION" in report_upper:
        return "CAUTION"
    if "HOLD" in report_upper:
        return "HOLD"
    if "BUY" in report_upper:
        if risk_level in ("HIGH", "VERY_HIGH"):
            return "CAUTION"
        return "BUY"

    if inv_score >= 75:
        return "STRONG_BUY" if risk_level == "LOW" else "BUY"
    if inv_score >= 60:
        return "BUY" if risk_level in ("LOW", "MODERATE") else "HOLD"
    if inv_score >= 45:
        return "HOLD"
    if inv_score >= 30:
        return "CAUTION"
    return "AVOID"


def _build_fallback_report(
    consultation,
    valuation,
    market_context,
    comparables_narrative,
    risk_assessment,
    neighborhood,
    negotiation,
    decision_lenses,
    decision_summary,
) -> str:
    """Structured fallback report when LLM is unavailable."""

    price = valuation.get("predicted_price", 0)
    confidence = valuation.get("confidence", 0)
    risk_level = risk_assessment.get("overall_risk", "Unknown")
    action_plan = decision_summary.get("next_steps", []) or negotiation.get("action_plan", [])
    decision_section = "\n".join(
        f"- {lens.get('lens', 'Lens')}: {lens.get('summary', '')}" for lens in decision_lenses
    ) or "Insufficient data for this section."

    return f"""## Client Brief

Here's my read as your property advisor. Client mode: **{consultation.get('client_mode', 'BUYER')}**. Objective: {consultation.get('objective', 'General decision support')}.
Budget: ${consultation.get('budget', 0):,.0f}. Asking price: ${consultation.get('asking_price', 0):,.0f}.
Risk tolerance: {consultation.get('risk_tolerance', 'BALANCED')}. Financing: {consultation.get('financing', 'MIXED')}.

## Property Valuation Summary

The pricing signal I would anchor on is **${price:,.0f}**, with model confidence of **{confidence:.1f}%**.
Price range: ${valuation.get('price_low', 0):,.0f} – ${valuation.get('price_high', 0):,.0f}.
Price per square foot: ${valuation.get('price_per_sqft', 0):,.0f}.
Market status: {valuation.get('market_status', 'Unknown')}.

## Market Context

{market_context}

## Comparable Sales Analysis

{comparables_narrative}

## Neighborhood Intelligence

Overall neighborhood score: **{neighborhood.get('overall_score', 0)}/100** with market heat **{neighborhood.get('market_heat', 'BALANCED')}**.
Livability: {neighborhood.get('livability_score', 0)}/100, liquidity: {neighborhood.get('liquidity_score', 0)}/100, upside: {neighborhood.get('upside_score', 0)}/100.
{neighborhood.get('narrative', 'Neighborhood analysis unavailable.')}

## Risk Assessment

Risk is the part I would not gloss over. Overall risk level: **{risk_level}** (Score: {risk_assessment.get('risk_score', 0):.1f}/100).

{risk_assessment.get('risk_narrative', 'Risk narrative unavailable.')}

## Negotiation Playbook

Anchor: **${negotiation.get('anchor_price', 0):,.0f}** | Target: **${negotiation.get('target_price', 0):,.0f}** | Walk-away: **${negotiation.get('walk_away_price', 0):,.0f}**

{negotiation.get('strategy_summary', 'Negotiation strategy unavailable.')}

## Decision Lens

{decision_section}

## Investment Recommendation

Based on an investment score of {valuation.get('investment_score', 0)}/100 and risk level of {risk_level}, my practical stance is **{valuation.get('investment_label', 'N/A')}**.

## Action Plan

{chr(10).join(f"- {step}" for step in action_plan) if action_plan else "- Review the pricing corridor and risk profile before proceeding."}

## Disclaimers

This report is AI-generated for educational purposes only. It does not constitute financial, legal, or professional real estate advice. The ML model was trained on King County, WA data from 2014-2015 and may not reflect current market conditions. Always consult qualified professionals before making investment decisions."""
