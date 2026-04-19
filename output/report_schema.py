"""
Pydantic models for structured advisory output validation.
Ensures every report has all required sections and valid fields.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class RiskFactorOutput(BaseModel):
    factor: str = Field(description="Name of the risk factor")
    severity: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    score: float = Field(ge=0, le=12.5, description="Score out of 12.5")
    explanation: str
    mitigation: str


class ValuationOutput(BaseModel):
    predicted_price: float = Field(ge=0)
    price_low: float = Field(ge=0)
    price_high: float = Field(ge=0)
    confidence: float = Field(ge=0, le=100)
    price_per_sqft: float = Field(ge=0)
    market_status: str
    investment_score: int = Field(ge=0, le=100)
    investment_label: str


class ConsultationOutput(BaseModel):
    client_mode: Literal["BUYER", "SELLER", "INVESTOR"]
    objective: str
    budget: float = Field(ge=0)
    asking_price: float = Field(ge=0)
    target_hold_years: int = Field(ge=0)
    risk_tolerance: Literal["CONSERVATIVE", "BALANCED", "AGGRESSIVE"]
    financing: Literal["CASH", "MORTGAGE", "MIXED"]
    monthly_rent_estimate: float = Field(ge=0)
    renovation_budget: float = Field(ge=0)
    must_haves: list[str] = Field(default_factory=list)
    raw_notes: str = ""


class NeighborhoodOutput(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    livability_score: int = Field(ge=0, le=100)
    liquidity_score: int = Field(ge=0, le=100)
    upside_score: int = Field(ge=0, le=100)
    rental_demand_score: int = Field(ge=0, le=100)
    pricing_power_score: int = Field(ge=0, le=100)
    market_heat: Literal["COOL", "BALANCED", "HOT"]
    highlights: list[str] = Field(default_factory=list)
    narrative: str


class NegotiationOutput(BaseModel):
    anchor_price: float = Field(ge=0)
    target_price: float = Field(ge=0)
    walk_away_price: float = Field(ge=0)
    leverage_points: list[str] = Field(default_factory=list)
    caution_points: list[str] = Field(default_factory=list)
    action_plan: list[str] = Field(default_factory=list)
    strategy_summary: str


class DecisionLensOutput(BaseModel):
    lens: Literal["Bull Case", "Base Case", "Risk Case"]
    stance: str
    summary: str


class DecisionSummaryOutput(BaseModel):
    headline: str
    executive_summary: str
    next_steps: list[str] = Field(default_factory=list)


class AdvisoryReport(BaseModel):
    """Complete structured advisory report output."""
    consultation: ConsultationOutput
    valuation: ValuationOutput
    market_context: str = Field(min_length=10)
    rag_sources: list[str] = Field(default_factory=list)
    comparables_narrative: str = Field(min_length=10)
    num_comparables: int = Field(ge=0)
    risk_level: Literal["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    risk_score: float = Field(ge=0, le=100)
    risk_factors: list[RiskFactorOutput] = Field(default_factory=list)
    risk_narrative: str
    neighborhood: NeighborhoodOutput
    negotiation: NegotiationOutput
    decision_lenses: list[DecisionLensOutput] = Field(default_factory=list)
    decision_summary: DecisionSummaryOutput
    recommendation: Literal["STRONG_BUY", "BUY", "HOLD", "CAUTION", "AVOID"]
    advisory_markdown: str = Field(min_length=50, description="Full markdown report")
    disclaimers: str
    errors: list[str] = Field(default_factory=list)


def build_report_from_state(state: dict) -> AdvisoryReport:
    """Convert LangGraph final state into a validated AdvisoryReport."""
    valuation = state.get("valuation", {})
    risk = state.get("risk_assessment", {})
    consultation = state.get("consultation_context", {})
    neighborhood = state.get("neighborhood_analysis", {})
    negotiation = state.get("negotiation_strategy", {})
    decision_summary = state.get("decision_summary", {})

    return AdvisoryReport(
        consultation=ConsultationOutput(
            client_mode=consultation.get("client_mode", "BUYER"),
            objective=consultation.get("objective", "General purchase review"),
            budget=consultation.get("budget", 0),
            asking_price=consultation.get("asking_price", 0),
            target_hold_years=consultation.get("target_hold_years", 0),
            risk_tolerance=consultation.get("risk_tolerance", "BALANCED"),
            financing=consultation.get("financing", "MIXED"),
            monthly_rent_estimate=consultation.get("monthly_rent_estimate", 0),
            renovation_budget=consultation.get("renovation_budget", 0),
            must_haves=consultation.get("must_haves", []),
            raw_notes=consultation.get("raw_notes", ""),
        ),
        valuation=ValuationOutput(
            predicted_price=valuation.get("predicted_price", 0),
            price_low=valuation.get("price_low", 0),
            price_high=valuation.get("price_high", 0),
            confidence=valuation.get("confidence", 0),
            price_per_sqft=valuation.get("price_per_sqft", 0),
            market_status=valuation.get("market_status", "Unknown"),
            investment_score=valuation.get("investment_score", 0),
            investment_label=valuation.get("investment_label", "N/A"),
        ),
        market_context=state.get("market_context", "Not available"),
        rag_sources=state.get("rag_sources", []),
        comparables_narrative=state.get("comparables_narrative", "Not available"),
        num_comparables=len(state.get("comparables", [])),
        risk_level=risk.get("overall_risk", "MODERATE"),
        risk_score=risk.get("risk_score", 0),
        risk_factors=[
            RiskFactorOutput(**f) for f in risk.get("risk_factors", [])
        ],
        risk_narrative=risk.get("risk_narrative", "Not available"),
        neighborhood=NeighborhoodOutput(
            overall_score=neighborhood.get("overall_score", 50),
            livability_score=neighborhood.get("livability_score", 50),
            liquidity_score=neighborhood.get("liquidity_score", 50),
            upside_score=neighborhood.get("upside_score", 50),
            rental_demand_score=neighborhood.get("rental_demand_score", 50),
            pricing_power_score=neighborhood.get("pricing_power_score", 50),
            market_heat=neighborhood.get("market_heat", "BALANCED"),
            highlights=neighborhood.get("highlights", []),
            narrative=neighborhood.get("narrative", "Neighborhood analysis not available."),
        ),
        negotiation=NegotiationOutput(
            anchor_price=negotiation.get("anchor_price", 0),
            target_price=negotiation.get("target_price", 0),
            walk_away_price=negotiation.get("walk_away_price", 0),
            leverage_points=negotiation.get("leverage_points", []),
            caution_points=negotiation.get("caution_points", []),
            action_plan=negotiation.get("action_plan", []),
            strategy_summary=negotiation.get("strategy_summary", "Negotiation strategy not available."),
        ),
        decision_lenses=[
            DecisionLensOutput(**lens) for lens in state.get("decision_lenses", [])
        ],
        decision_summary=DecisionSummaryOutput(
            headline=decision_summary.get("headline", "Property decision overview"),
            executive_summary=decision_summary.get("executive_summary", "Decision summary not available."),
            next_steps=decision_summary.get("next_steps", []),
        ),
        recommendation=state.get("recommendation", "HOLD"),
        advisory_markdown=state.get("advisory_report", "Report generation failed."),
        disclaimers=state.get("disclaimers", ""),
        errors=state.get("error_log", []),
    )
