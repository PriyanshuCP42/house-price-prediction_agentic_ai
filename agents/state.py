"""
AdvisoryState — Central state schema for the LangGraph multi-agent pipeline.
Every agent reads from and writes to this shared typed state.
"""

from typing import TypedDict, Optional, Literal, Annotated
from langgraph.graph.message import add_messages


class PropertyInput(TypedDict):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int
    house_age: int
    renovated: int
    amenity_score: int
    zipcode_encoded: float
    zipcode: int
    yr_built: int


class ConsultationContext(TypedDict):
    client_mode: Literal["BUYER", "SELLER", "INVESTOR"]
    objective: str
    budget: float
    asking_price: float
    target_hold_years: int
    risk_tolerance: Literal["CONSERVATIVE", "BALANCED", "AGGRESSIVE"]
    financing: Literal["CASH", "MORTGAGE", "MIXED"]
    monthly_rent_estimate: float
    renovation_budget: float
    must_haves: list[str]
    raw_notes: str


class ValuationResult(TypedDict):
    predicted_price: float
    price_low: float
    price_high: float
    confidence: float
    std_dev: float
    investment_score: int
    investment_label: str
    market_status: str
    price_per_sqft: float
    valuation_explanation: str


class ComparableProperty(TypedDict):
    price: float
    bedrooms: int
    bathrooms: float
    sqft_living: int
    grade: int
    condition: int
    house_age: int
    price_per_sqft: float


class RiskFactor(TypedDict):
    factor: str
    severity: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    score: float
    explanation: str
    mitigation: str


class RiskAssessment(TypedDict):
    overall_risk: Literal["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    risk_score: float
    risk_factors: list[RiskFactor]
    risk_narrative: str


class NeighborhoodAnalysis(TypedDict):
    overall_score: int
    livability_score: int
    liquidity_score: int
    upside_score: int
    rental_demand_score: int
    pricing_power_score: int
    market_heat: Literal["COOL", "BALANCED", "HOT"]
    highlights: list[str]
    narrative: str


class NegotiationStrategy(TypedDict):
    anchor_price: float
    target_price: float
    walk_away_price: float
    leverage_points: list[str]
    caution_points: list[str]
    action_plan: list[str]
    strategy_summary: str


class DecisionLens(TypedDict):
    lens: Literal["Bull Case", "Base Case", "Risk Case"]
    stance: str
    summary: str


class DecisionSummary(TypedDict):
    headline: str
    executive_summary: str
    next_steps: list[str]


class AdvisoryState(TypedDict):
    # --- User Input ---
    property_input: Optional[PropertyInput]
    consultation_context: Optional[ConsultationContext]
    user_query: str

    # --- Message trail for LangGraph ---
    messages: Annotated[list, add_messages]

    # --- Valuation (set by valuation_agent) ---
    valuation: Optional[ValuationResult]

    # --- Market Context (set by market_analyst via RAG) ---
    market_context: Optional[str]
    rag_sources: Optional[list[str]]

    # --- Comparables (set by comparables_agent) ---
    comparables: Optional[list[ComparableProperty]]
    comparables_narrative: Optional[str]

    # --- Risk (set by risk_assessor) ---
    risk_assessment: Optional[RiskAssessment]

    # --- Marketability & Strategy ---
    neighborhood_analysis: Optional[NeighborhoodAnalysis]
    negotiation_strategy: Optional[NegotiationStrategy]
    decision_lenses: Optional[list[DecisionLens]]
    decision_summary: Optional[DecisionSummary]

    # --- Final Output (set by advisory_synthesizer) ---
    advisory_report: Optional[str]
    recommendation: Optional[Literal["STRONG_BUY", "BUY", "HOLD", "CAUTION", "AVOID"]]
    disclaimers: Optional[str]

    # --- Control Flow ---
    needs_human_review: bool
    iteration_count: int
    error_log: list[str]
    current_phase: str
