"""
LangGraph StateGraph — Wires all agent nodes with conditional routing,
human-in-the-loop checkpoints, and quality validation.

Graph topology:
    START → intake → valuation → [confidence routing] →
        (high conf) → market_analyst → comparables → risk → [risk routing] →
        (low conf)  → comparables → risk → [risk routing] →
            (very high risk) → human_review → synthesizer → [quality check] → END
            (else)           → synthesizer → [quality check] → END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from agents.state import AdvisoryState
from agents.nodes.intake_agent import intake_agent
from agents.nodes.valuation_agent import valuation_agent
from agents.nodes.market_analyst import market_analyst
from agents.nodes.comparables_agent import comparables_agent
from agents.nodes.risk_assessor import risk_assessor
from agents.nodes.neighborhood_analyst import neighborhood_analyst
from agents.nodes.negotiation_agent import negotiation_agent
from agents.nodes.decision_agent import decision_agent
from agents.nodes.advisory_synthesizer import advisory_synthesizer
from agents.nodes.human_review import human_review
from config.settings import CONFIDENCE_THRESHOLD, MAX_SYNTHESIZER_RETRIES


# --- Conditional edge functions ---

def route_by_confidence(state: AdvisoryState) -> str:
    """Skip RAG market analysis for low-confidence predictions.
    Low confidence means the model has little training data for this property
    type, so market context from RAG may be irrelevant."""
    valuation = state.get("valuation", {})
    confidence = valuation.get("confidence", 0)
    if confidence >= CONFIDENCE_THRESHOLD:
        return "market_analyst"
    return "comparables_agent"


def route_by_risk(state: AdvisoryState) -> str:
    """Route very high risk properties through human review checkpoint."""
    if state.get("needs_human_review", False):
        return "human_review"
    return "neighborhood_analyst"


def quality_check(state: AdvisoryState) -> str:
    """Verify the advisory report has all required sections.
    Retry synthesizer up to MAX_SYNTHESIZER_RETRIES times."""
    report = state.get("advisory_report", "")
    iteration = state.get("iteration_count", 0)

    required_sections = [
        "Client Brief",
        "Valuation Summary",
        "Market Context",
        "Neighborhood Intelligence",
        "Negotiation Playbook",
        "Recommendation",
    ]
    has_all = all(section.lower() in report.lower() for section in required_sections)

    if has_all or iteration >= MAX_SYNTHESIZER_RETRIES:
        return "end"
    return "advisory_synthesizer"


# --- Build the graph ---

def build_advisory_graph():
    """Construct and compile the LangGraph StateGraph."""

    graph = StateGraph(AdvisoryState)

    # Add nodes
    graph.add_node("intake_agent", intake_agent)
    graph.add_node("valuation_agent", valuation_agent)
    graph.add_node("market_analyst", market_analyst)
    graph.add_node("comparables_agent", comparables_agent)
    graph.add_node("risk_assessor", risk_assessor)
    graph.add_node("human_review", human_review)
    graph.add_node("neighborhood_analyst", neighborhood_analyst)
    graph.add_node("negotiation_agent", negotiation_agent)
    graph.add_node("decision_agent", decision_agent)
    graph.add_node("advisory_synthesizer", advisory_synthesizer)

    # Entry edge
    graph.add_edge(START, "intake_agent")

    # Conditional: intake → valuation (or END on failure)
    def route_after_intake(state: AdvisoryState) -> str:
        if state.get("current_phase") == "intake_failed":
            return "end"
        return "valuation_agent"

    graph.add_conditional_edges(
        "intake_agent",
        route_after_intake,
        {
            "end": END,
            "valuation_agent": "valuation_agent",
        },
    )

    # Conditional: valuation → market_analyst OR comparables
    graph.add_conditional_edges(
        "valuation_agent",
        route_by_confidence,
        {
            "market_analyst": "market_analyst",
            "comparables_agent": "comparables_agent",
        },
    )

    # market_analyst → comparables
    graph.add_edge("market_analyst", "comparables_agent")

    # comparables → risk_assessor
    graph.add_edge("comparables_agent", "risk_assessor")

    # Conditional: risk → human_review OR synthesizer
    graph.add_conditional_edges(
        "risk_assessor",
        route_by_risk,
        {
            "human_review": "human_review",
            "neighborhood_analyst": "neighborhood_analyst",
        },
    )

    # human_review → neighborhood analysis
    graph.add_edge("human_review", "neighborhood_analyst")

    # neighborhood analysis → negotiation
    graph.add_edge("neighborhood_analyst", "negotiation_agent")

    # negotiation → decision
    graph.add_edge("negotiation_agent", "decision_agent")

    # decision → synthesizer
    graph.add_edge("decision_agent", "advisory_synthesizer")

    # Conditional: synthesizer → END or retry
    graph.add_conditional_edges(
        "advisory_synthesizer",
        quality_check,
        {
            "end": END,
            "advisory_synthesizer": "advisory_synthesizer",
        },
    )

    # Compile with checkpointer for state persistence
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


# Module-level cached graph instance
_graph = None


def get_advisory_graph():
    """Returns a singleton compiled graph instance."""
    global _graph
    if _graph is None:
        _graph = build_advisory_graph()
    return _graph
