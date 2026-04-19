"""
Human Review Node — LangGraph interrupt checkpoint for VERY_HIGH risk properties.
Uses LangGraph's NodeInterrupt to pause the graph. The Streamlit UI catches this
and shows a warning banner with risk details and a 'Proceed' button.
"""

from langchain_core.messages import SystemMessage
from langgraph.errors import NodeInterrupt
from agents.state import AdvisoryState


def human_review(state: AdvisoryState) -> dict:
    """Interrupt the graph for human review on VERY_HIGH risk properties.

    Raises NodeInterrupt which pauses graph execution. The Streamlit app
    catches this, displays the risk warning, and resumes the graph when
    the user clicks 'Proceed'.
    """

    risk = state.get("risk_assessment", {})
    risk_level = risk.get("overall_risk", "UNKNOWN")
    risk_score = risk.get("risk_score", 0)

    # Build summary of top risk factors
    top_factors = sorted(
        risk.get("risk_factors", []),
        key=lambda f: -f.get("score", 0),
    )[:3]

    warning_lines = [
        f"HUMAN REVIEW REQUIRED — Risk Level: {risk_level} (Score: {risk_score:.1f}/100)",
        "",
        "Top risk factors:",
    ]
    for f in top_factors:
        warning_lines.append(
            f"  - [{f.get('severity', '?')}] {f.get('factor', '?')}: {f.get('explanation', '')}"
        )

    warning_message = "\n".join(warning_lines)

    # Raise NodeInterrupt to actually pause graph execution
    raise NodeInterrupt(warning_message)
