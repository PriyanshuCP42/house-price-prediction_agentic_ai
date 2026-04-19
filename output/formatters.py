"""
Formatters — Utility functions for rendering advisory data in Streamlit.
"""


def recommendation_badge_html(recommendation: str) -> str:
    """Returns styled HTML badge for the investment recommendation."""
    colors = {
        "STRONG_BUY": ("#065f46", "#d1fae5"),
        "BUY": ("#065f46", "#d1fae5"),
        "HOLD": ("#92400e", "#fef3c7"),
        "CAUTION": ("#9a3412", "#ffedd5"),
        "AVOID": ("#991b1b", "#fee2e2"),
    }
    text_color, bg_color = colors.get(recommendation, ("#1e293b", "#f3f4f6"))
    display = recommendation.replace("_", " ")
    return (
        f'<span style="background:{bg_color};color:{text_color};'
        f'padding:8px 20px;border-radius:20px;font-weight:700;font-size:16px;">'
        f'{display}</span>'
    )


def risk_level_color(risk_level: str) -> str:
    """Returns hex color for risk level."""
    return {
        "LOW": "#10b981",
        "MODERATE": "#f59e0b",
        "HIGH": "#f97316",
        "VERY_HIGH": "#ef4444",
    }.get(risk_level, "#6b7280")


def severity_emoji(severity: str) -> str:
    """Returns emoji for risk severity."""
    return {
        "LOW": "🟢",
        "MODERATE": "🟡",
        "HIGH": "🟠",
        "CRITICAL": "🔴",
    }.get(severity, "⚪")
