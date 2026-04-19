"""
Risk Calculator Tool — 8-dimensional risk assessment for property investment.
Each dimension scores 0-12.5, total risk score is 0-100.
"""

from langchain_core.tools import tool
from agents.tools.market_stats import get_zipcode_stats_raw


@tool
def calculate_risk_factors(
    predicted_price: float,
    confidence: float,
    house_age: int,
    grade: int,
    condition: int,
    sqft_living: int,
    sqft_lot: int,
    renovated: int,
    waterfront: int,
    zipcode: int,
    comp_avg_price: float,
) -> str:
    """Calculate 8 investment risk dimensions for a property.
    Each factor scores 0-12.5. Total risk score is 0-100.
    Higher score = higher risk."""

    factors = compute_risk_factors_raw(
        predicted_price, confidence, house_age, grade, condition,
        sqft_living, sqft_lot, renovated, waterfront, zipcode, comp_avg_price
    )

    lines = [f"RISK ASSESSMENT (Total Score: {factors['total_score']:.1f}/100)\n"]
    lines.append(f"Overall Risk Level: {factors['overall_risk']}\n")

    for f in factors["factors"]:
        lines.append(
            f"  [{f['severity']}] {f['factor']}: {f['score']:.1f}/12.5\n"
            f"    {f['explanation']}\n"
            f"    Mitigation: {f['mitigation']}"
        )

    return "\n".join(lines)


def compute_risk_factors_raw(
    predicted_price: float,
    confidence: float,
    house_age: int,
    grade: int,
    condition: int,
    sqft_living: int,
    sqft_lot: int,
    renovated: int,
    waterfront: int,
    zipcode: int,
    comp_avg_price: float,
) -> dict:
    """Returns structured risk assessment for internal use."""

    factors = []
    zip_stats = get_zipcode_stats_raw(zipcode)

    # 1. Price vs Comparables Deviation
    if comp_avg_price > 0:
        deviation = abs(predicted_price - comp_avg_price) / comp_avg_price * 100
        if deviation > 25:
            score, sev = 12.5, "CRITICAL"
            expl = f"Predicted price deviates {deviation:.0f}% from comparable average (${comp_avg_price:,.0f})"
        elif deviation > 15:
            score, sev = 8.0, "HIGH"
            expl = f"Predicted price deviates {deviation:.0f}% from comparable average"
        elif deviation > 8:
            score, sev = 4.0, "MODERATE"
            expl = f"Predicted price deviates {deviation:.0f}% from comparable average"
        else:
            score, sev = 1.0, "LOW"
            expl = f"Predicted price aligns well with comparables ({deviation:.0f}% deviation)"
    else:
        score, sev = 8.0, "HIGH"
        expl = "No comparable data available for price validation"

    factors.append({
        "factor": "Price vs Comparables",
        "severity": sev,
        "score": score,
        "explanation": expl,
        "mitigation": "Get a professional appraisal to validate the predicted price."
    })

    # 2. Model Confidence
    if confidence < 30:
        score, sev = 12.5, "CRITICAL"
        expl = f"Model confidence is very low ({confidence:.0f}%) — prediction unreliable"
    elif confidence < 50:
        score, sev = 8.0, "HIGH"
        expl = f"Model confidence is below average ({confidence:.0f}%)"
    elif confidence < 70:
        score, sev = 4.0, "MODERATE"
        expl = f"Model confidence is moderate ({confidence:.0f}%)"
    else:
        score, sev = 1.0, "LOW"
        expl = f"Model confidence is high ({confidence:.0f}%)"

    factors.append({
        "factor": "Model Confidence",
        "severity": sev,
        "score": score,
        "explanation": expl,
        "mitigation": "Consider additional data sources for high-uncertainty predictions."
    })

    # 3. Property Age
    if house_age > 80:
        score, sev = 12.5, "CRITICAL"
        expl = f"Property is {house_age} years old — significant structural concerns"
    elif house_age > 50:
        score, sev = 8.0, "HIGH"
        expl = f"Property is {house_age} years old — likely needs major systems updates"
    elif house_age > 30:
        score, sev = 4.0, "MODERATE"
        expl = f"Property is {house_age} years old — moderate wear expected"
    else:
        score, sev = 1.0, "LOW"
        expl = f"Property is {house_age} years old — relatively new"

    factors.append({
        "factor": "Property Age",
        "severity": sev,
        "score": score,
        "explanation": expl,
        "mitigation": "Budget for potential renovation and maintenance costs."
    })

    # 4. Grade/Condition Mismatch
    if grade <= 5 and predicted_price > 400000:
        score, sev = 10.0, "HIGH"
        expl = f"Low grade ({grade}/13) for a ${predicted_price:,.0f} property"
    elif condition <= 2 and grade >= 8:
        score, sev = 8.0, "HIGH"
        expl = f"Poor condition ({condition}/5) despite decent grade ({grade}/13)"
    elif condition <= 3 and grade <= 6:
        score, sev = 5.0, "MODERATE"
        expl = f"Below-average condition ({condition}/5) and grade ({grade}/13)"
    else:
        score, sev = 1.5, "LOW"
        expl = f"Grade ({grade}/13) and condition ({condition}/5) are consistent"

    factors.append({
        "factor": "Grade-Condition Alignment",
        "severity": sev,
        "score": score,
        "explanation": expl,
        "mitigation": "Inspect the property for hidden defects that may not match grade rating."
    })

    # 5. Lot Size Anomaly
    if zip_stats and zip_stats.get("avg_sqft"):
        avg_sqft = zip_stats["avg_sqft"]
        sqft_ratio = sqft_living / avg_sqft
        if sqft_ratio < 0.5 or sqft_ratio > 2.0:
            score, sev = 8.0, "HIGH"
            expl = f"Living area ({sqft_living}sqft) is significantly different from zipcode avg ({avg_sqft:.0f}sqft)"
        elif sqft_ratio < 0.7 or sqft_ratio > 1.5:
            score, sev = 4.0, "MODERATE"
            expl = f"Living area differs from zipcode average ({sqft_living} vs {avg_sqft:.0f}sqft)"
        else:
            score, sev = 1.0, "LOW"
            expl = f"Living area aligns with zipcode norms ({sqft_living} vs {avg_sqft:.0f}sqft)"
    else:
        score, sev = 3.0, "MODERATE"
        expl = "Unable to compare against zipcode averages"

    factors.append({
        "factor": "Size vs Neighborhood",
        "severity": sev,
        "score": score,
        "explanation": expl,
        "mitigation": "Verify the property fits the neighborhood profile for resale value."
    })

    # 6. Renovation Gap
    if house_age > 40 and not renovated:
        score, sev = 10.0, "HIGH"
        expl = f"Property is {house_age} years old with no renovation on record"
    elif house_age > 25 and not renovated:
        score, sev = 5.0, "MODERATE"
        expl = f"Property is {house_age} years old without recorded renovation"
    else:
        score, sev = 1.0, "LOW"
        expl = "Property is either recently built or has been renovated"

    factors.append({
        "factor": "Renovation Gap",
        "severity": sev,
        "score": score,
        "explanation": expl,
        "mitigation": "Factor renovation costs into total investment budget."
    })

    # 7. Zipcode Price Volatility
    if zip_stats and zip_stats.get("price_std") and zip_stats.get("median_price"):
        cv = zip_stats["price_std"] / zip_stats["median_price"]
        if cv > 0.8:
            score, sev = 10.0, "HIGH"
            expl = f"High price volatility in zipcode {zipcode} (CV={cv:.2f})"
        elif cv > 0.5:
            score, sev = 5.0, "MODERATE"
            expl = f"Moderate price volatility in zipcode {zipcode} (CV={cv:.2f})"
        else:
            score, sev = 1.5, "LOW"
            expl = f"Stable pricing in zipcode {zipcode} (CV={cv:.2f})"
    else:
        score, sev = 4.0, "MODERATE"
        expl = "Insufficient data to assess price stability"

    factors.append({
        "factor": "Zipcode Price Volatility",
        "severity": sev,
        "score": score,
        "explanation": expl,
        "mitigation": "Research recent sales trends and economic factors for this area."
    })

    # 8. Data Staleness
    score, sev = 6.0, "MODERATE"
    expl = "Model trained on 2014-2015 data — market conditions may have changed significantly"
    factors.append({
        "factor": "Data Staleness",
        "severity": sev,
        "score": score,
        "explanation": expl,
        "mitigation": "Cross-reference with current market data from real estate platforms."
    })

    total_score = sum(f["score"] for f in factors)
    if total_score >= 75:
        overall = "VERY_HIGH"
    elif total_score >= 50:
        overall = "HIGH"
    elif total_score >= 30:
        overall = "MODERATE"
    else:
        overall = "LOW"

    return {
        "overall_risk": overall,
        "total_score": total_score,
        "factors": factors,
    }
