"""
Comparables Finder Tool — Wraps the comparable property matching logic
from Milestone 1 as a LangChain tool for LangGraph agents.
"""

import pandas as pd
import datetime
from langchain_core.tools import tool
from config.settings import DATA_PATH

_df = None


def _load_dataset() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(DATA_PATH)
        _df = _df[_df["bedrooms"] <= 10]
        _df["house_age"] = datetime.datetime.now().year - _df["yr_built"]
        _df["renovated"] = _df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)
        _df["amenity_score"] = (
            _df["waterfront"] + _df["view"] + _df["condition"] + _df["grade"]
        )
        _df["price_per_sqft"] = _df["price"] / _df["sqft_living"]
    return _df


@tool
def find_comparable_properties(
    zipcode: int, bedrooms: int, sqft_living: int, predicted_price: float, n: int = 6
) -> str:
    """Find comparable recently-sold properties matching the subject property.
    Matches on same zipcode, bedrooms +/-1, sqft_living +/-25%.
    Returns up to n most similar properties sorted by price proximity."""

    df = _load_dataset()
    mask = (
        (df["zipcode"] == zipcode)
        & (df["bedrooms"].between(bedrooms - 1, bedrooms + 1))
        & (df["sqft_living"].between(sqft_living * 0.75, sqft_living * 1.25))
    )
    sub = df[mask].copy()

    if len(sub) < 3:
        sub = df[df["zipcode"] == zipcode].copy()

    if len(sub) == 0:
        return "No comparable properties found in this zipcode."

    sub["diff"] = abs(sub["price"] - predicted_price)
    comps = sub.nsmallest(min(n, len(sub)), "diff")[
        [
            "price",
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "grade",
            "condition",
            "house_age",
            "price_per_sqft",
        ]
    ].reset_index(drop=True)

    result_lines = [f"Found {len(comps)} comparable properties in zipcode {zipcode}:\n"]
    for i, row in comps.iterrows():
        diff_pct = ((row["price"] - predicted_price) / max(predicted_price, 1)) * 100
        sign = "+" if diff_pct > 0 else ""
        result_lines.append(
            f"  Comp {i+1}: ${row['price']:,.0f} | {int(row['bedrooms'])}BR/{row['bathrooms']}BA | "
            f"{int(row['sqft_living'])}sqft | Grade {int(row['grade'])} | "
            f"Age {int(row['house_age'])}yr | ${row['price_per_sqft']:,.0f}/sqft | "
            f"{sign}{diff_pct:.1f}% vs predicted"
        )

    avg_price = comps["price"].mean()
    median_price = comps["price"].median()
    result_lines.append(f"\nComparable Average: ${avg_price:,.0f}")
    result_lines.append(f"Comparable Median: ${median_price:,.0f}")
    result_lines.append(f"Subject Predicted: ${predicted_price:,.0f}")
    diff_from_avg = ((predicted_price - avg_price) / max(avg_price, 1)) * 100
    result_lines.append(
        f"Subject vs Average: {'+' if diff_from_avg > 0 else ''}{diff_from_avg:.1f}%"
    )

    return "\n".join(result_lines)


def find_comparables_raw(
    zipcode: int, bedrooms: int, sqft_living: int, predicted_price: float, n: int = 6
) -> pd.DataFrame:
    """Direct comparable lookup — returns DataFrame for internal use."""
    df = _load_dataset()
    mask = (
        (df["zipcode"] == zipcode)
        & (df["bedrooms"].between(bedrooms - 1, bedrooms + 1))
        & (df["sqft_living"].between(sqft_living * 0.75, sqft_living * 1.25))
    )
    sub = df[mask].copy()
    if len(sub) < 3:
        sub = df[df["zipcode"] == zipcode].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    sub["diff"] = abs(sub["price"] - predicted_price)
    return sub.nsmallest(min(n, len(sub)), "diff")[
        [
            "price",
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "grade",
            "condition",
            "house_age",
            "price_per_sqft",
        ]
    ].reset_index(drop=True)
