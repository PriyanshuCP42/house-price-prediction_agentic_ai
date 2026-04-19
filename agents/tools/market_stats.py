"""
Market Statistics Tool — Computes zipcode-level statistics from the dataset.
Used by the valuation agent to compare a property against its local market.
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
def get_zipcode_market_stats(zipcode: int) -> str:
    """Get comprehensive market statistics for a specific King County zipcode.
    Returns median price, average price, price range, sales volume, and more."""

    df = _load_dataset()
    zdf = df[df["zipcode"] == zipcode]

    if len(zdf) == 0:
        return f"No data available for zipcode {zipcode}."

    stats = {
        "zipcode": zipcode,
        "sales_count": len(zdf),
        "median_price": zdf["price"].median(),
        "mean_price": zdf["price"].mean(),
        "min_price": zdf["price"].min(),
        "max_price": zdf["price"].max(),
        "price_std": zdf["price"].std(),
        "avg_sqft": zdf["sqft_living"].mean(),
        "avg_grade": zdf["grade"].mean(),
        "avg_condition": zdf["condition"].mean(),
        "avg_price_per_sqft": zdf["price_per_sqft"].mean(),
        "pct_waterfront": zdf["waterfront"].mean() * 100,
        "avg_house_age": zdf["house_age"].mean(),
        "pct_renovated": zdf["renovated"].mean() * 100,
    }

    # Percentile rank of this zipcode among all zipcodes
    zip_medians = df.groupby("zipcode")["price"].median()
    rank = (zip_medians < stats["median_price"]).sum()
    total_zips = len(zip_medians)
    percentile = (rank / total_zips) * 100

    return (
        f"Market Statistics for Zipcode {zipcode}:\n"
        f"  Sales Volume: {stats['sales_count']} properties\n"
        f"  Median Price: ${stats['median_price']:,.0f}\n"
        f"  Mean Price: ${stats['mean_price']:,.0f}\n"
        f"  Price Range: ${stats['min_price']:,.0f} – ${stats['max_price']:,.0f}\n"
        f"  Price Std Dev: ${stats['price_std']:,.0f}\n"
        f"  Avg Living Area: {stats['avg_sqft']:,.0f} sqft\n"
        f"  Avg Grade: {stats['avg_grade']:.1f}/13\n"
        f"  Avg Condition: {stats['avg_condition']:.1f}/5\n"
        f"  Avg Price/Sqft: ${stats['avg_price_per_sqft']:,.0f}\n"
        f"  Waterfront Properties: {stats['pct_waterfront']:.1f}%\n"
        f"  Avg House Age: {stats['avg_house_age']:.0f} years\n"
        f"  Renovated Properties: {stats['pct_renovated']:.1f}%\n"
        f"  Zipcode Percentile Rank: {percentile:.0f}th (by median price among {total_zips} KC zipcodes)"
    )


def get_zipcode_stats_raw(zipcode: int) -> dict:
    """Returns raw statistics dict for internal use."""
    df = _load_dataset()
    zdf = df[df["zipcode"] == zipcode]
    if len(zdf) == 0:
        return {}
    return {
        "sales_count": len(zdf),
        "median_price": float(zdf["price"].median()),
        "mean_price": float(zdf["price"].mean()),
        "price_std": float(zdf["price"].std()),
        "avg_sqft": float(zdf["sqft_living"].mean()),
        "avg_grade": float(zdf["grade"].mean()),
        "avg_price_per_sqft": float(zdf["price_per_sqft"].mean()),
    }
