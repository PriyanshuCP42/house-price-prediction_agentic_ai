"""
ML Predictor Tool — Wraps the trained Random Forest model (model.pkl)
as LangChain tools that LangGraph agents can invoke.
"""

import pickle
import numpy as np
import pandas as pd
import datetime
from langchain_core.tools import tool
from config.settings import MODEL_PATH, DATA_PATH

# Load model artifacts once at import time
_artifacts = None


def _load_artifacts():
    global _artifacts
    if _artifacts is None:
        with open(MODEL_PATH, "rb") as f:
            _artifacts = pickle.load(f)
    return _artifacts


@tool
def predict_property_price(
    bedrooms: int,
    bathrooms: float,
    sqft_living: int,
    sqft_lot: int,
    floors: float,
    waterfront: int,
    view: int,
    condition: int,
    grade: int,
    sqft_above: int,
    sqft_basement: int,
    lat: float,
    long: float,
    sqft_living15: int,
    sqft_lot15: int,
    house_age: int,
    renovated: int,
    amenity_score: int,
    zipcode_encoded: float,
) -> str:
    """Predict the price of a property using the trained Random Forest model.
    Returns predicted price, confidence interval, and model confidence score.
    All 19 features must be provided exactly as the model expects."""

    arts = _load_artifacts()
    model = arts["model"]
    scaler = arts["scaler"]
    feature_names = arts["feature_names"]

    input_dict = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "lat": lat,
        "long": long,
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15,
        "house_age": house_age,
        "renovated": renovated,
        "amenity_score": amenity_score,
        "zipcode_encoded": zipcode_encoded,
    }

    feature_frame = pd.DataFrame([[input_dict.get(f, 0) for f in feature_names]], columns=feature_names)
    scaled = scaler.transform(feature_frame)
    tree_preds = np.array([t.predict(scaled)[0] for t in model.estimators_])
    price = float(tree_preds.mean())
    std = float(tree_preds.std())
    confidence = max(0.0, min(100.0, 100 - (std / max(price, 1) * 100)))
    price_low = max(0.0, price - std)
    price_high = price + std
    ppsf = price / max(sqft_living, 1)

    return (
        f"Predicted Price: ${price:,.0f}\n"
        f"Price Range: ${price_low:,.0f} – ${price_high:,.0f}\n"
        f"Model Confidence: {confidence:.1f}%\n"
        f"Standard Deviation: ${std:,.0f}\n"
        f"Price per sqft: ${ppsf:,.0f}"
    )


def predict_price_raw(input_dict: dict) -> tuple:
    """Direct prediction without tool decoration — for internal use."""
    arts = _load_artifacts()
    model = arts["model"]
    scaler = arts["scaler"]
    feature_names = arts["feature_names"]

    feature_frame = pd.DataFrame([[input_dict.get(f, 0) for f in feature_names]], columns=feature_names)
    scaled = scaler.transform(feature_frame)
    tree_preds = np.array([t.predict(scaled)[0] for t in model.estimators_])
    price = float(tree_preds.mean())
    std = float(tree_preds.std())
    confidence = max(0.0, min(100.0, 100 - (std / max(price, 1) * 100)))
    return price, max(0.0, price - std), price + std, confidence, std


def compute_investment_score(
    price: float,
    sqft: int,
    grade: int,
    condition: int,
    house_age: int,
    waterfront: int,
    view: int,
    renovated: int,
) -> int:
    """Heuristic investment scoring (19 rules, 0-100 scale)."""
    score = 50
    ppsf = price / max(sqft, 1)

    if ppsf < 150:
        score += 25
    elif ppsf < 200:
        score += 15
    elif ppsf < 250:
        score += 5
    elif ppsf > 320:
        score -= 20

    if grade >= 10:
        score += 12
    elif grade >= 8:
        score += 6
    elif grade <= 5:
        score -= 10

    if condition >= 4:
        score += 8
    elif condition <= 2:
        score -= 8

    if house_age <= 15:
        score += 10
    elif house_age > 60:
        score -= 12

    if waterfront:
        score += 15
    if view >= 3:
        score += 8
    elif view >= 1:
        score += 4
    if renovated:
        score += 6

    return max(0, min(100, score))


def compute_market_status(price: float, sqft: int) -> tuple[str, float]:
    """Returns (status_text, price_per_sqft)."""
    ppsf = price / max(sqft, 1)
    if ppsf < 180:
        return "Underpriced", ppsf
    elif ppsf < 290:
        return "Fair Price", ppsf
    else:
        return "Overpriced", ppsf


def get_feature_importances() -> dict[str, float]:
    """Returns feature importance scores from the Random Forest."""
    arts = _load_artifacts()
    feature_names = arts["feature_names"]
    importances = arts["model"].feature_importances_
    return dict(sorted(zip(feature_names, importances), key=lambda x: -x[1]))


def get_zipcode_mean() -> dict:
    """Returns the zipcode->mean price mapping from training data."""
    arts = _load_artifacts()
    return arts["zipcode_mean"]
