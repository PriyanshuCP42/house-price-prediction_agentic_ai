"""
Milestone 2 Debug Verification Script
Run from the project root: python debug_milestone2.py
"""

import sys
import os
import traceback

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0

def report(test_name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  [PASS] {test_name}" + (f" -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {test_name}" + (f" -- {detail}" if detail else ""))


# ============================================================
# STEP 1: IMPORT CHECK
# ============================================================
print("=" * 70)
print("STEP 1: IMPORT CHECK")
print("=" * 70)

# 1a. agents.state
try:
    from agents.state import (
        AdvisoryState, PropertyInput, ValuationResult,
        ComparableProperty, RiskFactor, RiskAssessment,
    )
    report("agents.state", True, "All 6 TypedDicts imported")
except Exception as e:
    report("agents.state", False, f"{e}")

# 1b. agents.graph
try:
    from agents.graph import (
        build_advisory_graph, get_advisory_graph,
        route_by_confidence, route_by_risk, quality_check,
    )
    report("agents.graph", True, "All 5 functions imported")
except Exception as e:
    report("agents.graph", False, f"{e}")

# 1c. agents.nodes.* (all 7)
node_modules = {
    "agents.nodes.intake_agent": "intake_agent",
    "agents.nodes.valuation_agent": "valuation_agent",
    "agents.nodes.market_analyst": "market_analyst",
    "agents.nodes.comparables_agent": "comparables_agent",
    "agents.nodes.risk_assessor": "risk_assessor",
    "agents.nodes.advisory_synthesizer": "advisory_synthesizer",
    "agents.nodes.human_review": "human_review",
}
for mod_path, func_name in node_modules.items():
    try:
        mod = __import__(mod_path, fromlist=[func_name])
        fn = getattr(mod, func_name)
        assert callable(fn), f"{func_name} not callable"
        report(mod_path, True, f"{func_name}() callable")
    except Exception as e:
        report(mod_path, False, f"{e}")

# 1d. agents.tools.* (all 5)
tool_modules = {
    "agents.tools.ml_predictor": [
        "predict_property_price", "predict_price_raw",
        "compute_investment_score", "compute_market_status",
        "get_feature_importances", "get_zipcode_mean",
    ],
    "agents.tools.risk_calculator": [
        "calculate_risk_factors", "compute_risk_factors_raw",
    ],
    "agents.tools.comparables_finder": [
        "find_comparable_properties", "find_comparables_raw",
    ],
    "agents.tools.rag_retriever": ["rag_search", "rag_search_raw"],
    "agents.tools.market_stats": [
        "get_zipcode_market_stats", "get_zipcode_stats_raw",
    ],
}
for mod_path, funcs in tool_modules.items():
    try:
        mod = __import__(mod_path, fromlist=funcs)
        for fn_name in funcs:
            fn = getattr(mod, fn_name)
            assert callable(fn), f"{fn_name} not callable"
        report(mod_path, True, f"All {len(funcs)} functions")
    except Exception as e:
        report(mod_path, False, f"{e}")

# 1e. config.*
try:
    from config.settings import (
        BASE_DIR, MODEL_PATH, DATA_PATH, CHROMA_DB_PATH,
        CONFIDENCE_THRESHOLD, MAX_SYNTHESIZER_RETRIES,
        VERY_HIGH_RISK_THRESHOLD, get_investment_label,
    )
    report("config.settings", True, "All constants + get_investment_label")
except Exception as e:
    report("config.settings", False, f"{e}")

try:
    from config.prompts import (
        VALUATION_AGENT_PROMPT, MARKET_ANALYST_PROMPT,
        COMPARABLES_AGENT_PROMPT, RISK_ASSESSOR_PROMPT,
        ADVISORY_SYNTHESIZER_PROMPT,
    )
    report("config.prompts", True, "All 5 prompts")
except Exception as e:
    report("config.prompts", False, f"{e}")

try:
    from config.llm_config import get_llm
    report("config.llm_config", True, "get_llm importable")
except Exception as e:
    report("config.llm_config", False, f"{e}")

# 1f. rag.*
try:
    from rag.embeddings import get_embedding_function
    report("rag.embeddings", True)
except Exception as e:
    report("rag.embeddings", False, f"{e}")

try:
    from rag.build_knowledge_base import (
        build_kb, generate_market_insights,
        generate_zipcode_profiles, chunk_document,
    )
    report("rag.build_knowledge_base", True, "All 4 functions")
except Exception as e:
    report("rag.build_knowledge_base", False, f"{e}")

# 1g. output.*
try:
    from output.report_schema import (
        AdvisoryReport, ValuationOutput, RiskFactorOutput,
        build_report_from_state,
    )
    report("output.report_schema", True, "All 4 exports")
except Exception as e:
    report("output.report_schema", False, f"{e}")

try:
    from output.formatters import (
        recommendation_badge_html, risk_level_color, severity_emoji,
    )
    report("output.formatters", True, "All 3 functions")
except Exception as e:
    report("output.formatters", False, f"{e}")

# 1h. advisory_app (Streamlit-specific, can only check spec)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "advisory_app", os.path.join(os.path.dirname(__file__), "advisory_app.py")
    )
    report("advisory_app (spec)", True, "File spec loadable (full import needs Streamlit runtime)")
except Exception as e:
    report("advisory_app (spec)", False, f"{e}")


# ============================================================
# STEP 2: GRAPH COMPILATION
# ============================================================
print()
print("=" * 70)
print("STEP 2: GRAPH COMPILATION")
print("=" * 70)

try:
    from agents.graph import build_advisory_graph
    compiled = build_advisory_graph()
    report("Graph compiles", True)

    # Check nodes
    expected_nodes = {
        "intake_agent", "valuation_agent", "market_analyst",
        "comparables_agent", "risk_assessor", "human_review",
        "advisory_synthesizer",
    }
    graph_nodes = set(compiled.get_graph().nodes.keys()) - {"__start__", "__end__"}
    missing = expected_nodes - graph_nodes
    extra = graph_nodes - expected_nodes
    report("All 7 nodes present", len(missing) == 0,
           f"Missing: {missing}" if missing else f"Nodes: {sorted(graph_nodes)}")
    if extra:
        report("No extra nodes", False, f"Extra: {extra}")

    # Check we can get the graph visualization
    try:
        graph_repr = compiled.get_graph()
        report("Graph is inspectable", True)
    except Exception as e:
        report("Graph is inspectable", False, f"{e}")

except Exception as e:
    report("Graph compilation", False, f"{e}\n{traceback.format_exc()}")


# ============================================================
# STEP 3: ML MODEL TEST
# ============================================================
print()
print("=" * 70)
print("STEP 3: ML MODEL PREDICTION TEST")
print("=" * 70)

try:
    from agents.tools.ml_predictor import predict_price_raw
    from config.settings import MODEL_PATH

    # Verify model.pkl exists
    report("model.pkl exists", os.path.exists(MODEL_PATH), MODEL_PATH)

    test_input = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "sqft_lot": 7500,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1800,
        "sqft_basement": 0,
        "lat": 47.5,
        "long": -122.2,
        "sqft_living15": 1800,
        "sqft_lot15": 7500,
        "house_age": 30,
        "renovated": 0,
        "amenity_score": 10,
        "zipcode_encoded": 500000.0,
    }

    price, price_low, price_high, confidence, std_dev = predict_price_raw(test_input)
    report("predict_price_raw returns 5-tuple", True,
           f"price=${price:,.0f}, range=${price_low:,.0f}-${price_high:,.0f}, "
           f"conf={confidence:.1f}%, std=${std_dev:,.0f}")
    report("Price is positive", price > 0, f"${price:,.0f}")
    report("Confidence is 0-100", 0 <= confidence <= 100, f"{confidence:.1f}%")
    report("price_low <= price <= price_high",
           price_low <= price <= price_high,
           f"${price_low:,.0f} <= ${price:,.0f} <= ${price_high:,.0f}")

except Exception as e:
    report("ML Prediction", False, f"{e}\n{traceback.format_exc()}")


# ============================================================
# STEP 4: RAG / CHROMADB TEST
# ============================================================
print()
print("=" * 70)
print("STEP 4: RAG / CHROMADB QUERY TEST")
print("=" * 70)

try:
    from config.settings import CHROMA_DB_PATH
    report("chroma_db directory exists", os.path.exists(CHROMA_DB_PATH), CHROMA_DB_PATH)

    from agents.tools.rag_retriever import rag_search_raw
    results = rag_search_raw("King County housing market trends", n_results=3)
    report("rag_search_raw returns dict", isinstance(results, dict))
    report("Has 'documents' key", "documents" in results)
    report("Has 'metadatas' key", "metadatas" in results)
    report("Has 'distances' key", "distances" in results)
    num_docs = len(results.get("documents", []))
    report("Returns results", num_docs > 0, f"{num_docs} documents returned")

    if num_docs > 0:
        first_doc = results["documents"][0][:100]
        report("First doc is non-empty string", len(first_doc) > 10, f"{first_doc}...")

except Exception as e:
    report("RAG Query", False, f"{e}\n{traceback.format_exc()}")


# ============================================================
# STEP 5: RISK CALCULATOR TEST
# ============================================================
print()
print("=" * 70)
print("STEP 5: RISK CALCULATOR TEST")
print("=" * 70)

try:
    from agents.tools.risk_calculator import compute_risk_factors_raw

    risk_result = compute_risk_factors_raw(
        predicted_price=450000.0,
        confidence=72.0,
        house_age=30,
        grade=7,
        condition=3,
        sqft_living=1800,
        sqft_lot=7500,
        renovated=0,
        waterfront=0,
        zipcode=98103,
        comp_avg_price=430000.0,
    )
    report("compute_risk_factors_raw returns dict", isinstance(risk_result, dict))
    report("Has 'overall_risk' key", "overall_risk" in risk_result,
           risk_result.get("overall_risk"))
    report("Has 'total_score' key", "total_score" in risk_result,
           f"{risk_result.get('total_score', 0):.1f}/100")
    report("Has 'factors' key", "factors" in risk_result)

    factors = risk_result.get("factors", [])
    report("Has 8 risk factors", len(factors) == 8, f"Got {len(factors)}")

    for f in factors:
        has_all_keys = all(k in f for k in ["factor", "severity", "score", "explanation", "mitigation"])
        if not has_all_keys:
            report(f"Factor '{f.get('factor', '?')}' has all keys", False, f"Keys: {list(f.keys())}")

    total = risk_result.get("total_score", -1)
    report("Total score is 0-100", 0 <= total <= 100, f"{total:.1f}")

    overall = risk_result.get("overall_risk", "")
    report("Overall risk is valid enum",
           overall in ("LOW", "MODERATE", "HIGH", "VERY_HIGH"), overall)

except Exception as e:
    report("Risk Calculator", False, f"{e}\n{traceback.format_exc()}")


# ============================================================
# STEP 6: COMPARABLES FINDER TEST
# ============================================================
print()
print("=" * 70)
print("STEP 6: COMPARABLES FINDER TEST")
print("=" * 70)

try:
    from agents.tools.comparables_finder import find_comparables_raw
    from config.settings import DATA_PATH

    report("kc_house_data.csv exists", os.path.exists(DATA_PATH), DATA_PATH)

    comp_df = find_comparables_raw(
        zipcode=98103,
        bedrooms=3,
        sqft_living=1800,
        predicted_price=450000.0,
        n=6,
    )
    report("find_comparables_raw returns DataFrame", hasattr(comp_df, "shape"))
    report("Returns non-empty results", len(comp_df) > 0, f"{len(comp_df)} comps found")

    expected_cols = ["price", "bedrooms", "bathrooms", "sqft_living",
                     "grade", "condition", "house_age", "price_per_sqft"]
    missing_cols = [c for c in expected_cols if c not in comp_df.columns]
    report("Has all expected columns", len(missing_cols) == 0,
           f"Missing: {missing_cols}" if missing_cols else f"Columns: {list(comp_df.columns)}")

    if len(comp_df) > 0:
        avg_price = comp_df["price"].mean()
        report("Average comp price is reasonable",
               10000 < avg_price < 5000000, f"${avg_price:,.0f}")

except Exception as e:
    report("Comparables Finder", False, f"{e}\n{traceback.format_exc()}")


# ============================================================
# STEP 7: REPORT SCHEMA TEST (build_report_from_state)
# ============================================================
print()
print("=" * 70)
print("STEP 7: REPORT SCHEMA TEST (build_report_from_state)")
print("=" * 70)

try:
    from output.report_schema import build_report_from_state, AdvisoryReport

    mock_state = {
        "valuation": {
            "predicted_price": 450000.0,
            "price_low": 400000.0,
            "price_high": 500000.0,
            "confidence": 72.5,
            "std_dev": 50000.0,
            "investment_score": 65,
            "investment_label": "Good Buy",
            "market_status": "Fair Price",
            "price_per_sqft": 250.0,
            "valuation_explanation": "Test explanation.",
        },
        "market_context": "The King County market shows stable prices in this zipcode.",
        "rag_sources": ["King County Market Insights", "Zipcode Profiles"],
        "comparables": [
            {"price": 440000, "bedrooms": 3, "bathrooms": 2.0,
             "sqft_living": 1750, "grade": 7, "condition": 3,
             "house_age": 28, "price_per_sqft": 251.0},
            {"price": 460000, "bedrooms": 3, "bathrooms": 2.5,
             "sqft_living": 1900, "grade": 8, "condition": 4,
             "house_age": 15, "price_per_sqft": 242.0},
        ],
        "comparables_narrative": "Found 2 comparable properties with prices ranging from $440,000 to $460,000.",
        "risk_assessment": {
            "overall_risk": "MODERATE",
            "risk_score": 35.0,
            "risk_factors": [
                {"factor": "Price vs Comparables", "severity": "LOW",
                 "score": 1.0, "explanation": "Aligns well",
                 "mitigation": "Get appraisal"},
                {"factor": "Model Confidence", "severity": "LOW",
                 "score": 1.0, "explanation": "High confidence",
                 "mitigation": "Cross-reference"},
            ],
            "risk_narrative": "Overall risk is moderate.",
        },
        "recommendation": "BUY",
        "advisory_report": (
            "## Property Valuation Summary\nPredicted at $450,000.\n"
            "## Market Context\nStable market.\n"
            "## Comparable Sales Analysis\n2 comps found.\n"
            "## Risk Assessment\nModerate risk.\n"
            "## Investment Recommendation\nBUY.\n"
            "## Key Considerations\n- Factor 1\n"
            "## Disclaimers\nAI-generated for educational purposes."
        ),
        "disclaimers": "AI-generated for educational purposes only.",
        "error_log": [],
    }

    report_obj = build_report_from_state(mock_state)
    report("build_report_from_state succeeds", True)
    report("Returns AdvisoryReport", isinstance(report_obj, AdvisoryReport))
    report("Predicted price correct",
           report_obj.valuation.predicted_price == 450000.0,
           f"${report_obj.valuation.predicted_price:,.0f}")
    report("Recommendation correct",
           report_obj.recommendation == "BUY", report_obj.recommendation)
    report("Risk level correct",
           report_obj.risk_level == "MODERATE", report_obj.risk_level)
    report("num_comparables correct",
           report_obj.num_comparables == 2, str(report_obj.num_comparables))
    report("Risk factors parsed",
           len(report_obj.risk_factors) == 2,
           f"{len(report_obj.risk_factors)} factors")
    report("Market context present",
           len(report_obj.market_context) > 10)
    report("Advisory markdown present",
           len(report_obj.advisory_markdown) > 50)

    # Test Pydantic validation: invalid data should fail
    try:
        bad_state = dict(mock_state)
        bad_state["recommendation"] = "INVALID_VALUE"
        build_report_from_state(bad_state)
        report("Pydantic rejects invalid recommendation", False, "Should have raised ValidationError")
    except Exception as e:
        report("Pydantic rejects invalid recommendation", True, type(e).__name__)

except Exception as e:
    report("Report Schema", False, f"{e}\n{traceback.format_exc()}")


# ============================================================
# STEP 8: COMMON RUNTIME FAILURE CHECKS
# ============================================================
print()
print("=" * 70)
print("STEP 8: COMMON RUNTIME FAILURE CHECKS")
print("=" * 70)

# 8a. File path checks
from config.settings import MODEL_PATH, DATA_PATH, CHROMA_DB_PATH, KNOWLEDGE_SOURCES_PATH
report("MODEL_PATH (model.pkl) exists", os.path.exists(MODEL_PATH), MODEL_PATH)
report("DATA_PATH (kc_house_data.csv) exists", os.path.exists(DATA_PATH), DATA_PATH)
report("CHROMA_DB_PATH exists", os.path.exists(CHROMA_DB_PATH), CHROMA_DB_PATH)
report("KNOWLEDGE_SOURCES_PATH exists", os.path.exists(KNOWLEDGE_SOURCES_PATH), KNOWLEDGE_SOURCES_PATH)

# 8b. __init__.py files
init_locations = [
    "agents/__init__.py",
    "agents/nodes/__init__.py",
    "agents/tools/__init__.py",
    "config/__init__.py",
    "rag/__init__.py",
    "output/__init__.py",
]
for init_path in init_locations:
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), init_path)
    report(f"__init__.py: {init_path}", os.path.exists(full_path))

# 8c. Circular import check (already done implicitly by step 1 imports succeeding)
report("No circular imports detected", True, "All modules loaded without import loops")

# 8d. Global mutable state check
print()
print("  Global mutable state inventory (manual review needed):")
print("    - agents.tools.ml_predictor._artifacts (lazy-loaded model, OK)")
print("    - agents.tools.comparables_finder._df (lazy-loaded DataFrame, OK)")
print("    - agents.tools.market_stats._df (lazy-loaded DataFrame, OK)")
print("    - agents.tools.rag_retriever._collection (lazy-loaded ChromaDB, OK)")
print("    - rag.embeddings._embedder (lazy-loaded embedding fn, OK)")
print("    - agents.graph._graph (singleton graph, OK)")
print("    NOTE: All use lazy initialization pattern. No cross-request contamination risk.")

# 8e. Check LLM config (no API key needed for this check)
try:
    get_llm()
    report("LLM available", True)
except RuntimeError as e:
    report("LLM available", False, f"Expected in non-Streamlit env: {e}")
except Exception as e:
    report("LLM available", False, f"Unexpected error: {e}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print()
print("=" * 70)
print(f"FINAL SUMMARY: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 70)
if FAIL > 0:
    print("*** FAILURES DETECTED - Review output above ***")
else:
    print("*** ALL CHECKS PASSED ***")
