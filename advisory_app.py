"""
Milestone 2 — Agentic AI Real Estate Advisory System
Multi-agent LangGraph pipeline with RAG, conditional routing,
human-in-the-loop, and live streaming.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import pickle
import uuid
import html
from io import BytesIO
from config.guardrails import sanitize_plaintext

# --- Page Config ---
try:
    st.set_page_config(
        page_title="AI Real Estate Advisory",
        page_icon="https://img.icons8.com/fluency/48/real-estate.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except st.StreamlitAPIException:
    pass

# --- Custom CSS ---
st.markdown("""
<style>
/* ── Global overflow fix ── */
.stMarkdown, .stMarkdown div, .stMarkdown p,
.element-container, [data-testid="column"] {
    overflow-wrap: break-word;
    word-wrap: break-word;
    word-break: break-word;
}
[data-testid="column"] {
    overflow: hidden;
}

/* ── Metric cards ── */
.metric-card {background:white;border-radius:14px;padding:22px 20px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);text-align:center;
    border-top:4px solid #3b82f6;margin-bottom:10px;}
.metric-card-green {border-top-color:#10b981!important;}
.metric-card-orange {border-top-color:#f59e0b!important;}
.metric-card-red {border-top-color:#ef4444!important;}
.metric-card-purple {border-top-color:#8b5cf6!important;}
.metric-label {font-size:11px;font-weight:700;text-transform:uppercase;
    color:#6b7280;letter-spacing:1px;}
.metric-value {font-size:28px;font-weight:800;color:#1e293b;margin:8px 0 4px;}
.metric-sub {font-size:12px;color:#94a3b8;}

/* ── Badges ── */
.badge {display:inline-block;padding:5px 16px;border-radius:20px;font-size:14px;font-weight:700;}
.badge-buy {background:#d1fae5;color:#065f46;}
.badge-hold {background:#fef3c7;color:#92400e;}
.badge-caution {background:#ffedd5;color:#9a3412;}
.badge-avoid {background:#fee2e2;color:#991b1b;}

/* ── Section titles ── */
.section-title {font-size:18px;font-weight:700;color:#1e293b;
    padding:10px 0 6px;border-bottom:2px solid #e2e8f0;margin-bottom:14px;}

/* ── Agent queue ── */
.agent-node {display:inline-block;padding:6px 14px;border-radius:8px;
    font-size:12px;font-weight:600;margin:2px 4px;}
.agent-done {background:#d1fae5;color:#065f46;}
.agent-running {background:#dbeafe;color:#1e40af;animation:pulse 1.5s infinite;}
.agent-pending {background:#f3f4f6;color:#9ca3af;}
.agent-error {background:#fee2e2;color:#991b1b;}
.risk-bar {height:24px;border-radius:12px;margin:4px 0;}

/* ── Buttons ── */
@keyframes pulse {0%,100%{opacity:1}50%{opacity:0.6}}
.stButton>button {background:linear-gradient(135deg,#3b82f6,#1d4ed8);color:white;
    border-radius:10px;font-weight:600;padding:12px 24px;border:none;
    font-size:15px;width:100%;}

/* ── Hide default Streamlit chrome ── */
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>""", unsafe_allow_html=True)


# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "advisory_result" not in st.session_state:
    st.session_state.advisory_result = None
if "agent_log" not in st.session_state:
    st.session_state.agent_log = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0
if "chat_thread_id" not in st.session_state:
    st.session_state.chat_thread_id = f"chat-{uuid.uuid4()}"

_form_defaults = {
    "client_mode_input": "BUYER",
    "objective_input": "Primary residence decision",
    "budget_input": 750000.0,
    "asking_price_input": 650000.0,
    "risk_tolerance_input": "BALANCED",
    "financing_input": "MORTGAGE",
    "hold_years_input": 7,
    "rent_input": 3200.0,
    "reno_budget_input": 25000.0,
    "must_haves_input": ["walkability", "upside potential"],
    "listing_notes_input": "",
    "bedrooms_input": 3,
    "bathrooms_input": 2.0,
    "floors_input": 1.0,
    "sqft_living_input": 1800,
    "sqft_lot_input": 7500,
    "sqft_above_input": 1800,
    "sqft_living15_input": 1800,
    "sqft_lot15_input": 7500,
    "grade_input": 7,
    "condition_input": 3,
    "zipcode_input": 98103,
    "lat_input": 47.50,
    "lng_input": -122.20,
    "yr_built_input": 1990,
    "renovated_input": False,
    "waterfront_input": False,
    "view_input": 0,
}
for _key, _value in _form_defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value


# --- Load Milestone 1 assets ---
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    df = pd.read_csv("kc_house_data.csv")
    df = df[df["bedrooms"] <= 10]
    df["house_age"] = datetime.datetime.now().year - df["yr_built"]
    df["renovated"] = df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)
    df["amenity_score"] = df["waterfront"] + df["view"] + df["condition"] + df["grade"]
    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    return df

try:
    artifacts = load_model()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    zipcode_mean = artifacts["zipcode_mean"]
    df_full = load_dataset()
    MODEL_OK = True
except Exception:
    MODEL_OK = False

# --- Sidebar Mode Switch ---
st.sidebar.image("https://img.icons8.com/fluency/96/real-estate.png", width=60)
st.sidebar.title("AI Real Estate Advisor")

# Check if user clicked the chatbot button on home page
_default_mode = 1 if st.session_state.get("_chatbot_switch") else 0

app_mode = st.sidebar.radio(
    "Choose Mode",
    ["Advisory Report (King County)", "AI Property Chatbot (Any Location)"],
    index=_default_mode,
    help="Advisory Report uses ML model for King County. Chatbot uses AI for any US location."
)

# Clear the switch flag after it's been consumed
if st.session_state.get("_chatbot_switch"):
    st.session_state["_chatbot_switch"] = False

IS_CHATBOT_MODE = "Chatbot" in app_mode

# --- Sidebar Inputs (only shown in Advisory mode) ---
if not IS_CHATBOT_MODE:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Client Profile")
    client_mode = st.sidebar.selectbox(
        "Client Mode",
        ["BUYER", "SELLER", "INVESTOR"],
        key="client_mode_input",
        format_func=lambda v: {"BUYER": "Buyer", "SELLER": "Seller", "INVESTOR": "Investor"}[v],
    )
    objective = st.sidebar.selectbox(
        "Objective",
        [
            "Primary residence decision",
            "Acquire for long-term hold",
            "Renovate and resell",
            "Price to sell efficiently",
            "Portfolio expansion",
        ],
        key="objective_input",
    )
    budget = st.sidebar.number_input("Budget", min_value=0.0, step=10000.0, key="budget_input")
    asking_price = st.sidebar.number_input("Asking / Target Price", min_value=0.0, step=5000.0, key="asking_price_input")
    risk_tolerance = st.sidebar.select_slider(
        "Risk Tolerance",
        options=["CONSERVATIVE", "BALANCED", "AGGRESSIVE"],
        key="risk_tolerance_input",
        format_func=lambda v: v.title(),
    )
    financing = st.sidebar.selectbox(
        "Financing",
        ["CASH", "MORTGAGE", "MIXED"],
        key="financing_input",
        format_func=lambda v: v.title(),
    )
    target_hold_years = st.sidebar.slider("Target Hold Period (Years)", 0, 15, key="hold_years_input")
    monthly_rent_estimate = st.sidebar.number_input("Monthly Rent Estimate", min_value=0.0, step=100.0, key="rent_input")
    renovation_budget = st.sidebar.number_input("Renovation Budget", min_value=0.0, step=5000.0, key="reno_budget_input")
    must_haves = st.sidebar.multiselect(
        "Decision Priorities",
        [
            "walkability",
            "school district",
            "upside potential",
            "cash flow",
            "low maintenance",
            "luxury finish",
            "faster resale",
        ],
        key="must_haves_input",
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Listing Notes")
    listing_notes = st.sidebar.text_area(
        "Paste listing text for auto-fill",
        key="listing_notes_input",
        height=120,
        placeholder="Example: 3 bed, 2 bath, 1800 sqft in 98103, built 1994, asking $725,000...",
    )
    if st.sidebar.button("Parse Listing Notes", width="stretch"):
        parsed = parse_listing_text(listing_notes)
        if parsed:
            for field, value in parsed.items():
                st.session_state[field] = value
            st.sidebar.success("Listing details parsed into the form.")
        else:
            st.sidebar.info("No property fields were detected in the notes.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Property Details")
    bedrooms = st.sidebar.slider("Bedrooms", 1, 10, key="bedrooms_input")
    bathrooms = st.sidebar.slider("Bathrooms", 0.0, 8.0, key="bathrooms_input")
    floors = st.sidebar.slider("Floors", 1.0, 3.5, step=0.5, key="floors_input")
    sqft_living = st.sidebar.number_input("Living Area (sqft)", min_value=300, max_value=13000, key="sqft_living_input")
    sqft_lot = st.sidebar.number_input("Lot Size (sqft)", min_value=500, max_value=1600000, key="sqft_lot_input")
    sqft_above = st.sidebar.number_input("Above Ground (sqft)", min_value=300, max_value=13000, key="sqft_above_input")
    sqft_basement = max(0, int(sqft_living) - int(sqft_above))
    sqft_living15 = st.sidebar.number_input("Neighbor Avg Living (sqft)", min_value=300, max_value=13000, key="sqft_living15_input")
    sqft_lot15 = st.sidebar.number_input("Neighbor Avg Lot (sqft)", min_value=500, max_value=1600000, key="sqft_lot15_input")
    st.sidebar.markdown("---")
    grade = st.sidebar.slider("Grade (1-13)", 1, 13, key="grade_input", help="7=Average, 10=Very Good, 13=Mansion")
    condition = st.sidebar.slider("Condition (1-5)", 1, 5, key="condition_input", help="1=Poor, 3=Average, 5=Excellent")
    st.sidebar.markdown("---")
    zipcode = st.sidebar.number_input("Zipcode", min_value=98001, max_value=98199, key="zipcode_input")
    lat = st.sidebar.number_input("Latitude", min_value=47.10, max_value=47.80, format="%.4f", key="lat_input")
    lng = st.sidebar.number_input("Longitude", min_value=-122.60, max_value=-121.30, format="%.4f", key="lng_input")
    st.sidebar.markdown("---")
    yr_built = st.sidebar.number_input("Year Built", min_value=1900, max_value=2024, key="yr_built_input")
    renovated = st.sidebar.checkbox("Has Been Renovated", key="renovated_input")
    waterfront = st.sidebar.checkbox("Waterfront Property", key="waterfront_input")
    view = st.sidebar.slider("View Quality (0-4)", 0, 4, key="view_input")
else:
    client_mode = "BUYER"
    objective = "Primary residence decision"
    budget, asking_price = 750000.0, 650000.0
    risk_tolerance, financing = "BALANCED", "MORTGAGE"
    target_hold_years, monthly_rent_estimate, renovation_budget = 7, 3200.0, 25000.0
    must_haves = ["walkability", "upside potential"]
    listing_notes = ""
    bedrooms, bathrooms, floors, sqft_living, sqft_lot = 3, 2.0, 1.0, 1800, 7500
    sqft_above, sqft_basement, sqft_living15, sqft_lot15 = 1800, 0, 1800, 7500
    grade, condition, zipcode, lat, lng = 7, 3, 98103, 47.5, -122.2
    yr_built, renovated, waterfront, view = 1990, False, False, 0

# Build property input dict
property_input = {
    "bedrooms": bedrooms,
    "bathrooms": float(bathrooms),
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": float(floors),
    "waterfront": 1 if waterfront else 0,
    "view": view,
    "condition": condition,
    "grade": grade,
    "sqft_above": sqft_above,
    "sqft_basement": sqft_basement,
    "lat": lat,
    "long": lng,
    "sqft_living15": sqft_living15,
    "sqft_lot15": sqft_lot15,
    "zipcode": zipcode,
    "yr_built": yr_built,
    "renovated": 1 if renovated else 0,
}

consultation_context = {
    "client_mode": client_mode,
    "objective": objective,
    "budget": float(budget),
    "asking_price": float(asking_price),
    "target_hold_years": int(target_hold_years),
    "risk_tolerance": risk_tolerance,
    "financing": financing,
    "monthly_rent_estimate": float(monthly_rent_estimate),
    "renovation_budget": float(renovation_budget),
    "must_haves": must_haves,
    "raw_notes": sanitize_plaintext(listing_notes, max_chars=1600),
}


# --- Header ---
if IS_CHATBOT_MODE:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px;'>
        <h1 style='margin:0;color:#1e293b;'>AI Property Price Chatbot</h1>
        <p style='color:#64748b;font-size:16px;margin-top:4px;'>
            Ask about property prices in any US location | Powered by AI + ML
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px;'>
        <h1 style='margin:0;color:#1e293b;'>AI Property Decision Copilot</h1>
        <p style='color:#64748b;font-size:16px;margin-top:4px;'>
            Light-theme multi-agent property intelligence for pricing, negotiation, and decision support
        </p>
    </div>
    """, unsafe_allow_html=True)


# --- Run Advisory Button (only in advisory mode) ---
run_advisory = st.sidebar.button("Run Property Decision Copilot", type="primary") if not IS_CHATBOT_MODE else False

# ─── Floating "Chat" button (bottom-right, hidden in chatbot mode) ───
if not IS_CHATBOT_MODE:
    st.markdown("""
    <style>
    .floating-chat-btn {
        position: fixed; bottom: 30px; right: 30px; z-index: 99999;
        background: linear-gradient(135deg, #0f766e, #0891b2); color: white;
        border: none; border-radius: 50px; padding: 16px 28px;
        font-size: 16px; font-weight: 700; cursor: pointer;
        box-shadow: 0 6px 25px rgba(8,145,178,0.35);
        transition: all 0.3s ease; display: flex; align-items: center; gap: 8px;
        text-decoration: none; font-family: -apple-system, sans-serif;
    }
    .floating-chat-btn:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 35px rgba(8,145,178,0.45);
    }
    .floating-chat-pulse {
        position: absolute; top: -3px; right: -3px;
        width: 14px; height: 14px; background: #10b981;
        border-radius: 50%; border: 2px solid white;
        animation: fpulse 2s infinite;
    }
    @keyframes fpulse { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.3);opacity:0.7} }
    </style>
    <button class="floating-chat-btn" onclick="
        const doc = window.parent.document;
        const labels = doc.querySelectorAll('[data-testid=stSidebar] label');
        for (const l of labels) {
            if (l.textContent.includes('Chatbot')) { l.click(); break; }
        }
    ">
        Ask a Question
        <span class="floating-chat-pulse"></span>
    </button>
    """, unsafe_allow_html=True)

# --- Chart functions (reused from Milestone 1) ---
def chart_feature_importance():
    fd = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    fd = fd.sort_values("Importance", ascending=True).tail(12)
    fig = px.bar(fd, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues",
                 title="Top 12 Price Drivers (Explainable AI)")
    fig.update_layout(height=420, coloraxis_showscale=False, plot_bgcolor="white",
                      paper_bgcolor="white", margin=dict(l=10, r=10, t=50, b=30))
    return fig

def chart_price_dist(price):
    zdf = df_full[df_full["zipcode"] == zipcode]
    if len(zdf) < 5:
        zdf = df_full
        title = "Price Distribution — All King County"
    else:
        title = f"Price Distribution — Zipcode {zipcode}"
    fig = px.histogram(zdf, x="price", nbins=40, title=title,
                       color_discrete_sequence=["#93c5fd"], opacity=0.8)
    fig.add_vline(x=price, line_dash="dash", line_color="#1d4ed8", line_width=3,
                  annotation_text=f"Predicted: ${price:,.0f}")
    fig.add_vline(x=zdf["price"].mean(), line_dash="dot", line_color="#ef4444",
                  annotation_text=f"Avg: ${zdf['price'].mean():,.0f}")
    fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=30), showlegend=False)
    return fig

def chart_confidence_gauge(conf):
    color = "#10b981" if conf >= 70 else "#f59e0b" if conf >= 50 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=conf,
        number={"suffix": "%", "font": {"size": 32}},
        title={"text": "Model Confidence"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [
                   {"range": [0, 40], "color": "#fee2e2"},
                   {"range": [40, 70], "color": "#fef3c7"},
                   {"range": [70, 100], "color": "#d1fae5"}]}))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=20), paper_bgcolor="white")
    return fig

def chart_investment_gauge(score):
    color = "#10b981" if score >= 75 else "#22c55e" if score >= 60 else "#f59e0b" if score >= 45 else "#f97316" if score >= 30 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"font": {"size": 32}},
        title={"text": "Investment Score"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [
                   {"range": [0, 30], "color": "#fee2e2"},
                   {"range": [30, 45], "color": "#ffedd5"},
                   {"range": [45, 60], "color": "#fef3c7"},
                   {"range": [60, 75], "color": "#dcfce7"},
                   {"range": [75, 100], "color": "#d1fae5"}]}))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=60, b=20), paper_bgcolor="white")
    return fig

def chart_risk_breakdown(risk_factors):
    if not risk_factors:
        return None
    df_risk = pd.DataFrame(risk_factors)
    color_map = {"LOW": "#10b981", "MODERATE": "#f59e0b", "HIGH": "#f97316", "CRITICAL": "#ef4444"}
    df_risk["color"] = df_risk["severity"].map(color_map)
    fig = px.bar(df_risk, x="score", y="factor", orientation="h",
                 color="severity", color_discrete_map=color_map,
                 title="Risk Factor Breakdown (8 Dimensions)")
    fig.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=30), xaxis_title="Risk Score",
                      yaxis_title="")
    return fig


def format_currency(value):
    return f"${value:,.0f}"


def escape_html(value) -> str:
    return html.escape(str(value or ""), quote=True)


def parse_listing_text(raw_text: str) -> dict:
    """Best-effort parser for pasted listing notes."""
    import re
    from config.guardrails import sanitize_plaintext

    raw_text = sanitize_plaintext(raw_text, max_chars=1600)
    if not raw_text.strip():
        return {}

    patterns = {
        "bedrooms_input": r"(\d+)\s*(?:bed|br|bedroom)",
        "bathrooms_input": r"(\d+(?:\.\d+)?)\s*(?:bath|ba|bathroom)",
        "sqft_living_input": r"(\d{3,5})\s*(?:sq\s*ft|sqft|square feet)",
        "sqft_lot_input": r"lot\s*(?:size)?[:\s]+(\d{3,7})",
        "zipcode_input": r"\b(98\d{3})\b",
        "grade_input": r"grade[:\s]+(\d{1,2})",
        "condition_input": r"condition[:\s]+(\d)\b",
        "yr_built_input": r"(?:built|year built)[:\s]+(19\d{2}|20\d{2})",
        "asking_price_input": r"\$([\d,]+)",
    }

    parsed = {}
    lowered = raw_text.lower()
    for key, pattern in patterns.items():
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if match:
            raw_value = match.group(1).replace(",", "")
            parsed[key] = float(raw_value) if "." in raw_value else int(raw_value)

    if "renovated" in lowered:
        parsed["renovated_input"] = True
    if "waterfront" in lowered:
        parsed["waterfront_input"] = True
    view_match = re.search(r"view[:\s]+(\d)", lowered)
    if view_match:
        parsed["view_input"] = int(view_match.group(1))

    sqft_living = int(parsed.get("sqft_living_input", st.session_state.get("sqft_living_input", 1800)))
    if "sqft_above_input" not in parsed:
        parsed["sqft_above_input"] = sqft_living
    if "sqft_living15_input" not in parsed:
        parsed["sqft_living15_input"] = sqft_living
    if "sqft_lot_input" in parsed and "sqft_lot15_input" not in parsed:
        parsed["sqft_lot15_input"] = int(parsed["sqft_lot_input"])

    return parsed


def render_metric_card(title: str, value: str, subtitle: str, tone: str = "neutral"):
    tone_map = {
        "neutral": "#2563eb",
        "positive": "#16a34a",
        "warning": "#d97706",
        "danger": "#dc2626",
        "accent": "#0f766e",
    }
    border = tone_map.get(tone, tone_map["neutral"])
    title = escape_html(title)
    value = escape_html(value)
    subtitle = escape_html(subtitle)
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #dbe4f0;border-top:4px solid {border};
                    border-radius:18px;padding:18px 18px 16px;box-shadow:0 12px 26px rgba(15,23,42,0.06);
                    overflow:hidden;word-break:break-word;">
            <div style="font-size:11px;font-weight:700;letter-spacing:0.12em;color:#64748b;text-transform:uppercase;">{title}</div>
            <div style="font-size:24px;font-weight:800;color:#0f172a;margin-top:8px;line-height:1.2;">{value}</div>
            <div style="font-size:12px;color:#475569;margin-top:6px;line-height:1.4;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tag_strip(tags: list[str], tone: str = "#e2e8f0"):
    if not tags:
        return
    chips = "".join(
        f'<span style="display:inline-block;padding:6px 10px;margin:0 8px 8px 0;'
        f'border-radius:999px;background:{tone};color:#0f172a;font-size:12px;font-weight:600;">{escape_html(tag)}</span>'
        for tag in tags
    )
    st.markdown(chips, unsafe_allow_html=True)


def build_queue_status(agent_log: list[dict]):
    ordered = [
        "Intake",
        "Valuation",
        "Market Analyst",
        "Comparables",
        "Risk Assessor",
        "Human Review",
        "Neighborhood Analyst",
        "Negotiation Strategist",
        "Decision Analyst",
        "Synthesizer",
    ]
    optional_agents = {"Market Analyst", "Human Review"}
    running = next((entry["agent"] for entry in reversed(agent_log) if entry.get("phase") == "running"), None)
    completed = {
        entry["agent"]
        for entry in agent_log
        if entry.get("phase") != "running" and "error" not in entry.get("phase", "")
    }
    failed = {entry["agent"] for entry in agent_log if "error" in entry.get("phase", "")}

    current = running
    has_terminal_progress = any(entry["agent"] == "Synthesizer" for entry in agent_log)
    if current is None:
        for name in ordered:
            if name not in completed and name not in failed:
                current = name
                break

    queue = []
    for name in ordered:
        if name in failed:
            status = "Failed"
        elif name in completed:
            status = "Completed"
        elif name in optional_agents and any(
            ordered.index(entry["agent"]) > ordered.index(name)
            for entry in agent_log
            if entry.get("agent") in ordered
        ):
            status = "Skipped"
        elif has_terminal_progress:
            status = "Skipped"
        elif name == current and agent_log:
            status = "Running"
        else:
            status = "Queued"
        queue.append({"agent": name, "status": status})
    return queue


def render_queue_panel(agent_log: list[dict]):
    queue = build_queue_status(agent_log)
    status_colors = {
        "Queued": ("#f8fafc", "#64748b", "#cbd5e1"),
        "Running": ("#eff6ff", "#1d4ed8", "#93c5fd"),
        "Completed": ("#f0fdf4", "#15803d", "#86efac"),
        "Skipped": ("#f8fafc", "#64748b", "#e2e8f0"),
        "Failed": ("#fef2f2", "#dc2626", "#fca5a5"),
    }
    rows = []
    for item in queue:
        bg, fg, border = status_colors[item["status"]]
        agent_name = escape_html(item["agent"])
        status = escape_html(item["status"])
        rows.append(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:10px 12px;border:1px solid {border};border-radius:14px;background:{bg};margin-bottom:10px;">'
            f'<span style="font-weight:700;color:#0f172a;">{agent_name}</span>'
            f'<span style="font-size:12px;font-weight:700;color:{fg};">{status}</span>'
            f'</div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def build_scenario_table(final_state: dict, report, consultation: dict, market_shift_pct: float, offer_delta_pct: float, rate_shift_pct: float):
    if not report:
        return pd.DataFrame()

    base_price = report.valuation.predicted_price
    asking_price = consultation.get("asking_price", base_price) or base_price
    target_price = final_state.get("negotiation_strategy", {}).get("target_price", base_price)
    rent = consultation.get("monthly_rent_estimate", 0)
    renovation = consultation.get("renovation_budget", 0)

    scenarios = []
    for label, local_shift, extra_risk in [
        ("Bull", market_shift_pct + 4, -4),
        ("Base", market_shift_pct, 0),
        ("Risk", market_shift_pct - 6, 8 + max(rate_shift_pct, 0) * 2),
    ]:
        scenario_value = base_price * (1 + local_shift / 100) + renovation * 0.45
        effective_offer = target_price * (1 + offer_delta_pct / 100)
        spread = scenario_value - effective_offer
        annual_rent = rent * 12
        gross_yield = (annual_rent / max(effective_offer + renovation, 1)) * 100 if annual_rent else 0
        scenarios.append(
            {
                "Scenario": label,
                "Scenario Value": scenario_value,
                "Effective Offer": effective_offer,
                "Value Spread": spread,
                "Gross Yield %": gross_yield,
                "Risk Bias": report.risk_score + extra_risk,
            }
        )

    df = pd.DataFrame(scenarios)
    for column in ["Scenario Value", "Effective Offer", "Value Spread"]:
        df[column] = df[column].map(format_currency)
    df["Gross Yield %"] = df["Gross Yield %"].map(lambda x: f"{x:.1f}%")
    df["Risk Bias"] = df["Risk Bias"].map(lambda x: f"{x:.1f}")
    return df


def build_chat_snapshot(response_text: str) -> str:
    """Extract a simple professional summary block from a chatbot answer."""
    import re

    price_matches = re.findall(r"\$[\d,]+", response_text)
    confidence_match = re.search(r"(\d+(?:\.\d+)?)%\s*(?:confidence|confidence level)", response_text, flags=re.IGNORECASE)
    risk_word = None
    for word in ["STRONG BUY", "BUY", "HOLD", "CAUTION", "AVOID", "OVERPRICED", "UNDERPRICED", "FAIR"]:
        if word in response_text.upper():
            risk_word = word.title()
            break

    lines = ["**Consultation Snapshot**"]
    if price_matches:
        lines.append(f"- Price signal: {price_matches[0]}")
        if len(price_matches) >= 3:
            lines.append(f"- Range: {price_matches[1]} to {price_matches[2]}")
    if confidence_match:
        lines.append(f"- Confidence: {confidence_match.group(1)}%")
    if risk_word:
        lines.append(f"- Positioning: {risk_word}")
    if len(lines) == 1:
        lines.append("- The response contains guidance, but no structured pricing snapshot was detected.")
    return "\n".join(lines)


# --- Run the Agent Pipeline ---
def run_advisory_pipeline(prop_input, consultation_input):
    """Execute the full LangGraph advisory pipeline with streaming."""
    from agents.graph import get_advisory_graph
    from output.report_schema import build_report_from_state

    graph = get_advisory_graph()
    thread_id = f"{st.session_state.session_id}-{uuid.uuid4()}"
    st.session_state.current_advisory_thread_id = thread_id

    initial_state = {
        "property_input": prop_input,
        "consultation_context": consultation_input,
        "user_query": f"Analyze this property in zipcode {prop_input['zipcode']}",
        "messages": [],
        "valuation": None,
        "market_context": None,
        "rag_sources": None,
        "comparables": None,
        "comparables_narrative": None,
        "risk_assessment": None,
        "neighborhood_analysis": None,
        "negotiation_strategy": None,
        "decision_lenses": None,
        "decision_summary": None,
        "advisory_report": None,
        "recommendation": None,
        "disclaimers": None,
        "needs_human_review": False,
        "iteration_count": 0,
        "error_log": [],
        "current_phase": "starting",
    }

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 16}

    agent_names = {
        "intake_agent": "Intake",
        "valuation_agent": "Valuation",
        "market_analyst": "Market Analyst",
        "comparables_agent": "Comparables",
        "risk_assessor": "Risk Assessor",
        "human_review": "Human Review",
        "neighborhood_analyst": "Neighborhood Analyst",
        "negotiation_agent": "Negotiation Strategist",
        "decision_agent": "Decision Analyst",
        "advisory_synthesizer": "Synthesizer",
    }

    agent_log = []
    final_state = initial_state.copy()

    # Streaming progress
    progress_container = st.container()
    with progress_container:
        st.markdown('<div class="section-title">Live Agent Queue</div>', unsafe_allow_html=True)
        status_placeholder = st.empty()
        detail_container = st.container()
    with status_placeholder.container():
        render_queue_panel([])

    completed_agents = []

    try:
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__start__":
                    continue

                display_name = agent_names.get(node_name, node_name)
                completed_agents.append(display_name)

                # Update queue panel
                status_placeholder.empty()
                with status_placeholder.container():
                    render_queue_panel(agent_log + [{
                        "agent": display_name,
                        "phase": "running",
                        "message": node_output.get("messages", [])[-1].content if node_output.get("messages") else "",
                        "output_keys": list(node_output.keys()),
                    }])

                # Log agent output
                phase = node_output.get("current_phase", "")
                messages = node_output.get("messages", [])
                msg_text = messages[-1].content if messages else ""

                agent_log.append({
                    "agent": display_name,
                    "phase": phase,
                    "message": msg_text,
                    "output_keys": list(node_output.keys()),
                })

                with detail_container:
                    with st.status(f"{display_name}", state="complete"):
                        st.write(msg_text)

                # Merge into final state
                for k, v in node_output.items():
                    if v is not None:
                        final_state[k] = v

                # Handle human review pause
                if node_name == "human_review" and final_state.get("needs_human_review"):
                    with detail_container:
                        st.warning("High risk detected! The system has flagged this property for review.")
                        st.info("Continuing analysis with risk acknowledgment...")

    except Exception as e:
        error_type = type(e).__name__
        # Handle NodeInterrupt (human-in-the-loop pause)
        if "NodeInterrupt" in error_type or "GraphInterrupt" in error_type:
            with detail_container:
                st.warning(f"**Human Review Triggered:** {e}")
                st.info("Resuming analysis with risk acknowledgment...")

            # Resume the graph from the interrupt
            try:
                from langgraph.types import Command
                for event in graph.stream(Command(resume=True), config, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        if node_name == "__start__":
                            continue
                        display_name = agent_names.get(node_name, node_name)
                        completed_agents.append(display_name)
                        status_placeholder.empty()
                        with status_placeholder.container():
                            render_queue_panel(agent_log)
                        phase = node_output.get("current_phase", "")
                        messages = node_output.get("messages", [])
                        msg_text = messages[-1].content if messages else ""
                        agent_log.append({
                            "agent": display_name, "phase": phase,
                            "message": msg_text, "output_keys": list(node_output.keys()),
                        })
                        with detail_container:
                            with st.status(f"{display_name}", state="complete"):
                                st.write(msg_text)
                        for k, v in node_output.items():
                            if v is not None:
                                final_state[k] = v
            except Exception as resume_err:
                st.error(f"Resume error: {resume_err}")
                agent_log.append({"agent": "System", "phase": "resume_error", "message": str(resume_err), "output_keys": []})
        else:
            st.error(f"Pipeline error: {e}")
            agent_log.append({"agent": "System", "phase": "error", "message": str(e), "output_keys": []})

    # Build validated report
    try:
        report = build_report_from_state(final_state)
    except Exception as e:
        report = None
        st.error(f"Report validation error: {e}")

    return final_state, report, agent_log


# --- Main Content ---
if not MODEL_OK:
    st.error("Model not found. Ensure model.pkl and kc_house_data.csv are in the working directory.")
    st.stop()


# ═══════════════════════════════════════════════════════════
# CHATBOT MODE
# ═══════════════════════════════════════════════════════════
if IS_CHATBOT_MODE:
    from config.guardrails import run_input_guardrails
    try:
        from agents.nodes.chatbot_agent import chat_with_advisor
    except Exception as chatbot_import_error:
        st.error("AI Property Chatbot is temporarily unavailable.")
        st.info(
            "The chatbot runtime dependencies are missing or not fully initialized. "
            "Install the AI packages from `requirements.txt` and restart the app."
        )
        st.caption(f"Runtime details: {chatbot_import_error}")
        st.stop()

    # Sidebar chatbot controls
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.session_state.chat_count = 0
        st.session_state.chat_thread_id = f"chat-{uuid.uuid4()}"
        st.rerun()


    # Welcome message
    if not st.session_state.chat_messages:
        welcome = (
            "Hi, I'm your AI property advisor. Think of me like a real estate agent who can talk through "
            "pricing, market fit, negotiation angles, and risk in plain English.\n\n"
            "**For King County, WA** (zipcodes 98001-98199), I use a trained ML model "
            "(Random Forest, R²=0.88) for precise predictions.\n\n"
            "**For all other locations**, I use AI reasoning with market data from 50+ US metro areas.\n\n"
            "**Try asking me like you would ask an agent:**\n"
            "- *What's a 3BR/2BA 1800sqft house worth in zipcode 98103?*\n"
            "- *How much would a 4BR house cost in Austin, TX?*\n"
            "- *Compare prices between Denver and Phoenix for a 2000sqft home*\n"
            "- *If I were buying this, what would you watch out for?*\n\n"
            "I only answer real estate questions, and I will keep the advice practical rather than generic."
        )
        st.session_state.chat_messages.append({"role": "assistant", "content": welcome})

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask like you're talking to a property advisor..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run input guardrails BEFORE adding to history (prevents injection persistence)
        st.session_state.chat_count += 1  # count ALL queries toward rate limit
        is_allowed, guard_msg = run_input_guardrails(user_input, st.session_state.chat_count)

        if not is_allowed:
            # Guardrail blocked — store the user input but mark it for filtering
            st.session_state.chat_messages.append({"role": "user", "content": user_input, "_blocked": True})
            st.session_state.chat_messages.append({"role": "assistant", "content": guard_msg})
            with st.chat_message("assistant"):
                st.markdown(guard_msg)
        else:
            # Safe query — add to history and process
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = chat_with_advisor(
                        user_input,
                        st.session_state.chat_messages[:-1],
                        thread_id=st.session_state.chat_thread_id,
                    )
                st.markdown(response)
                st.info(build_chat_snapshot(response))
            st.session_state.chat_messages.append({"role": "assistant", "content": response})


# ═══════════════════════════════════════════════════════════
# ADVISORY REPORT MODE (existing)
# ═══════════════════════════════════════════════════════════
else:
    tab1, tab2, tab3, tab4 = st.tabs(["Decision Copilot", "Scenario Lab", "Model Insights", "Agent Queue"])

    if run_advisory:
        with tab1:
            from config.guardrails import validate_advisory_inputs

            is_valid_input, input_errors, input_warnings = validate_advisory_inputs(
                property_input,
                consultation_context,
            )
            if not is_valid_input:
                st.error("The advisory request was blocked by input guardrails.")
                for error in input_errors:
                    st.warning(error)
            else:
                for warning in input_warnings:
                    st.warning(warning)
                with st.spinner("Running the property decision copilot..."):
                    final_state, report, agent_log = run_advisory_pipeline(property_input, consultation_context)
                    st.session_state.advisory_result = (final_state, report)
                    st.session_state.agent_log = agent_log

    if st.session_state.advisory_result:
        final_state, report = st.session_state.advisory_result

        with tab1:
            if report:
                val = report.valuation
                consult = report.consultation
                headline_html = escape_html(report.decision_summary.headline)
                executive_summary_html = escape_html(report.decision_summary.executive_summary)
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
                                border:1px solid #d9e5f2;border-radius:24px;padding:24px 26px;
                                box-shadow:0 18px 40px rgba(15,23,42,0.08);margin-bottom:18px;">
                        <div style="font-size:12px;font-weight:800;color:#2563eb;letter-spacing:0.14em;text-transform:uppercase;">
                            AI Property Decision Copilot
                        </div>
                        <div style="font-size:34px;font-weight:800;color:#0f172a;margin-top:10px;">
                            {headline_html}
                        </div>
                        <div style="font-size:15px;color:#475569;margin-top:10px;max-width:900px;">
                            {executive_summary_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                render_tag_strip([
                    consult.client_mode.title(),
                    consult.objective,
                    f"Risk: {consult.risk_tolerance.title()}",
                    f"Financing: {consult.financing.title()}",
                    f"Hold: {consult.target_hold_years} yrs",
                ], tone="#e0f2fe")
                render_tag_strip(consult.must_haves, tone="#ecfccb")

                k1, k2, k3, k4, k5 = st.columns(5)
                with k1:
                    render_metric_card("Predicted Price", format_currency(val.predicted_price), f"Range {format_currency(val.price_low)} - {format_currency(val.price_high)}", "neutral")
                with k2:
                    render_metric_card("Recommendation", report.recommendation.replace("_", " "), f"Risk {report.risk_level}", "accent")
                with k3:
                    render_metric_card("Neighborhood", f"{report.neighborhood.overall_score}/100", report.neighborhood.market_heat.title(), "positive" if report.neighborhood.overall_score >= 60 else "warning")
                with k4:
                    render_metric_card("Negotiation Target", format_currency(report.negotiation.target_price), f"Walk-away {format_currency(report.negotiation.walk_away_price)}", "warning")
                with k5:
                    render_metric_card("Investment Score", f"{val.investment_score}/100", f"Confidence {val.confidence:.1f}%", "positive" if val.investment_score >= 60 else "warning")

                st.markdown("---")
                left_col, right_col = st.columns([1.6, 1])
                with left_col:
                    st.markdown('<div class="section-title">Executive Advisory Memo</div>', unsafe_allow_html=True)
                    import re as _re_memo
                    _clean_advisory = _re_memo.sub(r'<[^>]+>', '', report.advisory_markdown or "")
                    st.markdown(_clean_advisory)
                with right_col:
                    st.markdown('<div class="section-title">Client Brief</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        - **Client mode:** {consult.client_mode.title()}
                        - **Budget:** {format_currency(consult.budget)}
                        - **Asking price:** {format_currency(consult.asking_price)}
                        - **Objective:** {consult.objective}
                        - **Financing:** {consult.financing.title()}
                        - **Monthly rent assumption:** {format_currency(consult.monthly_rent_estimate)}
                        - **Renovation budget:** {format_currency(consult.renovation_budget)}
                        """
                    )
                    st.markdown('<div class="section-title">Negotiation Playbook</div>', unsafe_allow_html=True)
                    st.markdown(report.negotiation.strategy_summary)
                    st.markdown("**Leverage Points**")
                    for item in report.negotiation.leverage_points:
                        st.markdown(f"- {item}")
                    st.markdown("**Caution Points**")
                    for item in report.negotiation.caution_points:
                        st.markdown(f"- {item}")

                st.markdown("---")
                score_col, lens_col = st.columns([1, 1.1])
                with score_col:
                    st.markdown('<div class="section-title">Neighborhood Scorecard</div>', unsafe_allow_html=True)
                    score_df = pd.DataFrame(
                        {
                            "Metric": ["Livability", "Liquidity", "Upside", "Rental Demand", "Pricing Power"],
                            "Score": [
                                report.neighborhood.livability_score,
                                report.neighborhood.liquidity_score,
                                report.neighborhood.upside_score,
                                report.neighborhood.rental_demand_score,
                                report.neighborhood.pricing_power_score,
                            ],
                        }
                    )
                    fig_scores = px.bar(
                        score_df,
                        x="Score",
                        y="Metric",
                        orientation="h",
                        range_x=[0, 100],
                        color="Score",
                        color_continuous_scale=["#dbeafe", "#60a5fa", "#1d4ed8"],
                        title="Neighborhood Intelligence Breakdown",
                    )
                    fig_scores.update_layout(height=320, coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig_scores, width="stretch")
                    import re as _re
                    _clean_narrative = _re.sub(r'<[^>]+>', '', report.neighborhood.narrative or "")
                    st.caption(_clean_narrative)
                    _clean_highlights = [_re.sub(r'<[^>]+>', '', h) for h in (report.neighborhood.highlights or [])]
                    render_tag_strip(_clean_highlights, tone="#fef3c7")

                with lens_col:
                    st.markdown('<div class="section-title">Decision Lenses</div>', unsafe_allow_html=True)
                    for lens in report.decision_lenses:
                        lens_name = escape_html(lens.lens)
                        lens_summary = escape_html(lens.summary)
                        st.markdown(
                            f"""
                            <div style="background:#ffffff;border:1px solid #dbe4f0;border-radius:18px;padding:16px 18px;
                                        box-shadow:0 8px 20px rgba(15,23,42,0.04);margin-bottom:12px;
                                        overflow:hidden;word-break:break-word;">
                                <div style="font-size:13px;font-weight:800;color:#0f766e;text-transform:uppercase;letter-spacing:0.08em;">
                                    {lens_name}
                                </div>
                                <div style="font-size:14px;color:#475569;margin-top:8px;line-height:1.5;">
                                    {lens_summary}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.markdown("**Recommended Next Steps**")
                    for step in report.decision_summary.next_steps:
                        st.markdown(f"- {step}")

                st.markdown("---")
                risk_col, comp_col = st.columns([1, 1.15])
                with risk_col:
                    st.markdown('<div class="section-title">Risk & Confidence</div>', unsafe_allow_html=True)
                    risk_data = [{"factor": f.factor, "score": f.score, "severity": f.severity} for f in report.risk_factors]
                    fig = chart_risk_breakdown(risk_data)
                    if fig:
                        st.plotly_chart(fig, width="stretch")
                    st.plotly_chart(chart_confidence_gauge(val.confidence), width="stretch")
                with comp_col:
                    st.markdown('<div class="section-title">Comparable Properties</div>', unsafe_allow_html=True)
                    comps = final_state.get("comparables", [])
                    if comps:
                        comp_df = pd.DataFrame(comps)
                        comp_df.columns = ["Price", "Beds", "Baths", "Sqft", "Grade", "Condition", "Age", "$/Sqft"]
                        comp_df["Price"] = comp_df["Price"].apply(format_currency)
                        comp_df["$/Sqft"] = comp_df["$/Sqft"].apply(format_currency)
                        st.dataframe(comp_df, width="stretch", hide_index=True)
                    else:
                        st.info("Comparable sales populate after the pipeline completes.")
                    st.plotly_chart(chart_investment_gauge(val.investment_score), width="stretch")

                st.markdown("---")
                action_col, source_col = st.columns([1, 1])
                with action_col:
                    try:
                        from output.pdf_generator import generate_advisory_pdf
                        pdf_buffer = generate_advisory_pdf(report)
                        st.download_button(
                            label="Download Professional Advisory Report (PDF)",
                            data=pdf_buffer,
                            file_name="property_decision_copilot_report.pdf",
                            mime="application/pdf",
                            width="stretch",
                        )
                    except Exception as e:
                        st.caption(f"PDF generation unavailable: {e}")
                with source_col:
                    if report.rag_sources:
                        st.markdown("**Knowledge Sources**")
                        for src in report.rag_sources:
                            st.markdown(f"- {src}")

                if report.errors:
                    with st.expander("Pipeline Warnings / Fallbacks"):
                        for err in report.errors:
                            st.warning(err)

        with tab2:
            if report:
                st.markdown('<div class="section-title">Scenario Lab</div>', unsafe_allow_html=True)
                s1, s2, s3 = st.columns(3)
                with s1:
                    market_shift_pct = st.slider("Market Shift %", -15, 15, 0, key="scenario_market_shift")
                with s2:
                    offer_delta_pct = st.slider("Offer Delta vs Target %", -10, 10, 0, key="scenario_offer_shift")
                with s3:
                    rate_shift_pct = st.slider("Interest Rate Shock %", -2, 4, 1, key="scenario_rate_shift")

                scenario_df = build_scenario_table(final_state, report, consultation_context, market_shift_pct, offer_delta_pct, rate_shift_pct)
                st.dataframe(scenario_df, width="stretch", hide_index=True)

                scenario_chart_raw = pd.DataFrame([
                    {"Scenario": "Bull", "Scenario Value": report.valuation.predicted_price * (1 + (market_shift_pct + 4) / 100)},
                    {"Scenario": "Base", "Scenario Value": report.valuation.predicted_price * (1 + market_shift_pct / 100)},
                    {"Scenario": "Risk", "Scenario Value": report.valuation.predicted_price * (1 + (market_shift_pct - 6) / 100)},
                ])
                fig_scenario = px.bar(
                    scenario_chart_raw,
                    x="Scenario",
                    y="Scenario Value",
                    color="Scenario",
                    color_discrete_sequence=["#16a34a", "#2563eb", "#dc2626"],
                    title="Scenario Value Envelope",
                )
                fig_scenario.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
                st.plotly_chart(fig_scenario, width="stretch")
                st.info(
                    f"Current negotiation target is {format_currency(report.negotiation.target_price)}. "
                    f"Use this lab to test how market movement and bidding discipline affect downside protection."
                )
            else:
                st.info("Run the advisory pipeline to unlock the scenario lab.")

        with tab3:
            if report:
                st.markdown('<div class="section-title">Model & Market Insights</div>', unsafe_allow_html=True)
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.plotly_chart(chart_feature_importance(), width="stretch")
                with col_c2:
                    st.plotly_chart(chart_price_dist(report.valuation.predicted_price), width="stretch")

                try:
                    prop_for_pred = final_state.get("property_input", {})
                    feature_frame = pd.DataFrame(
                        [[prop_for_pred.get(f, 0) for f in feature_names]],
                        columns=feature_names,
                    )
                    scaled = scaler.transform(feature_frame)
                    tree_preds = np.array([t.predict(scaled)[0] for t in model.estimators_])
                    fig_forest = px.histogram(
                        x=tree_preds,
                        nbins=30,
                        title="Random Forest Distribution",
                        color_discrete_sequence=["#93c5fd"],
                        opacity=0.85,
                    )
                    fig_forest.add_vline(x=tree_preds.mean(), line_dash="dash", line_color="#1d4ed8", annotation_text=f"Mean: {format_currency(tree_preds.mean())}")
                    fig_forest.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
                    st.plotly_chart(fig_forest, width="stretch")
                except Exception:
                    pass

                try:
                    sample = df_full.sample(min(800, len(df_full)), random_state=42)
                    fig_scatter = px.scatter(
                        sample,
                        x="sqft_living",
                        y="price",
                        color="grade",
                        color_continuous_scale=["#bfdbfe", "#2563eb", "#1e3a8a"],
                        title="Price vs Living Area (King County)",
                    )
                    fig_scatter.add_scatter(
                        x=[sqft_living],
                        y=[report.valuation.predicted_price],
                        mode="markers",
                        marker=dict(size=16, color="#dc2626", symbol="star"),
                        name="Subject Property",
                    )
                    fig_scatter.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig_scatter, width="stretch")
                except Exception:
                    pass
            else:
                st.info("Model insights will appear after the decision copilot runs.")

        with tab4:
            st.markdown('<div class="section-title">Agent Queue & Reasoning Summaries</div>', unsafe_allow_html=True)
            render_queue_panel(st.session_state.agent_log)
            st.caption("This queue shows reasoning summaries and status transitions for the multi-agent pipeline. Raw chain-of-thought is intentionally not exposed.")
            with st.expander("Guardrails & Optimization Paths"):
                st.markdown(
                    """
                    - Each advisory run uses an isolated graph thread to prevent stale state leakage.
                    - Chat and web-search inputs pass through topic, injection, length, and sensitive-data filters.
                    - Web search uses region detection, result ranking, blocked-domain filtering, timeouts, and cache reuse.
                    - Agent recursion limits prevent runaway tool loops while keeping the workflow explainable.
                    - Unsafe HTML panels escape dynamic user/LLM text before rendering.
                    """
                )
            if st.session_state.agent_log:
                for idx, entry in enumerate(st.session_state.agent_log, start=1):
                    with st.expander(f"{idx}. {entry['agent']} — {entry['phase']}"):
                        st.markdown(entry["message"])
                        st.caption(f"Output keys: {', '.join(entry['output_keys'])}")
            else:
                st.info("Run the advisory pipeline to see the queue and reasoning summaries.")

    else:
        with tab1:
            st.markdown(
                """
                <div style="background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
                            border:1px solid #d9e5f2;border-radius:24px;padding:28px;
                            box-shadow:0 18px 40px rgba(15,23,42,0.08);">
                    <div style="font-size:12px;font-weight:800;color:#2563eb;letter-spacing:0.14em;text-transform:uppercase;">
                        Get Started
                    </div>
                    <div style="font-size:34px;font-weight:800;color:#0f172a;margin-top:10px;">
                        AI-Powered Property Decision Intelligence
                    </div>
                    <div style="font-size:15px;color:#475569;margin-top:10px;max-width:900px;">
                        Configure the client profile and property details in the sidebar, then run the copilot to generate
                        a comprehensive advisory report with valuation, market context, risk analysis, negotiation strategy, and scenario simulation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("---")
            intro_cols = st.columns(4)
            intro_features = [
                ("Valuation", "ML-driven price prediction with confidence intervals and comparable sales analysis.", "positive"),
                ("Risk Analysis", "8-dimension risk assessment covering market, location, and property-specific factors.", "warning"),
                ("Negotiation", "Data-backed anchor, target, and walk-away pricing with leverage and caution points.", "accent"),
                ("Advisory Report", "Comprehensive decision memo with market context, scenarios, and downloadable PDF.", "neutral"),
            ]
            for col, (title, copy, tone) in zip(intro_cols, intro_features):
                with col:
                    st.markdown(
                        f"""
                        <div style="background:#ffffff;border:1px solid #dbe4f0;border-left:4px solid {'#16a34a' if tone == 'positive' else '#d97706' if tone == 'warning' else '#0f766e' if tone == 'accent' else '#2563eb'};
                                    border-radius:12px;padding:20px 18px;box-shadow:0 8px 20px rgba(15,23,42,0.05);
                                    min-height:140px;">
                            <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:8px;">{title}</div>
                            <div style="font-size:13px;color:#475569;line-height:1.5;">{copy}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


        with tab2:
            st.info("Scenario Lab will unlock after a property analysis is generated.")

        with tab3:
            st.info("Model insights will unlock after a property analysis is generated.")

        with tab4:
            st.info("The live queue will appear after the first advisory run.")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;padding:20px 0 10px;">
            <p style="color:#64748b;font-size:14px;margin-bottom:8px;">
                Need fast conversational guidance for any US city?
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if st.button("Open Property Chatbot", key="bottom_chat_btn", width="stretch"):
            st.session_state["_chatbot_switch"] = True
            st.rerun()