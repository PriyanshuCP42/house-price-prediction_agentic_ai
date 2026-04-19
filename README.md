# 🏠 Intelligent House Price Prediction & Agentic Real Estate Advisory

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-green.svg)](https://scikit-learn.org)
[![LangGraph](https://img.shields.io/badge/Agentic-LangGraph-purple.svg)](https://www.langchain.com/langgraph)

🔗 **Live Demo:** [Open App](https://house-price-predictionagenticai-zsvtnbxvp3t97pdarkua66.streamlit.app/) &nbsp;|&nbsp; 📦 **[Requirements](requirements.txt)** &nbsp;|&nbsp; 🚀 **[App Entry](streamlit_app.py)** &nbsp;|&nbsp; 📓 **[Training Notebook](HousePricePropertyPredictions.ipynb)**

> *From Demand Forecasting to Autonomous Real Estate Advisory* — a full-stack Agentic AI system that predicts house prices, detects market positioning, retrieves investment knowledge, and generates optimized buy/sell decisions.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Why This Project Matters](#-why-this-project-matters)
- [System Architecture](#-system-architecture)
- [LangGraph Agent Workflow](#-langgraph-agent-workflow)
- [Input and Output Specification](#-input-and-output-specification)
- [Repository Structure](#-repository-structure)
- [Model Performance](#-model-performance)
- [Optimization and Planning Logic](#-optimization-and-planning-logic)
- [RAG and Robustness Design](#-rag-and-robustness-design)
- [UI and Demo Flow](#-ui-and-demo-flow)
- [Installation and Run Instructions](#-installation-and-run-instructions)
- [Deployment](#-deployment)

---

## 📖 Project Overview

This project addresses real estate price intelligence as a two-layer AI system:

1. A **machine learning layer** predicts property prices from historical King County sales data using a Random Forest Regressor (R² = 0.88).
2. A **stateful LangGraph agent layer** interprets prediction outputs, retrieves investment guidelines through RAG, and generates explainable advisory reports with negotiation strategy and risk assessment.

The current repository is the agentic version of the project. It extends the original ML forecasting pipeline into a production-oriented real estate advisory assistant with price prediction, market status classification, LangGraph state transitions, RAG-based knowledge retrieval, optimization-aware reasoning, robust failure handling, and a Streamlit-based decision support UI.

---

## 🌍 Why This Project Matters

Real estate pricing is not only a prediction problem. Buyers, sellers, and investors also need to answer questions such as:

- Is this listing priced fairly relative to comparable properties?
- What is the optimal negotiation anchor and walk-away price?
- How do structural modifications (renovation, added bathrooms) affect value?
- What is the investment risk given current market conditions?

This system moves from raw property analytics to **explainable investment recommendations** useful for real-world buying, selling, and portfolio decisions.

---

## 🏗 System Architecture

The end-to-end system combines ML prediction, agentic reasoning, RAG retrieval, and UI delivery.

### Architecture Summary

| Layer | Responsibility | Main Files |
|---|---|---|
| Input Layer | Accept property features and user investment objective | `streamlit_app.py` |
| Validation Layer | Validate inputs, handle missing fields, detect edge cases | `advisory_app.py` |
| ML Layer | Load trained model, generate price predictions, compute confidence intervals | `model.pkl`, `advisory_app.py` |
| Agent Layer | Maintain explicit state and execute node-based workflow | `agents/` |
| Retrieval Layer | Load RAG knowledge base and fetch investment guidelines | `rag/` |
| Planning Layer | Optimize negotiation strategy, investment score, and risk classification | `agents/` |
| UI Layer | Present metrics, reasoning, advisory report, and structured output | `streamlit_app.py`, `advisory_app.py` |

---

## 🤖 LangGraph Agent Workflow

The system uses a LangGraph-based stateful agent workflow managed as a directed acyclic graph (DAG).

### State Object

| State Field | Purpose |
|---|---|
| `property_input` | User-provided property features for analysis |
| `processed_features` | Model-ready engineered feature frame |
| `predictions` | Predicted price, confidence interval, and tree distribution |
| `market_status` | Underpriced / Fair Value / Overpriced classification |
| `retrieved_docs` | Retrieved investment guidelines or fallback rules |
| `reasoning` | Intermediate negotiation and risk analysis insights |
| `final_plan` | Structured advisory output with recommendations |
| `summary` | Predicted price, investment score, confidence, risk level |
| `errors` | Critical issues surfaced to the UI |
| `warnings` | Non-fatal issues such as imputation or retrieval fallback |

### Execution Logic

| Node | What It Does |
|---|---|
| Input Node | Accepts property features and initializes workflow state |
| Preprocessing Node | Applies feature engineering (house age, amenity score, renovation flag, zipcode encoding) |
| Prediction Node | Runs the trained Random Forest and produces price prediction with confidence interval |
| Market Status Node | Classifies the property as Underpriced, Fair Value, or Overpriced |
| RAG Retrieval Node | Retrieves investment planning guidelines from the knowledge base |
| Reasoning Node | Applies investment scoring, negotiation anchoring, and risk assessment |
| Advisory Node | Produces final recommendation, negotiation strategy, and scenario analysis |
| Output Node | Formats a consistent final payload for the UI and PDF report |

---

## 📥 Input and Output Specification

### Input

| Field | Required | Description |
|---|---|---|
| `bedrooms` | Yes | Number of bedrooms |
| `bathrooms` | Yes | Number of bathrooms |
| `sqft_living` | Yes | Interior living area in square feet |
| `sqft_lot` | Yes | Total lot size in square feet |
| `floors` | Yes | Number of floors |
| `waterfront` | No | Binary flag for waterfront property |
| `view` | No | View quality score (0–4) |
| `condition` | No | Property condition rating (1–5) |
| `grade` | Yes | Construction and design grade (1–13) |
| `yr_built` | Yes | Year of original construction |
| `yr_renovated` | No | Year of most recent renovation |
| `zipcode` | Yes | Property zipcode for location premium encoding |
| `sqft_living15` | No | Average living area of 15 nearest neighbors |

If optional fields are missing, the system imputes defaults and continues with warnings instead of crashing.

### Output

| Output | Description |
|---|---|
| `predicted_price` | Mean price prediction across all 200 Random Forest trees |
| `price_range` | ±1 standard deviation confidence interval |
| `confidence_score` | 0–100 score based on tree prediction agreement |
| `investment_score` | Rule-based composite score (0–100) across value, condition, and location |
| `market_status` | Underpriced / Fair Value / Overpriced classification |
| `negotiation_strategy` | Anchor price and walk-away threshold |
| `comparable_sales` | Top 6 similar properties from the training dataset |
| `reasoning` | Explainable investment and risk analysis |
| `pdf_report` | Downloadable structured advisory report |
| `errors and warnings` | Safe failure and degradation feedback |

---

## 📂 Repository Structure

### Directory Overview

| Directory | Purpose |
|---|---|
| `agents/` | LangGraph workflow, state definition, agent node logic |
| `rag/` | RAG knowledge base and retrieval logic |
| `config/` | Configuration and prompt templates |
| `output/` | Generated PDF advisory reports |
| `.streamlit/` | Streamlit secrets and configuration |

### Key Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Streamlit Cloud entrypoint |
| `advisory_app.py` | Main AI Property Decision Copilot UI and agent orchestration |
| `app.py` | Original baseline single-model app |
| `model.pkl` | Trained Random Forest + scaler bundle |
| `kc_house_data.csv` | King County training dataset (21,613 records) |
| `HousePricePropertyPredictions.ipynb` | Full EDA and model training notebook |
| `requirements.txt` | Pinned dependency manifest |

---

## 📈 Model Performance

The ML layer uses the King County dataset with a chronological 80/20 train-test split.

### Current Held-Out Metrics

| Metric | Value | Interpretation |
|---|---|---|
| MAE | ~$80,000 | Average absolute prediction error |
| RMSE | ~$135,000 | Larger errors are penalized more strongly |
| R² Score | 0.88 | The model explains 88% of price variance in unseen data |

### Model Comparison

| Rank | Model | R² |
|---|---|---|
| 1 | Random Forest Regressor | 0.88 |
| 2 | Linear Regression | 0.67 |

### Evaluation Notes

- The best exported model is `model.pkl` (Random Forest with 200 estimators).
- The test set contains held-out rows based on an 80/20 chronological split.
- Feature engineering (house age, amenity score, renovation flag, zipcode encoding) was fitted on training data only to prevent leakage.

---

## ⚙️ Optimization and Planning Logic

The advisory layer is not a simple price lookup. It explicitly reasons over investment score, negotiation bounds, renovation ROI, and risk classification.

### Decision Factors

| Factor | How It Is Used |
|---|---|
| Investment score | Composite of predicted value vs. market price, condition, location, and grade |
| Negotiation bounds | Anchor price set below predicted value; walk-away threshold set at upper confidence bound |
| Renovation ROI | Scenario laboratory re-scores the property after structural modifications |
| Risk classification | Properties with condition/location mismatches are flagged as high risk |
| Market status | Underpriced properties trigger buy recommendations; Overpriced triggers caution |

### Planning Output Example

| Output Element | Example Interpretation |
|---|---|
| Predicted Price | Mean across 200 trees with ±1 std confidence band |
| Investment Score | 0–100 composite — higher means stronger buy signal |
| Market Status | Underpriced / Fair Value / Overpriced |
| Negotiation Strategy | Anchor and walk-away prices derived from confidence interval |
| Recommendations | Actionable steps for pricing, renovation, and timing |

---

## 🛡 RAG and Robustness Design

### Robustness Features

| Failure Case | System Behavior |
|---|---|
| Missing required input fields | Shows a clear validation error with required field names |
| Missing optional features | Imputes defaults and continues with warnings |
| Model prediction failure | Returns a safe prediction failure message |
| No RAG documents retrieved | Uses fallback guideline text |
| Agent workflow failure | Returns a safe structured fallback response |

### Why This Matters for Deployment

The hosted application must remain stable even when:
- the property input is incomplete
- RAG retrieval returns no relevant result
- the model or agent pipeline encounters malformed inputs

---

## 🖥 UI and Demo Flow

The Streamlit app is designed to be quick to understand during a demo or viva.

| Section | What the User Sees |
|---|---|
| Property Input | Form to enter property features and investment objective |
| Predicted Price | Price estimate with confidence interval and investment score |
| Market Status | Underpriced / Fair Value / Overpriced classification |
| Analytics Dashboard | Feature importance, forest distribution, comparable sales charts |
| Reasoning | Explainable investment logic and negotiation trade-offs |
| Scenario Laboratory | What-if simulation for renovations and structural changes |
| Advisory Chatbot | Conversational interface with property context memory |
| PDF Report | Downloadable structured advisory report |

---

## 🚀 Installation and Run Instructions

### Prerequisites

- Python 3.10 or higher
- `pip`
- Git

### Local Setup

```bash
git clone https://github.com/PriyanshuCP42/house-price-prediction_agentic_ai.git
cd house-price-prediction_agentic_ai
```

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

### Configure API Keys

Populate `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_key"
GOOGLE_API_KEY = "your_key"
MAPBOX_API_KEY = "your_key"
```

> The app runs without API keys using deterministic fallbacks. LLM-powered narrative and chatbot features require at least one of `GROQ_API_KEY` or `GOOGLE_API_KEY`.

### Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

### Optional Utility Runs

Rebuild the RAG knowledge base from local guideline documents:

```bash
python rag/build_knowledge_base.py
```

---

## ☁️ Deployment

| Component | Status | Link |
|---|---|---|
| Hosted Streamlit Demo | Available | [Open App](https://house-price-predictionagenticai-zsvtnbxvp3t97pdarkua66.streamlit.app/) |
| Local Streamlit App | Available | `streamlit run streamlit_app.py` |
| RAG Knowledge Base | Lightweight lexical search — no ChromaDB required at runtime | `rag/` |

### Deployment Notes

- `streamlit_app.py` imports `advisory_app.py`; Streamlit Cloud auto-detects the app correctly.
- `model.pkl` is committed to the repo and loaded at runtime — no model download step required.
- Add API keys under **App settings → Secrets** in the Streamlit Cloud dashboard.
- `chromadb` and `sentence-transformers` are only needed to rebuild the offline knowledge base locally.

---

## 👥 Contributors

| Name | Role |
|---|---|
| **Priyanshu Agrahari** | Team Lead & Architect — led end-to-end development; built ML pipeline, designed agentic AI system, and handled deployment |
| **Mihika Mathur** | ML Engineer & QA — implemented model & feature engineering; handled documentation and ensured system reliability through testing |
| **Vishuti Jamwal** | Research & System Design — conducted research, created architecture diagrams, wrote reports, and contributed to QA/testing |
| **Abhijeet** | QA & AI Agent Developer — worked on validation/testing in ML phase and co-developed the agent system with core logic & integration |

---

## 📄 License

This repository is intended for academic use in the house price prediction and agentic real estate advisory project.
