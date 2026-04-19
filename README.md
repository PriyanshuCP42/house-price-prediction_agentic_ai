# 🏠 House Price Prediction Agentic AI

A high-performance, multi-agent real estate advisory system that combines **Machine Learning (ML)** for precise valuations with **Large Language Models (LLM)** for strategic reasoning. 

Built using **LangGraph**, **Streamlit**, and **Groq**, this system provides institutional-grade property analysis for buyers, sellers, and investors.

---

## 🚀 Key Features

### 1. Decision Copilot (Multi-Agent Pipeline)
Our core engine uses a directed acyclic graph (DAG) to orchestrate specialized AI agents:
- **ML Predictor**: Uses a Random Forest model trained on King County house prices to provide factual baseline valuations.
- **Search Agent**: Real-time web search (DuckDuckGo) to fetch current market context, neighborhood trends, and recent sales.
- **Advisory Agent**: Analyzes property data, investment objectives, and risk factors to generate strategic guidance.
- **Synthesizer**: Consolidates quantitative and qualitative data into a professional advisory report.

### 2. Scenario Lab
Interactively simulate "what-if" scenarios:
- Adjust property sqft, condition, or grade and see real-time price delta simulations.
- Analyze how market volatility affects predicted property value.

### 3. Model Insights & Transparency
Gain deep visibility into the "Black Box" of ML:
- **Feature Importance**: See which factors (living area, location, age) drive price the most.
- **Random Forest Probability Distribution**: High-resolution histogram showing the variance across the model's decision trees.
- **Market Scatters**: Localized price vs. sqft living benchmarks against historical data.

### 4. AI Property Chatbot
A conversational interface designed for fast, practical real-estate guidance across any US location, powered by Llama 3.1 (via Groq) with a fallback to Google Gemini Flash.

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Reactive UI)
- **Orchestration**: LangGraph (Cyclic state management)
- **Models**: 
  - Llama 3.1 8B (via **Groq** for extreme speed)
  - Gemini 2.0 Flash (via **Google AI** as fallback)
  - Random Forest Regressor (Custom Scikit-Learn Model)
- **Data & Visualization**: Pandas, Numpy, Plotly
- **Tools**: DuckDuckGo Search, Mapbox (Geospatial context)

---

## 📂 Project Structure

```text
├── agents/            # Multi-agent node & tool logic
├── config/            # LLM, Guardrails, and Page settings
├── models/            # Saved ML artifacts (Scaler, Random Forest pkl)
├── output/            # Structured data schemas (Pydantic)
├── data/              # Sample dataset for live benchmarking
├── streamlit_app.py   # Primary production entrypoint
└── advisory_app.py    # Core application logic
```

---

## ⚙️ Installation & Setup

### Local Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/PriyanshuCP42/house-price-prediction_agentic_ai.git
   cd house-price-prediction_agentic_ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets**:
   Create a `.streamlit/secrets.toml` file:
   ```toml
   GROQ_API_KEY = "your_groq_key_here"
   GOOGLE_API_KEY = "your_google_key_here"
   ```

4. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

### Streamlit Cloud Deployment
1. Push your code to GitHub.
2. Deploy on Streamlit Community Cloud selecting `streamlit_app.py` as the entrypoint.
3. Add your `GROQ_API_KEY` and `GOOGLE_API_KEY` to the **Secrets** tab in Advanced Settings.

---

## 📋 Environment Variables

| Variable | Description | Source |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Primary LLM (Llama 3.1) | [Groq Console](https://console.groq.com/) |
| `GOOGLE_API_KEY` | Fallback LLM (Gemini 2.0) | [Google AI Studio](https://aistudio.google.com/) |

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

**Author**: [PriyanshuCP42](https://github.com/PriyanshuCP42)
