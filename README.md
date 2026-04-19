# 🏠 House Price Prediction Agentic AI

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ML](https://img.shields.io/badge/ML-Random_Forest-8CAAE6?style=for-the-badge&logo=scikit-learn&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-000000?style=for-the-badge&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-f3d122?style=for-the-badge&logo=groq&logoColor=black)

A state-of-the-art real estate intelligence platform that bridges the gap between **Deterministic Machine Learning** and **Agentic Reasoning**. This system doesn't just predict a price; it plans, researches, and advises like a professional human consultant.

---

## 🧠 Agentic AI Architecture (The Brain)

Unlike traditional models that only output a number, this project uses a **Multi-Agent Orchestration** framework powered by **LangGraph**. The system treats the advisory process as a series of specialized cognitive tasks:

### ⚙️ The Graph Workflow
1. **The Planner**: Receives the initial property data and determines the "Search Strategy." It decides what market data is missing for this specific zipcode.
2. **The ML Core Agent**: Directly interfaces with the **Random Forest Regressor**. It provides the "Ground Truth" valuation, which serves as a factual anchor for the rest of the pipeline.
3. **The Web Research Agent**: Equipped with **DuckDuckGo Search**, it scans for real-time market trends, local neighborhood developments, and interest rate impacts that a static model might miss.
4. **The Risk Analyst**: Performs a multi-dimensional assessment (Market Risk, Physical Risk, Location Risk) based on the combined output of search results and model predictions.
5. **The Final Synthesizer**: Uses **Llama 3.1 (via Groq)** to cross-reference all agent findings, resolving conflicts between "model prediction" and "actual market sentiment" to produce the final Decision Memo.

### 🤖 Why "Agentic"?
- **Autonomous Error Correction**: If the search agent finds no data, the graph can route back to attempt a different search query.
- **Stateful Memory**: The app maintains a "Thread" for each property run, allowing the AI to remember previous context during a session.
- **Tool Use**: Agents autonomously decide when to use the ML Model vs. when to use the Web Search tool.

---

## 🚀 Key Features

### 1. Decision Copilot
- **ML Baseline**: Precison valuation for King County properties.
- **Real-time Context**: Fetches neighborhood "vibe" and recent infrastructure changes.
- **Strategic Negotiation**: Generates Anchor, Target, and Walk-away prices.

### 2. Scenario Lab
- **Sensitivity Analysis**: See exactly how much a 1-point increase in property "Grade" increases your value ($).
- **Interactive Deltas**: Adjustable sliders let you simulate renovations and immediately view the ROI.

### 3. AI Property Chatbot
- **Universal Search**: Guidance for any US location, not just King County.
- **Natural Language Reasoning**: Ask complex questions like *"Is it better to buy a 3-bed in Seattle or a 4-bed in Renton right now?"*

---

## 🛠️ Technical Implementation

### Core Dependencies
- **LangGraph**: Orchestrates the cyclic agent workflow.
- **Scikit-Learn**: Powering the Random Forest ensemble model.
- **Groq & Llama 3.1**: Providing sub-second inference speeds for the LLM agents.
- **Plotly**: Generating high-fidelity visual insights and probability distributions.

### Model Transparency
We expose the **Random Forest Prediction Intervals** to the user. Instead of a single number, the **Model Insights** tab shows a distribution of all tree predictions, providing a "Confidence Window" for the valuation.

---

## 📂 Project Organization

```text
├── agents/             
│   ├── nodes/          # Task-specific agent nodes (Search, ML, Advisor)
│   ├── tools/          # Discrete tools (DuckDuckGo, Model Predictor)
│   └── state.py        # Shared Graph State definition
├── config/             
│   ├── settings.py     # Hyperparameters and model versions
│   └── llm_config.py   # Provider logic (Groq/Gemini)
├── models/             # Pre-trained ML pkl files and scalers
└── streamlit_app.py    # Main UI entrypoint
```

---

## ⚙️ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Secrets**:
   Add to `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

3. **Launch**:
   ```bash
   streamlit run streamlit_app.py
   ```

---

**Built with ❤️ for Modern Real Estate Intelligence.**
**Author**: [PriyanshuCP42](https://github.com/PriyanshuCP42)
