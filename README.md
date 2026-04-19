# 🏠 House Price Prediction System  
### AI-Powered Real Estate Intelligence — King County, Washington  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-green.svg)]()

🔗 **Live Application:** Deploy from this repository on Streamlit Community Cloud.

---

## 📌 Overview

Real estate pricing lacks transparency. Buyers and sellers often cannot verify whether a listing price is fair. Professional appraisals are expensive, and automated commercial estimates are frequently inaccurate.

This project builds a production-ready **Random Forest regression model** trained on 21,613 historical property sales from King County, Washington.

The application allows users to input property details and receive:

- AI-predicted price  
- Confidence range  
- Investment score  
- Market status classification  
- Comparable sales comparison  
- Downloadable PDF advisory report  

Built using Python, Scikit-learn, and Streamlit.

---

## 🚀 Features

### 🔮 Prediction Engine

- **Predicted Price** — Mean across 200 Random Forest trees  
- **Price Range** — ±1 standard deviation (model uncertainty)  
- **Confidence Score** — Based on tree prediction agreement  
- **Investment Score (0–100)** — Rule-based evaluation  
- **Market Status** — Underpriced / Fair / Overpriced  

---

### 📊 Analytics Dashboard

- Feature Importance (Top 12 price drivers)  
- Price Distribution Histogram  
- Forest Distribution (200 tree predictions)  
- Price vs Living Area Scatter Plot  
- Comparable Property Comparison (Top 6 similar homes)  

All visualizations are interactive using Plotly.

---

### 📄 PDF Advisory Report

Generates a structured professional report containing:

- Executive summary  
- AI recommendation  
- Confidence breakdown  
- Comparable sales table  
- Legal disclaimer  

Built using ReportLab.

---

## 🛠 Tech Stack

| Layer | Technology |
|--------|------------|
| Language | Python 3.10+ |
| ML Model | Scikit-learn (Random Forest Regressor) |
| Data Processing | Pandas, NumPy |
| Web App | Streamlit |
| Visualization | Plotly |
| PDF Generation | ReportLab |
| Training | Google Colab |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 📊 Dataset

| Attribute | Value |
|------------|-------|
| Source | King County House Sales Dataset |
| Records | 21,613 |
| Features | 20 predictors + 1 target (price) |
| Time Period | May 2014 – May 2015 |
| Geography | King County, Washington |
| Price Range | $75,000 – $7,700,000 |

---

### 🔎 Top Predictors by Correlation

- `sqft_living` (0.70)  
- `grade` (0.67)  
- `sqft_above` (0.61)  
- `sqft_living15` (0.59)  
- `bathrooms` (0.53)  
- `latitude` (0.31)  
- `waterfront` (0.27)  

---

## 🏗 Engineered Features

- `house_age` — Derived from year built  
- `renovated` — Binary renovation flag  
- `amenity_score` — Composite quality metric  
- `zipcode_encoded` — Target encoding for location premium  

---

## ⚙️ Machine Learning Pipeline

1. Load dataset  
2. Perform EDA  
3. Remove extreme outliers  
4. Clean and preprocess data  
5. Feature engineering  
6. Train-test split (80/20)  
7. Target encode zipcode (training set only)  
8. Scale features  
9. Train baseline Linear Regression  
10. Train final Random Forest model  
11. Evaluate on test set  
12. Export model bundle (`model.pkl`)  

---

## 🔐 Data Leakage Prevention

- Train-test split performed before encoding  
- Scaler fitted only on training data  
- Zipcode encoding computed using training rows only  
- Evaluation strictly on held-out test set  

---

## 📈 Model Performance

| Metric | Linear Regression | Random Forest |
|--------|------------------|---------------|
| R² | ~0.67 | ~0.88 |
| MAE | ~$110,000 | ~$80,000 |
| RMSE | ~$180,000 | ~$135,000 |

An R² of 0.88 means the model explains 88% of price variance in unseen data.

---

## 🌲 Random Forest Configuration

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

---

## 🎯 Confidence Score Formula

```python
confidence = max(0, min(100, 100 - (std / mean) * 100))
```

---

## 📂 Project Structure

```
house-price-prediction/
│
├── streamlit_app.py          # Streamlit Cloud entrypoint
├── advisory_app.py           # Main AI Property Decision Copilot
├── app.py                    # Original baseline app
├── model.pkl
├── kc_house_data.csv
├── runtime.txt
├── requirements.txt
└── README.md
```

---

## 🏁 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run advisory_app.py
```

Application runs at:

```
http://localhost:8501
```

---

## ☁️ Streamlit Cloud Deployment

This repo is prepared for Streamlit Community Cloud.

### Deployment Settings

| Setting | Value |
|--------|-------|
| Repository | `https://github.com/abhijeetnst/property-agent` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| Python runtime | `runtime.txt` (`python-3.12.8`) |

### Secrets

The app can start without API keys because the advisory pipeline has deterministic fallbacks. For the best chatbot and narrative quality, add at least one provider key in Streamlit Cloud under **App settings → Secrets**:

```toml
GROQ_API_KEY = "your_groq_key"
GOOGLE_API_KEY = "your_google_key"
```

You can also use `.streamlit/secrets.toml.example` as a template. Do not commit real secrets.

### Deployment Notes

- `streamlit_app.py` imports `advisory_app.py`, so Streamlit Cloud can auto-detect the app.
- `.venv`, Python caches, Playwright logs, Chroma cache, and real Streamlit secrets are ignored by git.
- RAG now uses a lightweight markdown knowledge-base search so the deployed app does not need heavy ChromaDB, Torch, or sentence-transformers installs.
- `model.pkl` is included in the repo and is required for the trained King County price model.

---

## 📦 requirements.txt

```
streamlit>=1.32.0,<2.0
pandas>=2.0.0,<3.0
numpy>=1.24.0,<2.0
scikit-learn>=1.6.1,<1.7
plotly>=5.15.0,<6.0
reportlab>=4.0.0,<5.0
langgraph>=1.1.0,<2.0
langchain-core>=1.2.0,<2.0
langchain-groq>=1.1.0,<2.0
langchain-google-genai>=4.0.0,<5.0
pydantic>=2.0.0,<3.0
ddgs>=9.0.0,<10.0
```

---

## ⚠️ Limitations

- Trained on 2014–2015 data  
- Limited to King County region  
- Investment score is heuristic-based  
- Not financial advice  

---

## 🤝 Contributing

1. Fork the repository  
2. Create feature branch  
3. Commit changes  
4. Push branch  
5. Open Pull Request  

---

## 👥 Contributors

**Priyanshu Agrahari — Team Lead & ML Pipeline Architect**  
- Led end-to-end project execution and team coordination  
- Designed and implemented the complete ML pipeline (preprocessing → feature engineering → model training → evaluation)  
- Contributed to key modeling decisions and performance improvements  
- Built and deployed the Streamlit application  

**Mihika Mathur**  
- Implemented and trained the Random Forest model  
- Contributed to feature engineering and model optimization  
- Assisted in evaluation and performance tuning  

**Vishuti Jamwal**  
- Conducted dataset research and data understanding  
- Managed documentation and reporting  
- Performed validation and testing  

**Abhijeet**  
- Supported validation, testing, and model review  

---
