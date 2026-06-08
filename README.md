# 🛡️ ChurnGuard AI

**Production-Oriented Customer Churn Prediction & Retention Intelligence Platform**

ChurnGuard AI predicts which customers are likely to churn, explains *why* using
SHAP, segments customers by risk, and generates concrete, rule-based retention
recommendations — all served through a FastAPI backend and an interactive
Streamlit dashboard, fully containerized with Docker.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Architecture](#-architecture)
4. [Dataset](#-dataset)
5. [Feature Engineering](#-feature-engineering)
6. [Model Comparison](#-model-comparison)
7. [Evaluation Metrics](#-evaluation-metrics)
8. [SHAP Explainability](#-shap-explainability)
9. [Risk Segmentation & Recommendations](#-risk-segmentation--recommendations)
10. [Quickstart](#-quickstart)
11. [API](#-api)
12. [Dashboard](#-dashboard)
13. [Docker Setup](#-docker-setup)
14. [Testing](#-testing)
15. [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

This is a portfolio-grade machine learning application designed to resemble a
lightweight production-ready analytics platform rather than a notebook project.
It includes:

- Machine learning predictions (binary churn classification)
- Explainable AI (global & local SHAP)
- Rule-based retention recommendations
- Interactive Streamlit dashboard
- FastAPI backend with structured JSON schemas
- Dockerized, reproducible deployment

## 📉 Business Problem

Acquiring a new customer costs **5–7× more** than retaining one. For a
subscription telco, reducing churn even slightly protects significant recurring
revenue. ChurnGuard AI gives retention teams the ability to **predict**,
**understand**, **segment**, and **act** on churn risk.

## 🏗️ Architecture

```text
project-root/
├── data/                # raw/ and processed/ datasets
├── notebooks/           # EDA, feature engineering, model experiments
├── src/                 # production ML logic
│   ├── config.py            # paths, schema, thresholds (no hardcoding)
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py       # trains & compares 3 models, saves artifacts
│   ├── predict.py           # inference + risk segmentation
│   ├── evaluate.py          # metrics
│   ├── explainability.py    # SHAP global/local
│   ├── recommendation_engine.py
│   └── utils.py
├── app/                 # Streamlit dashboard (Home + pages + components)
├── api/                 # FastAPI backend (main, routes, schemas)
├── models/              # saved artifacts (.pkl)
├── reports/             # metrics.json + figures
├── tests/               # pytest unit & API tests
├── docker/Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 📊 Dataset

**BigML Telecom Churn** dataset, delivered **pre-split** into:

- `data/raw/churn-bigml-80.csv` — 80% training split
- `data/raw/churn-bigml-20.csv` — 20% test split

- **Target:** `Churn` (`True` = churned, `False` = retained)
- ~3,300 customers, 19 features: account info (State, Account length, Area code),
  plans (International plan, Voice mail plan), call usage by period
  (day / evening / night / international minutes, calls, charges) and
  `Customer service calls`.

Because the data is already split, training uses the 80% file and evaluates on
the held-out 20% file — there is no internal train/test split.

## 🧩 Feature Engineering

Business-oriented, interpretable features (see `src/feature_engineering.py`):

| Feature | Description |
|---|---|
| `tenure_group` | Lifecycle stage bucketed from Account length (New … Veteran) |
| `service_count` | Number of subscribed plans (International + Voice mail) |
| `total_charge` | Total spend across day/eve/night/international usage |
| `avg_charge_per_minute` | Pricing-efficiency proxy (safe for zero usage) |
| `customer_value_score` | Approximate lifetime value (spend × account length) |
| `service_call_risk` | Support-contact intensity — the strongest churn signal |

Feature engineering, encoding (one-hot) and scaling (standardization) are all
wrapped in a **single fitted sklearn `Pipeline`** saved as
`models/xgboost_model.pkl`:

```text
raw features → FunctionTransformer(add_features) → ColumnTransformer(OneHot + StandardScaler) → XGBoost
```

So one object handles feature engineering, encoding, scaling and inference —
`pipeline.predict_proba(raw_df)` is all that's needed, with zero train/serve
skew. (The fitted `ColumnTransformer` step is also saved separately as
`models/scaler.pkl` for SHAP.)

## 🤖 Model Comparison

Three models are trained and compared in `src/train_model.py`:

1. **Logistic Regression** (balanced class weights)
2. **Random Forest** (balanced class weights)
3. **XGBoost** ← **production model** (`scale_pos_weight` for imbalance)

Full metrics for all three are written to `reports/metrics.json` after training.

## 📈 Evaluation Metrics

Accuracy is **never** used in isolation. Reported metrics (`src/evaluate.py`):

- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix

Class imbalance is handled via class weighting / `scale_pos_weight`.

## 🧠 SHAP Explainability

- **Global:** feature importance + SHAP summary (beeswarm) plot
- **Local:** per-customer waterfall / contribution breakdown showing which
  features **increase** and which **decrease** churn probability

Served on the dashboard's **SHAP Analysis** page and via the `/explain` endpoint.

## 🚦 Risk Segmentation & Recommendations

Churn probability is mapped to configurable segments (`src/config.py`):

| Segment | Probability |
|---|---|
| 🟢 Low Risk | < 0.30 |
| 🟠 Medium Risk | 0.30 – 0.70 |
| 🔴 High Risk | ≥ 0.70 |

The rule-based engine (`src/recommendation_engine.py`) turns risk + attributes
into actions, e.g.:

- High risk + high value → **loyalty discount**
- No tech support → **premium support package**
- Month-to-month contract → **annual subscription offer**

## ⚡ Quickstart

```bash
# 1. Create environment & install deps
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell)
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt

# 2. Add the dataset
#    -> place churn-bigml-80.csv and churn-bigml-20.csv in data/raw/

# 3. Train the model (writes artifacts to models/ and reports/metrics.json)
python -m src.train_model

# 4a. Run the dashboard
streamlit run app/Home.py

# 4b. ...or run the API
uvicorn api.main:app --reload --port 8000
```

## 🔌 API

Interactive docs at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness + model-loaded status |
| POST | `/predict` | Churn probability + risk segment |
| POST | `/explain` | Prediction + local SHAP explanation |
| POST | `/recommend` | Prediction + retention recommendations |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"State":"OH","Account length":84,"Area code":"408","International plan":"Yes","Voice mail plan":"No","Number vmail messages":0,"Total day minutes":299.4,"Total day calls":71,"Total day charge":50.9,"Total eve minutes":61.9,"Total eve calls":88,"Total eve charge":5.26,"Total night minutes":196.9,"Total night calls":89,"Total night charge":8.86,"Total intl minutes":6.6,"Total intl calls":7,"Total intl charge":1.78,"Customer service calls":5}'
```

## 🖥️ Dashboard

Streamlit pages:

- **Home** — overview & business problem
- **Prediction** — customer form + churn scoring (KPI cards, gauge, risk card)
- **SHAP Analysis** — global importance + per-customer explanations
- **Customer Insights** — EDA charts & KPIs
- **Recommendations** — prioritized retention strategies

> 📸 _Add dashboard screenshots to `reports/figures/` and embed them here._

## 🐳 Docker Setup

```bash
# Train artifacts on the host first (mounted into the containers)
python -m src.train_model

# Build & start API (:8000) and dashboard (:8501)
docker compose up --build
```

- Dashboard → http://localhost:8501
- API docs → http://localhost:8000/docs

## ✅ Testing

```bash
pytest -q
```

Tests cover feature engineering, evaluation metrics, risk segmentation, the
recommendation engine and the API. Tests requiring a trained model are skipped
automatically when artifacts are absent.

## 🔭 Future Improvements

- Real-time / streaming inference
- Authentication & role-based access
- Database integration for customer records
- Automated retraining pipelines & model monitoring
- LLM-powered, personalized retention messaging

---

_Built with Python · Pandas · scikit-learn · XGBoost · SHAP · Streamlit · FastAPI · Docker._
