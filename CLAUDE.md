# CLAUDE.md

# ChurnGuard AI

Production-Oriented Customer Churn Prediction & Retention Intelligence Platform

---

# Project Overview

This project is a portfolio-grade machine learning application designed to predict customer churn risk and provide explainable business insights.

The goal is not to create a simple notebook-based ML project.

The system should resemble a lightweight production-ready analytics platform with:

* machine learning predictions
* explainable AI
* retention recommendations
* interactive dashboard
* API support
* Dockerized deployment

---

# Main Objectives

The platform must:

1. Predict customer churn probability
2. Explain why a customer may churn
3. Categorize customers by risk level
4. Generate retention recommendations
5. Visualize insights through an interactive dashboard

---

# Dataset

Dataset:
BigML Telecom Churn Dataset (delivered pre-split into 80% / 20% files)

Files:

* data/raw/churn-bigml-80.csv  (training split)
* data/raw/churn-bigml-20.csv  (test split)

Because the data is already split, training uses the 80% file and evaluates on
the held-out 20% file. There is no internal train/test split.

Target column:

* Churn

Target values:

* True = churned customer
* False = retained customer

Key features:

* Account info: State, Account length, Area code
* Plans: International plan, Voice mail plan, Number vmail messages
* Usage by period: day / evening / night / international minutes, calls, charges
* Customer service calls (strongest churn signal)

---

# Tech Stack

## Core Technologies

* Python 3.11+
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* FastAPI
* Docker

---

# Project Architecture

```text id="i5v9m2"
project-root/
│
├── data/
│   ├── raw/
│   │   ├── churn-bigml-80.csv
│   │   └── churn-bigml-20.csv
│   │
│   └── processed/
│       ├── cleaned_train.csv
│       └── cleaned_test.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── model_experiments.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── recommendation_engine.py
│   └── utils.py
│
├── app/
│   ├── Home.py
│   ├── pages/
│   │   ├── Prediction.py
│   │   ├── SHAP_Analysis.py
│   │   ├── Customer_Insights.py
│   │   └── Recommendations.py
│   │
│   └── components/
│       ├── sidebar.py
│       ├── charts.py
│       └── risk_card.py
│
├── models/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── shap_explainer.pkl
│
├── reports/
│   ├── figures/
│   └── metrics.json
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes.py
│
├── tests/
│   ├── test_model.py
│   └── test_api.py
│
├── docker/
│   └── Dockerfile
│
├── requirements.txt
├── docker-compose.yml
├── README.md
└── CLAUDE.md
```

---

# Machine Learning Requirements

This project must solve a binary classification problem.

The ML pipeline should include:

* preprocessing
* feature engineering
* encoding
* scaling
* model training
* evaluation
* explainability

---

# Required Models

The project must compare:

1. Logistic Regression
2. Random Forest
3. XGBoost

The final production model will most likely be:

* XGBoost

---

# Evaluation Metrics

Do NOT rely only on accuracy.

Required metrics:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

The project should also consider class imbalance.

---

# Feature Engineering Requirements

The project must include business-oriented feature engineering.

Examples:

## tenure_group

Lifecycle stage bucketed from Account length (New … Veteran).

## service_count

Number of subscribed plans (International plan + Voice mail plan).

## total_charge

Total spend across day / evening / night / international usage.

## avg_charge_per_minute

Pricing-efficiency proxy (guards against zero usage).

## customer_value_score

Approximate customer lifetime value (total spend weighted by account length).

## service_call_risk

Customer-service-contact intensity — the strongest churn signal in this dataset.

Feature engineering should prioritize interpretability and business value.

---

# Explainability Requirements

The project MUST use SHAP explainability.

Required explainability features:

## Global Explainability

* Feature importance
* SHAP summary plots

## Local Explainability

* Individual customer explanations
* SHAP waterfall plots

The dashboard should clearly explain:

* why a customer is high risk
* which features increase churn probability
* which features reduce churn probability

---

# Risk Segmentation

The platform must classify customers into:

* Low Risk
* Medium Risk
* High Risk

Risk thresholds should be configurable.

---

# Recommendation Engine

The project must include a rule-based recommendation engine.

Purpose:
Generate actionable business retention suggestions.

Examples:

* High churn risk + high customer value
  → recommend loyalty discount

* No tech support
  → recommend premium support package

* Month-to-month contract
  → recommend annual subscription offer

Recommendations should feel realistic and business-oriented.

---

# Dashboard Requirements

Frontend must be built with Streamlit.

Required pages:

## Home

Project overview and business problem.

## Prediction

Customer input form and churn prediction.

## SHAP Analysis

Explainability visualizations.

## Customer Insights

EDA charts and analytics.

## Recommendations

Retention strategies and customer recommendations.

---

# UI/UX Expectations

Dashboard should feel modern and professional.

Preferred elements:

* KPI cards
* risk indicators
* progress bars
* clean layout
* intuitive navigation
* business-oriented visuals

The UI should resemble an analytics platform rather than a simple notebook output.

---

# API Requirements

FastAPI backend should expose endpoints such as:

```python id="bgdax7"
/predict
/explain
/recommend
/health
```

Responses should use structured JSON schemas.

---

# Docker Requirements

The entire system must be containerized.

Requirements:

* Dockerfile
* docker-compose.yml
* reproducible environment

The project should run using:

```bash id="yjlwm0"
docker compose up
```

---

# Code Quality Rules

Code must follow:

* modular design
* reusable functions
* meaningful naming
* type hints when appropriate
* readable structure
* clean separation of responsibilities

Avoid:

* giant monolithic scripts
* duplicated logic
* hardcoded values
* messy notebook-only workflows

---

# Notebook Philosophy

Notebooks are for:

* experimentation
* EDA
* prototyping

Production logic must live inside:

* src/

---

# README Requirements

README.md must include:

* project overview
* business problem
* architecture
* dataset explanation
* feature engineering
* model comparison
* evaluation metrics
* SHAP explainability
* dashboard screenshots
* Docker setup instructions
* future improvements

---

# Business-Oriented Philosophy

This project should prioritize:

* business impact
* interpretability
* maintainability
* realistic workflows

The final result should resemble:

* a startup MVP
* an internal analytics platform
* a production-oriented ML application

NOT:

* a simple Kaggle notebook
* a tutorial clone

---

# Future Improvements

Potential future enhancements:

* real-time inference
* authentication system
* database integration
* automated retraining
* LLM-powered retention suggestions
* advanced monitoring

---

# Important Development Principles

Always prioritize:

* readability over complexity
* modularity over shortcuts
* business value over unnecessary engineering
* maintainability over overengineering

The goal is to create a strong portfolio-grade ML application that demonstrates:

* end-to-end ML workflow
* explainable AI knowledge
* software engineering discipline
* business understanding
* dashboard development skills
* production mindset
