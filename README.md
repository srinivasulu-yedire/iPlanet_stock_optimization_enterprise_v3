# 📦 iPlanet Stock Optimization Platform (Enterprise v3)

## 📖 Project Overview
The **iPlanet Stock Optimization Platform** is an end-to-end, AI-powered supply chain management tool built for retail environments. It bridges the gap between historical sales data and future inventory requirements by combining advanced Machine Learning forecasting with classical mathematical optimization.

Instead of relying on static spreadsheets, this application dynamically predicts future demand, calculates ideal order quantities, and utilizes Generative AI to provide plain-English executive summaries for purchasing teams.

---

## ✨ Key Features
* **Multi-Algorithm Demand Forecasting:** Evaluates Prophet, XGBoost, and Random Forest models simultaneously.
* **Automated Hyperparameter Tuning:** Uses `RandomizedSearchCV` to dynamically find the most accurate model configurations for specific products.
* **Inventory Mathematics Engine:** Automatically calculates **Economic Order Quantity (EOQ)**, **Safety Stock**, and **Reorder Points (ROP)** based on predicted demand and user-defined service levels.
* **Interactive 'What-If' Simulator:** Allows supply chain managers to adjust Lead Times, Holding Rates, and Order Costs on the fly to simulate different supply chain disruptions.
* **GenAI Executive Insights:** Integrates with OpenAI (GPT-4) to read model metrics and generate concise, actionable purchasing strategies.
* **Lightning-Fast Bulk Export:** Processes the entire product catalog using a "Fast Mode" algorithm bypass, generating a comprehensive, downloadable CSV master report in seconds.

---

## 🛠️ Technical Architecture
The project is built with a modular, separation-of-concerns architecture:

1. **`src/streamlit_app.py` (The UI Layer):** The interactive web dashboard built with Streamlit and Altair for visualizations.
2. **`src/data_processing.py` (The ETL Layer):** Cleans raw Excel files, standardizes column names, handles missing dates, and engineers critical time-series features (lags, rolling averages, seasonality).
3. **`src/forecasting.py` (The ML Engine):** Handles train/test splitting, Time Series Cross-Validation, and predictions using Scikit-Learn, XGBoost, and Prophet.
4. **`src/optimization.py` (The Math Engine):** Applies inventory optimization formulas (EOQ, Z-scores for 95% service levels) based on ML outputs.
5. **`src/genai_layer.py` (The LLM Layer):** Prompts OpenAI's API with live metric data to return strategic insights.

---

## 🚀 Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed. You will also need an active OpenAI API Key.

### 2. Clone the Repository
```bash
git clone <your-repo-link>
cd iPlanet_stock_optimization_enterprise_v3
