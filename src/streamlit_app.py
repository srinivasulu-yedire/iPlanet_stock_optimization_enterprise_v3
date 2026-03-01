import streamlit as st
import pandas as pd
import os
import sys
import altair as alt
import json

# Helper function to load UI text from JSON with UTF-8 support
def load_ui_content():
    config_path = 'config/ui_content.json'
    if os.path.exists(config_path):
        try:
            # Adding encoding='utf-8' is the key fix here
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log(f"Error loading JSON: {e}")
            
    # Fallback content if file is missing or broken
    return {
        "forecasting_help": {
            "title": "🔍 Model Explanation",
            "paragraphs": ["Configuration file error. Please ensure config/ui_content.json is UTF-8 encoded."]
        }
    }
    
# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing import process_data
from forecasting import run_forecasting
from optimization import calculate_inventory
from utils.logger import log

st.set_page_config(page_title="iPlanet Optimizer", layout="wide")
st.title("📦 iPlanet Stock Optimization")

SALES_PATH = "data/sales_data.xlsx"
MASTER_PATH = "data/inventory_master_data.xlsx"

if os.path.exists(SALES_PATH) and os.path.exists(MASTER_PATH):
    # 1. Load and Clean Headers (Force strip spaces)
    df_sales = pd.read_excel(SALES_PATH)
    df_master = pd.read_excel(MASTER_PATH)
    df_sales.columns = df_sales.columns.str.strip()
    df_master.columns = df_master.columns.str.strip()
    
    # 2. Merge Data on ProductName
    df_merged = pd.merge(df_sales, df_master, on="ProductName", how="left")
    st.sidebar.success("✅ Files Loaded and Cleaned")

    # 3. Dynamic Sidebar Filters
    possible_cols = ["BusinessSegmentName", "Business Segment", "Segment", "Store"]
    store_col = next((c for c in possible_cols if c in df_merged.columns), None)

    if store_col:
        store_list = ["ALL"] + sorted(df_merged[store_col].dropna().unique().tolist())
    else:
        store_list = ["ALL"]
        st.sidebar.warning("Store/Segment column not found in Excel.")

    product_list = sorted(df_merged["ProductName"].dropna().unique().tolist())

    selected_store = st.sidebar.selectbox("Select Store/Segment", store_list)
    selected_product = st.sidebar.selectbox("Select Product", product_list)
    horizon = st.sidebar.selectbox("Horizon", [7, 30])

    # 4. Processing
    df_proc = process_data(df_merged, selected_store, selected_product)

    if not df_proc.empty:
        # 5. Forecasting
        res = run_forecasting(df_proc, horizon)
        model, p_metrics, x_metrics, rf_metrics, forecast, test, p_p, x_p, rf_p = res

        if model == "Insufficient Data":
            st.error(f"🛑 **Insufficient Data:** The product '{selected_product}' has fewer than 5 days of sales history.")
        else:
            # --- UI HEADER ---
            st.markdown(f"### 📈 Model Comparison Forecast: {selected_product}")
            st.caption(f"**Segment:** {selected_store}  |  **Winner:** {model}")

            # --- DUAL MODEL LINE CHART ---
            plot_df = forecast.melt('date', var_name='Algorithm', value_name='Predicted_Sales')
            plot_df['date_label'] = plot_df['date'].dt.strftime('%d-%m-%Y')
            
            chart = alt.Chart(plot_df).mark_line(point=True).encode(
                x=alt.X('date_label:O', 
                        title='Date', 
                        axis=alt.Axis(labelAngle=-90)), 
                y=alt.Y('Predicted_Sales:Q', title='Predicted Units'),
                color=alt.Color('Algorithm:N', 
                    scale=alt.Scale(domain=['Prophet', 'XGBoost', 'Random Forest'], 
                                  range=['#1f77b4', '#ff7f0e', '#2ca02c'])),
                tooltip=['date_label', 'Algorithm', 'Predicted_Sales']
            ).properties(height=400).interactive()

            st.altair_chart(chart, use_container_width=True)

            # --- DYNAMIC DETAILED EXPLANATION ---
            ui_text = load_ui_content()
            help_data = ui_text["forecasting_help"]
            with st.expander(help_data["title"]):
                st.markdown("\n\n".join(help_data["paragraphs"]))
            
            # --- ACCURACY COMPARISON TABLE ---
            st.write("#### 📊 Model Accuracy Comparison")
            metrics_df = pd.DataFrame({
                "Algorithm": ["Prophet", "XGBoost", "Random Forest"],
                "MAE (Mean Absolute Error)": [p_metrics[0], x_metrics[0], rf_metrics[0]],
                "RMSE (Root Mean Square Error)": [p_metrics[1], x_metrics[1], rf_metrics[1]]
            })
            # Highlight the best (minimum) MAE in green
            st.table(metrics_df.style.highlight_min(subset=["MAE (Mean Absolute Error)"], color="lightgreen", axis=0))

            # 6. Optimization
            prod_info = df_master[df_master["ProductName"] == selected_product].iloc[0]
            l_time = prod_info.get("LeadTime", 7)
            u_cost = prod_info.get("UnitCost", 100)
            h_cost = prod_info.get("HoldingCost", 5)

            # Use winning model data for inventory metrics
            avg_demand = forecast[model].mean()
            std_demand = forecast[model].std()
            
            eoq, ss, rop = calculate_inventory(avg_demand, std_demand, l_time, u_cost, h_cost)

            st.write("#### 📦 Final Inventory Recommendations")
            c1, c2, c3 = st.columns(3)
            c1.metric("Recommended Order Qty (EOQ)", f"{int(eoq)}")
            c2.metric("Safety Stock", f"{int(ss)}")
            c3.metric("Reorder Point", f"{int(rop)}")
    else:
        st.warning(f"No sales data found for '{selected_product}' in the selected segment.")
else:
    st.error("Missing files in /data. Please ensure sales_data.xlsx and inventory_master_data.xlsx are present.")