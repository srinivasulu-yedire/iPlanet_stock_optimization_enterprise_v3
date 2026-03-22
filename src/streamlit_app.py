import streamlit as st
import pandas as pd
import os
import sys
import altair as alt
import json

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.pdf_generator import generate_strategy_pdf
from data_processing import process_data
from forecasting import run_forecasting
from optimization import calculate_inventory
from genai_layer import generate_summary
from utils.logger import log

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="iPlanet Optimizer", layout="wide")
st.title("📦 iPlanet Stock Optimization Platform")

# -----------------------------
# Load UI Config
# -----------------------------
def load_ui_content():
    config_path = 'config/ui_content.json'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log(f"Error loading JSON: {e}")
    return {
        "forecasting_help": {
            "title": "🔍 Model Explanation",
            "paragraphs": ["Configuration file error. Please check UTF-8 encoding."]
        }
    }

# -----------------------------
# File Paths
# -----------------------------
SALES_PATH = "data/sales_data.xlsx"
MASTER_PATH = "data/inventory_master_data.xlsx"

# -----------------------------
# Main Logic
# -----------------------------
if os.path.exists(SALES_PATH) and os.path.exists(MASTER_PATH):

    # Load Data
    df_sales = pd.read_excel(SALES_PATH)
    df_master = pd.read_excel(MASTER_PATH)

    df_sales.columns = df_sales.columns.str.strip()
    df_master.columns = df_master.columns.str.strip()

    # Merge on ProductName
    df_merged = pd.merge(df_sales, df_master, on="ProductName", how="left")
    st.sidebar.success("✅ Files Loaded Successfully")

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    possible_store_cols = ["StoreName", "BusinessSegmentName", "Store", "Segment"]
    store_col = next((c for c in possible_store_cols if c in df_merged.columns), None)

    if store_col:
        store_list = ["ALL"] + sorted(df_merged[store_col].dropna().unique().tolist())
    else:
        store_list = ["ALL"]
        st.sidebar.warning("⚠ Store column not found.")

    product_list = ["ALL"] + sorted(df_merged["ProductName"].dropna().unique().tolist())

    selected_store = st.sidebar.selectbox("Select Store", store_list)
    selected_product = st.sidebar.selectbox("Select Product", product_list)
    horizon = st.sidebar.selectbox("Forecast Horizon (Days)", [7, 30])

    # -----------------------------
    # Data Processing (Single Product)
    # -----------------------------
    df_proc = process_data(df_merged, selected_store, selected_product)

    if not df_proc.empty:

        # -----------------------------
        # Forecasting & Tuning (Deep Tuning ON)
        # -----------------------------
        with st.spinner(f"🧠 Training and tuning machine learning models for '{selected_product}'... Please wait."):
            # We don't pass tune_models=False here, so it defaults to True for maximum accuracy
            results = run_forecasting(df_proc, horizon)
        
        model, p_metrics, x_metrics, rf_metrics, forecast, test, p_p, x_p, rf_p = results

        if model == "Insufficient Data":
            st.error(f"🛑 Insufficient sales history for '{selected_product}'.")
        else:
            
            # Global KPI Banner
            st.markdown("### 📊 High-Level Overview")
            kpi1, kpi2, kpi3 = st.columns(3)
            
            total_historical_sales = df_proc["sales"].sum()
            avg_daily_sales = df_proc["sales"].mean()
            projected_sales = forecast[model].sum()
            
            kpi1.metric("Total Historical Sales", f"{total_historical_sales:,.0f} units")
            kpi2.metric("Avg Daily Sales (Historical)", f"{avg_daily_sales:.1f} units/day")
            kpi3.metric(f"Projected Sales (Next {horizon} Days)", f"{projected_sales:,.0f} units")
            st.divider()

            # Extract base inventory data early for charts
            if selected_product != "ALL":
                prod_info = df_master[df_master["ProductName"] == selected_product]
                if not prod_info.empty:
                    prod_info = prod_info.iloc[0]
                    base_lead_time = prod_info.get("LeadTime", 7)
                    base_unit_cost = prod_info.get("UnitCost", 100)
                else:
                    base_lead_time = 7
                    base_unit_cost = 100
            else:
                base_lead_time = 7
                base_unit_cost = 100

            avg_demand = forecast[model].mean()
            std_demand = forecast[model].std()

            # The 4 Main Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📦 Optimization", "🤖 AI Insights", "📥 Bulk Export"])

            # ==========================================
            # TAB 1: FORECAST & CHARTS
            # ==========================================
            with tab1:
                st.markdown(f"### Demand Forecast: {selected_product}")
                st.caption(f"Store: {selected_store} | Winning Model: {model}")

                base_metrics = calculate_inventory(
                    demand_mean=avg_demand, demand_std=std_demand,
                    lead_time=base_lead_time, unit_cost=base_unit_cost
                )
                rop = base_metrics["Reorder Point"]

                plot_df = forecast.melt('date', var_name='Algorithm', value_name='Predicted_Sales')
                plot_df['date_label'] = plot_df['date'].dt.strftime('%d-%m-%Y')

                chart = alt.Chart(plot_df).mark_line(point=True).encode(
                    x=alt.X('date_label:O', title='Date', axis=alt.Axis(labelAngle=-90)),
                    y=alt.Y('Predicted_Sales:Q', title='Predicted Units'),
                    color=alt.Color('Algorithm:N'),
                    tooltip=['date_label', 'Algorithm', 'Predicted_Sales']
                )

                rop_line = alt.Chart(pd.DataFrame({'ROP': [rop]})).mark_rule(
                    color='red', strokeDash=[5, 5], size=2
                ).encode(y='ROP:Q')

                final_chart = (chart + rop_line).properties(height=400).interactive()
                st.altair_chart(final_chart, width="stretch")
                st.caption("🔴 Red dashed line indicates the recommended Reorder Point (based on default lead time).")

                ui_text = load_ui_content()
                help_data = ui_text.get("forecasting_help", {"title": "Help", "paragraphs": ["Content missing."]})

                with st.expander(help_data["title"]):
                    st.markdown("\n\n".join(help_data["paragraphs"]))

                st.write("#### 📊 Model Accuracy Comparison")
                metrics_df = pd.DataFrame({
                    "Algorithm": ["Prophet", "XGBoost", "Random Forest"],
                    "MAE": [p_metrics[0], x_metrics[0], rf_metrics[0]],
                    "RMSE": [p_metrics[1], x_metrics[1], rf_metrics[1]]
                })
                st.table(metrics_df.style.highlight_min(subset=["MAE"], color="lightgreen", axis=0))

            # ==========================================
            # TAB 2: INVENTORY WHAT-IF SIMULATOR
            # ==========================================
            with tab2:
                st.markdown("### 🎛️ 'What-If' Inventory Simulator")
                st.write("Adjust supply chain variables below to see how they impact your optimal order quantities and safety stock.")
                
                sim_col1, sim_col2, sim_col3 = st.columns(3)
                with sim_col1:
                    sim_lead_time = st.slider("Lead Time (Days)", min_value=1, max_value=30, value=int(base_lead_time))
                with sim_col2:
                    sim_holding_rate = st.slider("Holding Rate (%)", min_value=5, max_value=50, value=20) / 100
                with sim_col3:
                    sim_order_cost = st.slider("Ordering Cost ($)", min_value=50, max_value=2000, value=500, step=50)

                dynamic_metrics = calculate_inventory(
                    demand_mean=avg_demand, demand_std=std_demand,
                    lead_time=sim_lead_time, unit_cost=base_unit_cost,
                    ordering_cost=sim_order_cost, holding_rate=sim_holding_rate
                )

                st.divider()
                st.markdown("#### 📦 Simulated Inventory Recommendations")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("EOQ (Optimal Order)", dynamic_metrics["EOQ"])
                c2.metric("Safety Stock", dynamic_metrics["Safety Stock"])
                c3.metric("Reorder Point", dynamic_metrics["Reorder Point"])

                c4, c5 = st.columns(2)
                c4.metric("Predicted Annual Demand", dynamic_metrics["Annual Demand"])
                c5.metric("Total Annual Inv. Cost", f"${dynamic_metrics['Total Annual Cost']:,.2f}")

            # ==========================================
            # TAB 3: GENAI INTEGRATION (Now with PDF Export)
            # ==========================================
            with tab3:
                st.markdown("### 🧠 Executive Strategy Summary")
                st.write("AI-generated insights acting as your Senior Supply Chain Director.")
                
                # We use session state to remember the summary so the download button works
                if "ai_summary" not in st.session_state:
                    st.session_state.ai_summary = None

                if st.button("Generate AI Insights"):
                    with st.spinner("Analyzing ML metrics and inventory data..."):
                        
                        st.session_state.ai_summary = generate_summary(
                            product_name=selected_product,
                            selected_model=model,
                            p_metrics=p_metrics,
                            x_metrics=x_metrics,
                            inventory_metrics=base_metrics,
                            horizon=horizon
                        )
                
                # Display the summary and the PDF download button if the summary exists
                if st.session_state.ai_summary:
                    st.info(st.session_state.ai_summary)
                    
                    # Generate the PDF in memory
                    pdf_bytes = generate_strategy_pdf(
                        product_name=selected_product,
                        model_name=model,
                        inventory_metrics=base_metrics,
                        ai_summary=st.session_state.ai_summary
                    )
                    
                    st.download_button(
                        label="📄 Download Official Strategy Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"iPlanet_Strategy_{selected_product.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )

            # ==========================================
            # TAB 4: BULK EXPORT
            # ==========================================
            with tab4:
                st.markdown("### 📥 Bulk Forecast & Optimization Export")
                st.write("Run the AI forecasting and inventory optimization engine for **ALL** products and download a master report.")
                st.info("⚡ **Fast Mode Enabled:** Bulk export bypasses deep hyperparameter tuning to generate reports up to 10x faster.")

                if st.button("🚀 Run Master Bulk Analysis"):
                    
                    # Setup progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    bulk_results = []
                    
                    # Isolate actual products (ignore "ALL")
                    actual_products = [p for p in product_list if p != "ALL"]
                    total_products = len(actual_products)
                    
                    for i, prod in enumerate(actual_products):
                        status_text.text(f"Processing '{prod}' ({i+1}/{total_products})...")
                        
                        # Process data specifically for this product
                        prod_df = process_data(df_merged, selected_store, prod)
                        
                        if not prod_df.empty:
                            # 🚀 Pass tune_models=False for fast bulk processing
                            res = run_forecasting(prod_df, horizon, tune_models=False)
                            
                            m_model, m_p_metrics, m_x_metrics, m_rf_metrics, m_forecast, _, _, _, _ = res
                            
                            if m_model != "Insufficient Data":
                                m_avg_demand = m_forecast[m_model].mean()
                                m_std_demand = m_forecast[m_model].std()
                                
                                # Pull master info
                                m_prod_info = df_master[df_master["ProductName"] == prod]
                                m_lt = m_prod_info.iloc[0].get("LeadTime", 7) if not m_prod_info.empty else 7
                                m_cost = m_prod_info.iloc[0].get("UnitCost", 100) if not m_prod_info.empty else 100
                                
                                # Calculate metrics
                                inv_metrics = calculate_inventory(
                                    demand_mean=m_avg_demand, demand_std=m_std_demand, 
                                    lead_time=m_lt, unit_cost=m_cost
                                )
                                
                                # Append to master list
                                bulk_results.append({
                                    "Store": selected_store,
                                    "Product": prod,
                                    "Winning_Algorithm": m_model,
                                    f"Projected_Sales_{horizon}_Days": round(m_forecast[m_model].sum(), 1),
                                    "Avg_Daily_Demand": round(m_avg_demand, 2),
                                    "Unit_Cost": m_cost,
                                    "Lead_Time_Days": m_lt,
                                    "EOQ": inv_metrics["EOQ"],
                                    "Safety_Stock": inv_metrics["Safety Stock"],
                                    "Reorder_Point": inv_metrics["Reorder Point"]
                                })
                        
                        # Update progress bar
                        progress_bar.progress((i + 1) / total_products)
                    
                    status_text.text("✅ Bulk processing complete!")
                    
                    if bulk_results:
                        # Convert to DataFrame and show preview
                        bulk_df = pd.DataFrame(bulk_results)
                        st.dataframe(bulk_df)
                        
                        # Create CSV Download Button
                        csv_data = bulk_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Master CSV Report",
                            data=csv_data,
                            file_name=f"iplanet_master_inventory_report_{horizon}days.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error("Could not generate report. All products returned 'Insufficient Data'.")

    else:
        st.warning("⚠ No matching sales data found.")

else:
    st.error("❌ Missing files in /data folder. Please add sales_data.xlsx and inventory_master_data.xlsx.")
