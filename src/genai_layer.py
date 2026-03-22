import os
from dotenv import load_dotenv
from openai import OpenAI
from utils.logger import log

# Load variables from .env file
load_dotenv()

def generate_summary(product_name, selected_model, p_metrics, x_metrics, inventory_metrics, horizon):
    log("=== ADVANCED GENAI LAYER STARTED ===")

    # Retrieve the API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("ERROR: OpenAI API Key not found in environment.")
        return "⚠️ **Summary unavailable:** API Key not configured in .env file."

    # Initialize the modern OpenAI client
    client = OpenAI(api_key=api_key)

    # 1. System Persona (The Expert)
    system_prompt = """
    You are the Senior Supply Chain Director for iPlanet (an Apple Premium Reseller). 
    You are analyzing machine learning demand forecasts and inventory metrics to advise the purchasing team.
    Your tone must be authoritative, highly analytical, concise, and business-focused.
    Do not explain basic supply chain concepts (e.g., do not define what EOQ means). Focus strictly on strategy and risk.
    """

    # 2. Contextual Data Injection
    user_prompt = f"""
    **Target Product:** {product_name}
    **Forecast Horizon:** {horizon} days
    
    **Machine Learning Performance:**
    - Winning Algorithm: {selected_model}
    - Prophet Error (MAE / RMSE): {p_metrics[0]:.2f} / {p_metrics[1]:.2f}
    - XGBoost Error (MAE / RMSE): {x_metrics[0]:.2f} / {x_metrics[1]:.2f}
    
    **Calculated Inventory Directives:**
    - Economic Order Quantity (EOQ): {inventory_metrics.get('EOQ')} units
    - Safety Stock: {inventory_metrics.get('Safety Stock')} units
    - Reorder Point (ROP): {inventory_metrics.get('Reorder Point')} units
    - Total Projected Annual Cost: ${inventory_metrics.get('Total Annual Cost'):,.2f}

    Based strictly on this data, provide a structured briefing using EXACTLY these three markdown headings:
    
    ### 📊 Executive Summary
    (Provide 1-2 sentences summarizing the demand stability and your overall recommendation for this product.)
    
    ### ⚠️ Risk Assessment
    (Analyze the model error and safety stock. If error is high, warn about volatility and stockouts. If EOQ/Costs are high, warn about capital tie-up.)
    
    ### 🎯 Actionable Directives
    (Provide 3 punchy, specific bullet points instructing the purchasing team exactly what to do with the ROP and EOQ right now.)
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4", 
            temperature=0.2, # Low temperature for analytical precision, less "hallucination"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        log("=== GENAI LAYER COMPLETED ===")
        return response.choices[0].message.content
        
    except Exception as e:
        log(f"GENAI ERROR: {str(e)}")
        return f"⚠️ **An error occurred while generating the AI summary:**\n`{str(e)}`"
