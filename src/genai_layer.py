import openai
import os
from dotenv import load_dotenv
from utils.logger import log

# Load variables from .env file
load_dotenv()

def generate_summary(selected_model, prophet_metrics, xgb_metrics):
    log("=== GENAI LAYER STARTED ===")

    # Retrieve the API key from the environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log("ERROR: OpenAI API Key not found in environment.")
        return "Summary unavailable: API Key not configured."

    openai.api_key = api_key

    prompt = f"""
    Prophet metrics (MAE, RMSE, MAPE): {prophet_metrics}
    XGBoost metrics (MAE, RMSE, MAPE): {xgb_metrics}
    Selected model based on lowest MAE: {selected_model}

    Provide a concise executive summary and an inventory strategy recommendation 
    based on these forecasting results.
    """

    try:
        # Note: Ensure you are using the openai>=1.0.0 syntax if you've updated the library
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        log("=== GENAI LAYER COMPLETED ===")
        return response.choices[0].message.content
    except Exception as e:
        log(f"GENAI ERROR: {str(e)}")
        return f"An error occurred while generating the summary: {str(e)}"