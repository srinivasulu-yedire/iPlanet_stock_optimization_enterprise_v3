import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.logger import log

def run_forecasting(df, horizon):
    log("=== FORECASTING STARTED ===")
    
    # Safety check for minimal rows
    if len(df) < 5:
        return "Insufficient Data", [0,0], [0,0], [0,0], pd.DataFrame(), df, [], [], []

    # 80/20 Train-Test Split
    split = int(len(df) * 0.8)
    train, test = df[:split], df[split:]

    # --- 1. Prophet Model ---
    m_p = Prophet(daily_seasonality=True, yearly_seasonality=False)
    m_p.fit(train.rename(columns={"date": "ds", "sales": "y"}))
    p_pred = m_p.predict(test.rename(columns={"date": "ds"}))["yhat"].values
    p_mae = mean_absolute_error(test["sales"], p_pred)
    p_rmse = np.sqrt(mean_squared_error(test["sales"], p_pred))

    # --- 2. XGBoost Model ---
    m_x = XGBRegressor()
    m_x.fit(train[["day", "month", "year"]], train["sales"])
    x_pred = m_x.predict(test[["day", "month", "year"]])
    x_mae = mean_absolute_error(test["sales"], x_pred)
    x_rmse = np.sqrt(mean_squared_error(test["sales"], x_pred))

    # --- 3. Random Forest Model ---
    m_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    m_rf.fit(train[["day", "month", "year"]], train["sales"])
    rf_pred = m_rf.predict(test[["day", "month", "year"]])
    rf_mae = mean_absolute_error(test["sales"], rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(test["sales"], rf_pred))

    # Identify the winner based on MAE
    model_scores = {
        "Prophet": p_mae,
        "XGBoost": x_mae,
        "Random Forest": rf_mae
    }
    selected = min(model_scores, key=model_scores.get)
    log(f"Winning Model: {selected}")

    # --- 4. Generate Future Forecasts for ALL ---
    f_dates = pd.date_range(df["date"].max(), periods=horizon + 1)[1:]
    f_df = pd.DataFrame({"day": f_dates.day, "month": f_dates.month, "year": f_dates.year})

    # Prophet Future
    m_p_full = Prophet(daily_seasonality=True).fit(df.rename(columns={"date": "ds", "sales": "y"}))
    p_future = m_p_full.predict(m_p_full.make_future_dataframe(periods=horizon)).tail(horizon)
    
    # XGBoost Future
    x_future_preds = m_x.predict(f_df)

    # Random Forest Future
    rf_future_preds = m_rf.predict(f_df)

    # Combine into a single Forecast DataFrame
    forecast_df = pd.DataFrame({
        "date": f_dates,
        "Prophet": p_future["yhat"].values,
        "XGBoost": x_future_preds,
        "Random Forest": rf_future_preds
    })

    # Return structure expanded to include RF metrics
    return selected, [p_mae, p_rmse], [x_mae, x_rmse], [rf_mae, rf_rmse], forecast_df, test, p_pred, x_pred, rf_pred