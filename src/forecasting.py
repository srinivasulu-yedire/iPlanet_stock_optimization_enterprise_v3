import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from utils.logger import log


def run_forecasting(df, horizon, tune_models=True):
    """
    Runs the forecasting models. 
    If tune_models=True, performs hyperparameter grid search (slower, more accurate).
    If tune_models=False, uses robust defaults (extremely fast for bulk exports).
    """

    log(f"=== FORECASTING STARTED (Tuning: {tune_models}) ===")

    # 1️⃣ Safety Check
    if len(df) < 15:
        log("Insufficient data for robust forecasting.")
        return "Insufficient Data", [0, 0], [0, 0], [0, 0], pd.DataFrame(), df, [], [], []

    # 2️⃣ Lag Features
    df = df.sort_values("date")
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["rolling_7"] = df["sales"].shift(1).rolling(7).mean()
    df = df.dropna()

    # 3️⃣ Train-Test Split (80/20)
    split = int(len(df) * 0.8)
    train, test = df[:split], df[split:]

    feature_cols = [
        "day", "month", "year", "day_of_week", "week_of_year",
        "is_weekend", "lag_1", "lag_7", "rolling_7", "price"
    ]

    # =============================
    # 🔮 1. Prophet Model (Always Fast)
    # =============================
    m_p = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
    m_p.add_regressor("price")

    prophet_train = train.rename(columns={"date": "ds", "sales": "y"})
    m_p.fit(prophet_train[["ds", "y", "price"]])

    prophet_test = test.rename(columns={"date": "ds"})
    p_pred = m_p.predict(prophet_test[["ds", "price"]])["yhat"].values

    p_mae = mean_absolute_error(test["sales"], p_pred)
    p_rmse = np.sqrt(mean_squared_error(test["sales"], p_pred))

    # =============================
    # 🚀 2 & 3. XGBoost and Random Forest Models
    # =============================
    if tune_models:
        log("Deep Tuning Enabled: Running RandomizedSearchCV...")
        tscv = TimeSeriesSplit(n_splits=3)

        # XGBoost Tuning
        xgb_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}
        xgb_tuned = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_param_grid, n_iter=5, cv=tscv, scoring='neg_mean_absolute_error', random_state=42)
        xgb_tuned.fit(train[feature_cols], train["sales"])
        m_x = xgb_tuned.best_estimator_

        # Random Forest Tuning
        rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}
        rf_tuned = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, n_iter=5, cv=tscv, scoring='neg_mean_absolute_error', random_state=42)
        rf_tuned.fit(train[feature_cols], train["sales"])
        m_rf = rf_tuned.best_estimator_

    else:
        log("Fast Mode Enabled: Using robust baseline parameters.")
        # XGBoost Default
        m_x = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
        m_x.fit(train[feature_cols], train["sales"])

        # Random Forest Default
        m_rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        m_rf.fit(train[feature_cols], train["sales"])

    # Predictions & Metrics for ML models
    x_pred = m_x.predict(test[feature_cols])
    x_mae = mean_absolute_error(test["sales"], x_pred)
    x_rmse = np.sqrt(mean_squared_error(test["sales"], x_pred))

    rf_pred = m_rf.predict(test[feature_cols])
    rf_mae = mean_absolute_error(test["sales"], rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(test["sales"], rf_pred))

    # -----------------------------
    # 4️⃣ Select Best Model
    # -----------------------------
    model_scores = {"Prophet": p_mae, "XGBoost": x_mae, "Random Forest": rf_mae}
    selected = min(model_scores, key=model_scores.get)
    log(f"Winning Model: {selected}")

    # =============================
    # 5️⃣ Future Forecast
    # =============================
    last_date = df["date"].max()
    future_dates = pd.date_range(last_date, periods=horizon + 1)[1:]

    future_df = pd.DataFrame({"date": future_dates})
    future_df["day"] = future_df["date"].dt.day
    future_df["month"] = future_df["date"].dt.month
    future_df["year"] = future_df["date"].dt.year
    future_df["day_of_week"] = future_df["date"].dt.dayofweek
    future_df["week_of_year"] = future_df["date"].dt.isocalendar().week.astype(int)
    future_df["is_weekend"] = future_df["day_of_week"].isin([5, 6]).astype(int)

    future_df["lag_1"] = df["sales"].iloc[-1]
    future_df["lag_7"] = df["sales"].iloc[-7] if len(df) >= 7 else df["sales"].iloc[-1]
    future_df["rolling_7"] = df["sales"].tail(7).mean()
    future_df["price"] = df["price"].iloc[-1]

    # Prophet Future
    m_p_full = Prophet(daily_seasonality=True)
    m_p_full.add_regressor("price")
    full_prophet = df.rename(columns={"date": "ds", "sales": "y"})
    m_p_full.fit(full_prophet[["ds", "y", "price"]])
    prophet_future = future_df.rename(columns={"date": "ds"})
    p_future = m_p_full.predict(prophet_future[["ds", "price"]])["yhat"].values

    # XGBoost and RF Futures
    x_future = m_x.predict(future_df[feature_cols])
    rf_future = m_rf.predict(future_df[feature_cols])

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "Prophet": p_future,
        "XGBoost": x_future,
        "Random Forest": rf_future
    })

    return (selected, [p_mae, p_rmse], [x_mae, x_rmse], [rf_mae, rf_rmse], forecast_df, test, p_pred, x_pred, rf_pred)
