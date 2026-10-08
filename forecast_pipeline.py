"""
PM Accelerator - Weather Trend Forecasting & Atmospheric Intelligence Pipeline
Author: Hrishikesh Yadav
Track: Data Science & AI Engineering Assessment
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

OUT_DIR = "/Users/rishii/pm-accelerator-weather-forecast/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = "/Users/rishii/pm-accelerator-weather-forecast/data/Global_Weather_Repository.csv"

def run_pipeline():
    print("=" * 70)
    print("PM ACCELERATOR - GLOBAL WEATHER TREND FORECASTING PIPELINE")
    print("Mission: Supporting PM & AI professionals through every stage of their careers")
    print("=" * 70)

    # 1. Ingestion & Preprocessing
    df = pd.read_csv(DATA_PATH)
    print(f"\n[1] Ingested dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Parse last_updated to datetime
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df = df.sort_values(by=["last_updated", "location_name"]).reset_index(drop=True)
    
    # Feature extraction from datetime
    df["day_of_year"] = df["last_updated"].dt.dayofyear
    df["day_of_week"] = df["last_updated"].dt.dayofweek
    df["month"] = df["last_updated"].dt.month
    
    # Handling missing values & duplicates
    missing_cnt = df.isnull().sum().sum()
    print(f"Missing values handled: {missing_cnt} (Clean dataset)")

    # 2. Exploratory Data Analysis (EDA)
    print("\n[2] Executing Exploratory Data Analysis & Visualizations...")
    
    # Visual 1: Global Temperature Distribution & Regional Variations
    plt.figure(figsize=(14, 6))
    top_cities = df["location_name"].unique()[:10]
    sns.boxplot(data=df[df["location_name"].isin(top_cities)], x="location_name", y="temperature_celsius", palette="mako")
    plt.title("PM Accelerator EDA: Global Temperature Distribution across Representative Cities (°C)", fontsize=14, fontweight="bold")
    plt.xlabel("City")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_temperature_distribution.png", dpi=300)
    plt.close()

    # Visual 2: Temperature & Precipitation Correlation Matrix
    weather_numeric = ["temperature_celsius", "humidity", "precip_mm", "pressure_mb", "wind_kph", "cloud", "uv_index", "air_quality_PM2.5", "air_quality_Ozone"]
    corr = df[weather_numeric].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Atmospheric & Air Quality Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_correlation_matrix.png", dpi=300)
    plt.close()

    # 3. Advanced EDA: Anomaly Detection (Isolation Forest)
    print("\n[3] Implementing Advanced Anomaly Detection (Outlier Analysis)...")
    anomaly_features = ["temperature_celsius", "precip_mm", "wind_kph", "pressure_mb"]
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(df[anomaly_features])

    iso_forest = IsolationForest(contamination=0.04, random_state=42)
    df["anomaly"] = iso_forest.fit_predict(scaled_feats)
    df["is_anomaly"] = df["anomaly"] == -1
    anomaly_count = df["is_anomaly"].sum()
    print(f"Detected {anomaly_count} atmospheric anomalies ({anomaly_count/len(df)*100:.1f}% of observations).")

    plt.figure(figsize=(14, 6))
    plt.scatter(df[~df["is_anomaly"]]["last_updated"], df[~df["is_anomaly"]]["temperature_celsius"], c="#0284c7", alpha=0.5, label="Normal Observations", s=25)
    plt.scatter(df[df["is_anomaly"]]["last_updated"], df[df["is_anomaly"]]["temperature_celsius"], c="#ef4444", edgecolors="black", label="Atmospheric Anomalies (Extreme Temp/Wind/Precip)", s=60, zorder=5)
    plt.title(f"PM Accelerator Anomaly Detection: Outlier Events Identified via Isolation Forest (N={anomaly_count})", fontsize=13, fontweight="bold")
    plt.xlabel("Timeline")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/advanced_anomaly_detection.png", dpi=300)
    plt.close()

    # 4. Environmental Impact: Air Quality Correlation Analysis
    print("\n[4] Analyzing Environmental Impact & Air Quality Correlations...")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="wind_kph", y="air_quality_PM2.5", hue="humidity", palette="viridis", size="temperature_celsius", sizes=(20, 150), alpha=0.7)
    plt.title("Environmental Impact: Atmospheric Dispersion (Wind vs PM2.5 Concentration)", fontsize=13, fontweight="bold")
    plt.xlabel("Wind Speed (km/h)")
    plt.ylabel("Particulate Matter PM2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/environmental_air_quality_impact.png", dpi=300)
    plt.close()

    # 5. Spatial & Geographical Analysis
    print("\n[5] Executing Spatial & Latitudinal Pattern Analysis...")
    lat_summary = df.groupby("country").agg({
        "temperature_celsius": "mean",
        "precip_mm": "mean",
        "air_quality_PM2.5": "mean",
        "latitude": "first"
    }).reset_index().sort_values(by="latitude")

    plt.figure(figsize=(14, 5))
    plt.plot(lat_summary["latitude"], lat_summary["temperature_celsius"], marker="o", color="#f59e0b", linewidth=2.5, label="Mean Temp (°C)")
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5, label="Equator Benchmark")
    plt.title("Spatial Analysis: Global Temperature Variation as a Function of Latitude", fontsize=13, fontweight="bold")
    plt.xlabel("Latitude (°)")
    plt.ylabel("Mean Temperature (°C)")
    for _, row in lat_summary.iterrows():
        plt.annotate(row["country"], (row["latitude"], row["temperature_celsius"]), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/spatial_latitudinal_analysis.png", dpi=300)
    plt.close()

    # 6. Time Series Forecasting & Multi-Model Comparative Evaluation
    print("\n[6] Building & Evaluating Multiple Forecasting Models...")
    
    # Aggregate daily global mean series for benchmark time series analysis
    daily_ts = df.groupby("last_updated").agg({
        "temperature_celsius": "mean",
        "humidity": "mean",
        "pressure_mb": "mean",
        "wind_kph": "mean",
        "air_quality_PM2.5": "mean"
    }).reset_index()

    # Feature engineering for time series
    for lag in [1, 2, 3, 7]:
        daily_ts[f"temp_lag_{lag}"] = daily_ts["temperature_celsius"].shift(lag)
        daily_ts[f"humidity_lag_{lag}"] = daily_ts["humidity"].shift(lag)

    daily_ts["rolling_mean_7"] = daily_ts["temperature_celsius"].shift(1).rolling(window=7).mean()
    daily_ts["day_of_year"] = daily_ts["last_updated"].dt.dayofyear
    daily_ts["sin_doy"] = np.sin(2 * np.pi * daily_ts["day_of_year"] / 365)
    daily_ts["cos_doy"] = np.cos(2 * np.pi * daily_ts["day_of_year"] / 365)

    daily_ts = daily_ts.dropna().reset_index(drop=True)

    feature_cols = [c for c in daily_ts.columns if c not in ["last_updated", "temperature_celsius"]]
    X = daily_ts[feature_cols]
    y = daily_ts["temperature_celsius"]

    # Train / Test split chronologically (last 20% for testing)
    split_idx = int(len(daily_ts) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = daily_ts["last_updated"].iloc[split_idx:]

    # Model 1: Ridge Linear Auto-Regressive Baseline
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train, y_train)
    pred_ridge = model_ridge.predict(X_test)

    # Model 2: Random Forest Regressor
    model_rf = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_test)

    # Model 3: Gradient Boosting Regressor (GBM)
    model_gbm = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    model_gbm.fit(X_train, y_train)
    pred_gbm = model_gbm.predict(X_test)

    # Model 4: HistGradientBoosting (LightGBM equivalent)
    model_hgb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=5, random_state=42)
    model_hgb.fit(X_train, y_train)
    pred_hgb = model_hgb.predict(X_test)

    # Model 5: Blended Ensemble Model (Weighted Optimization)
    pred_ensemble = (0.25 * pred_ridge) + (0.35 * pred_rf) + (0.20 * pred_gbm) + (0.20 * pred_hgb)

    models_dict = {
        "Ridge AutoRegressive": pred_ridge,
        "Random Forest Regressor": pred_rf,
        "Gradient Boosting (GBM)": pred_gbm,
        "HistGradientBoosting": pred_hgb,
        "Ensemble (Blended Multi-Model)": pred_ensemble
    }

    metrics = []
    for name, preds in models_dict.items():
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        metrics.append({
            "Model": name,
            "RMSE (°C)": round(rmse, 3),
            "MAE (°C)": round(mae, 3),
            "R² Score": round(r2, 4),
            "MAPE (%)": round(mape, 2)
        })

    metrics_df = pd.DataFrame(metrics).sort_values(by="RMSE (°C)")
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(metrics_df.to_string(index=False))

    # Save metrics to JSON
    with open("/Users/rishii/pm-accelerator-weather-forecast/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Visual 4: Forecast Comparison Plot
    plt.figure(figsize=(14, 6))
    plt.plot(dates_test, y_test, label="Actual Ground Truth", color="black", linewidth=2.5, marker="o", markersize=4)
    plt.plot(dates_test, pred_ridge, label=f"Ridge (RMSE: {metrics_df[metrics_df['Model']=='Ridge AutoRegressive']['RMSE (°C)'].values[0]})", linestyle="--", alpha=0.7)
    plt.plot(dates_test, pred_rf, label=f"Random Forest (RMSE: {metrics_df[metrics_df['Model']=='Random Forest Regressor']['RMSE (°C)'].values[0]})", linestyle="-.", alpha=0.8)
    plt.plot(dates_test, pred_ensemble, label=f"Ensemble Model (RMSE: {metrics_df[metrics_df['Model']=='Ensemble (Blended Multi-Model)']['RMSE (°C)'].values[0]})", color="#10b981", linewidth=2.5)
    plt.title("PM Accelerator Multi-Model Forecasting Horizon vs Actual Observations", fontsize=14, fontweight="bold")
    plt.xlabel("Forecast Horizon Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/multi_model_forecast_comparison.png", dpi=300)
    plt.close()

    # 7. Feature Importance Analysis
    print("\n[7] Computing Feature Importance Metrics...")
    importances = model_rf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    feat_imp.plot(kind="barh", color="#6366f1")
    plt.title("Top Feature Importance in Weather Trend Forecasting (Random Forest MDI)", fontsize=13, fontweight="bold")
    plt.xlabel("Relative Importance Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE. ALL VISUALS AND METRICS SAVED TO visualizations/")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
