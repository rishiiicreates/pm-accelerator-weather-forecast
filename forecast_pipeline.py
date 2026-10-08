"""
pm accelerator weather trend forecasting pipeline
author: hrishikesh yadav
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# basic plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

OUT_DIR = "/Users/rishii/pm-accelerator-weather-forecast/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = "/Users/rishii/pm-accelerator-weather-forecast/data/Global_Weather_Repository.csv"

def run_pipeline():
    print("loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"loaded {df.shape[0]} rows and {df.shape[1]} features")
    
    # parse timestamp
    df["last_updated"] = pd.to_datetime(df["last_updated"])
    df = df.sort_values(by=["last_updated", "location_name"]).reset_index(drop=True)
    
    # calendar features
    df["day_of_year"] = df["last_updated"].dt.dayofyear
    df["day_of_week"] = df["last_updated"].dt.dayofweek
    df["month"] = df["last_updated"].dt.month
    
    # eda: temperature spread
    plt.figure(figsize=(14, 6))
    top_cities = df["location_name"].unique()[:10]
    sns.boxplot(data=df[df["location_name"].isin(top_cities)], x="location_name", y="temperature_celsius", hue="location_name", palette="mako", legend=False)
    plt.title("temperature distribution across global cities (°c)")
    plt.xlabel("city")
    plt.ylabel("temperature (°c)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_temperature_distribution.png", dpi=300)
    plt.close()

    # eda: correlation heatmap
    weather_numeric = ["temperature_celsius", "humidity", "precip_mm", "pressure_mb", "wind_kph", "cloud", "uv_index", "air_quality_PM2.5", "air_quality_Ozone"]
    corr = df[weather_numeric].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("atmospheric correlation heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_correlation_matrix.png", dpi=300)
    plt.close()

    # anomaly detection with isolation forest
    anomaly_features = ["temperature_celsius", "precip_mm", "wind_kph", "pressure_mb"]
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(df[anomaly_features])

    iso_forest = IsolationForest(contamination=0.04, random_state=42)
    df["anomaly"] = iso_forest.fit_predict(scaled_feats)
    df["is_anomaly"] = df["anomaly"] == -1
    anomaly_count = df["is_anomaly"].sum()
    print(f"found {anomaly_count} outliers using isolation forest")

    plt.figure(figsize=(14, 6))
    plt.scatter(df[~df["is_anomaly"]]["last_updated"], df[~df["is_anomaly"]]["temperature_celsius"], c="#0284c7", alpha=0.5, label="regular", s=25)
    plt.scatter(df[df["is_anomaly"]]["last_updated"], df[df["is_anomaly"]]["temperature_celsius"], c="#ef4444", edgecolors="black", label="anomaly spikes", s=60, zorder=5)
    plt.title("outlier events flagged by isolation forest")
    plt.xlabel("timeline")
    plt.ylabel("temperature (°c)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/advanced_anomaly_detection.png", dpi=300)
    plt.close()

    # air quality vs wind speed
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="wind_kph", y="air_quality_PM2.5", hue="humidity", palette="viridis", size="temperature_celsius", sizes=(20, 150), alpha=0.7)
    plt.title("wind speed dispersion vs pm2.5 concentration")
    plt.xlabel("wind speed (km/h)")
    plt.ylabel("pm2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/environmental_air_quality_impact.png", dpi=300)
    plt.close()

    # latitude vs temp trend
    lat_summary = df.groupby("country").agg({
        "temperature_celsius": "mean",
        "latitude": "first"
    }).reset_index().sort_values(by="latitude")

    plt.figure(figsize=(14, 5))
    plt.plot(lat_summary["latitude"], lat_summary["temperature_celsius"], marker="o", color="#f59e0b", linewidth=2.5)
    plt.title("mean temperature vs latitude")
    plt.xlabel("latitude")
    plt.ylabel("mean temperature (°c)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/spatial_latitudinal_analysis.png", dpi=300)
    plt.close()

    # aggregate daily means for time series
    daily_ts = df.groupby("last_updated").agg({
        "temperature_celsius": "mean",
        "humidity": "mean",
        "pressure_mb": "mean",
        "wind_kph": "mean",
        "air_quality_PM2.5": "mean"
    }).reset_index()

    # lag features
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

    split_idx = int(len(daily_ts) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = daily_ts["last_updated"].iloc[split_idx:]

    # train models
    model_ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    model_rf = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42).fit(X_train, y_train)
    model_gbm = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42).fit(X_train, y_train)
    model_hgb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=5, random_state=42).fit(X_train, y_train)

    pred_ridge = model_ridge.predict(X_test)
    pred_rf = model_rf.predict(X_test)
    pred_gbm = model_gbm.predict(X_test)
    pred_hgb = model_hgb.predict(X_test)
    pred_ensemble = (0.25 * pred_ridge) + (0.35 * pred_rf) + (0.20 * pred_gbm) + (0.20 * pred_hgb)

    models_dict = {
        "Ridge": pred_ridge,
        "Random Forest": pred_rf,
        "Gradient Boosting": pred_gbm,
        "HistGradientBoosting": pred_hgb,
        "Ensemble Blend": pred_ensemble
    }

    metrics = []
    for name, preds in models_dict.items():
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        metrics.append({
            "model": name,
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "r2": round(r2, 4),
            "mape": round(mape, 2)
        })

    metrics_df = pd.DataFrame(metrics).sort_values(by="rmse")
    print(metrics_df.to_string(index=False))

    with open("/Users/rishii/pm-accelerator-weather-forecast/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # plot forecast predictions
    plt.figure(figsize=(14, 6))
    plt.plot(dates_test, y_test, label="actual", color="black", linewidth=2.5, marker="o", markersize=4)
    plt.plot(dates_test, pred_ridge, label="ridge", linestyle="--", alpha=0.7)
    plt.plot(dates_test, pred_rf, label="random forest", linestyle="-.", alpha=0.8)
    plt.plot(dates_test, pred_ensemble, label="ensemble blend", color="#10b981", linewidth=2.5)
    plt.title("model forecast comparison")
    plt.xlabel("date")
    plt.ylabel("temperature (°c)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/multi_model_forecast_comparison.png", dpi=300)
    plt.close()

    # feature importance
    importances = model_rf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    feat_imp.plot(kind="barh", color="#6366f1")
    plt.title("top features by importance (random forest)")
    plt.xlabel("importance score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=300)
    plt.close()

    print("all figures generated in visualizations folder")

if __name__ == "__main__":
    run_pipeline()
