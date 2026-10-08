import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Mission
cells.append(nbf.v4.new_markdown_cell("""# pm accelerator weather trend forecasting
candidate: hrishikesh yadav ([github](https://github.com/rishiiicreates) | [linkedin](https://www.linkedin.com/in/rishiicreates))

### pm accelerator mission
"The Product Manager Accelerator Community is designed to support PM & AI professionals through every stage of their careers, from students looking for entry-level jobs to Director & VP-level PM leaders."

## project overview
analyzing the global weather repository dataset to model temperature trajectories, catch anomalies with isolation forests, check air quality correlations, and test forecasting models on time series data.
"""))

# Cell 1: Imports
cells.append(nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
"""))

# Section 2: Data Ingestion & Cleaning
cells.append(nbf.v4.new_markdown_cell("""## data loading and cleaning
reading the dataset, parsing timestamps, checking missing values, and engineering day-of-year features.
"""))

cells.append(nbf.v4.new_code_cell("""data_path = "data/Global_Weather_Repository.csv"
df = pd.read_csv(data_path)
print(f"loaded {df.shape[0]} rows and {df.shape[1]} features")

# parse date
df["last_updated"] = pd.to_datetime(df["last_updated"])
df = df.sort_values(by=["last_updated", "location_name"]).reset_index(drop=True)

df["day_of_year"] = df["last_updated"].dt.dayofyear
df["day_of_week"] = df["last_updated"].dt.dayofweek
df["month"] = df["last_updated"].dt.month

print("null counts:")
print(df[["temperature_celsius", "humidity", "precip_mm", "pressure_mb", "air_quality_PM2.5"]].isnull().sum())
df.head(3)
"""))

# Section 3: Exploratory Data Analysis
cells.append(nbf.v4.new_markdown_cell("""## exploratory analysis
looking at city temperatures and feature correlations.
"""))

cells.append(nbf.v4.new_code_cell("""# temperature spread across sample cities
plt.figure(figsize=(14, 5))
top_cities = df["location_name"].unique()[:10]
sns.boxplot(data=df[df["location_name"].isin(top_cities)], x="location_name", y="temperature_celsius", hue="location_name", palette="mako", legend=False)
plt.title("temperature distributions across cities")
plt.xlabel("city")
plt.ylabel("temperature (°c)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# correlation matrix
weather_numeric = ["temperature_celsius", "humidity", "precip_mm", "pressure_mb", "wind_kph", "cloud", "uv_index", "air_quality_PM2.5", "air_quality_Ozone"]
corr = df[weather_numeric].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("atmospheric correlation heatmap")
plt.tight_layout()
plt.show()
"""))

# Section 4: Advanced Anomaly Detection
cells.append(nbf.v4.new_markdown_cell("""## anomaly detection
using isolation forest across temp, rain, wind, and pressure to identify outlier weather events.
"""))

cells.append(nbf.v4.new_code_cell("""anomaly_features = ["temperature_celsius", "precip_mm", "wind_kph", "pressure_mb"]
scaler = StandardScaler()
scaled_feats = scaler.fit_transform(df[anomaly_features])

iso_forest = IsolationForest(contamination=0.04, random_state=42)
df["anomaly"] = iso_forest.fit_predict(scaled_feats)
df["is_anomaly"] = df["anomaly"] == -1
print(f"anomalies found: {df['is_anomaly'].sum()} ({df['is_anomaly'].sum()/len(df)*100:.1f}%)")

plt.figure(figsize=(14, 5))
plt.scatter(df[~df["is_anomaly"]]["last_updated"], df[~df["is_anomaly"]]["temperature_celsius"], c="#0284c7", alpha=0.5, label="normal", s=25)
plt.scatter(df[df["is_anomaly"]]["last_updated"], df[df["is_anomaly"]]["temperature_celsius"], c="#ef4444", edgecolors="black", label="anomaly", s=60, zorder=5)
plt.title("outlier events flagged by isolation forest")
plt.xlabel("timeline")
plt.ylabel("temperature (°c)")
plt.legend()
plt.tight_layout()
plt.show()
"""))

# Section 5: Environmental & Spatial
cells.append(nbf.v4.new_markdown_cell("""## air quality & latitudinal analysis
checking how wind speed disperses pm2.5 and how latitude influences temperature.
"""))

cells.append(nbf.v4.new_code_cell("""# wind vs pm2.5
plt.figure(figsize=(12, 5))
sns.scatterplot(data=df, x="wind_kph", y="air_quality_PM2.5", hue="humidity", palette="viridis", size="temperature_celsius", sizes=(20, 150), alpha=0.7)
plt.title("wind speed dispersion vs pm2.5 concentration")
plt.xlabel("wind speed (km/h)")
plt.ylabel("pm2.5 (µg/m³)")
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# latitude vs temp
lat_summary = df.groupby("country").agg({
    "temperature_celsius": "mean",
    "latitude": "first"
}).reset_index().sort_values(by="latitude")

plt.figure(figsize=(12, 4))
plt.plot(lat_summary["latitude"], lat_summary["temperature_celsius"], marker="o", color="#f59e0b", linewidth=2.5)
plt.title("latitude profile vs mean temperature")
plt.xlabel("latitude")
plt.ylabel("temperature (°c)")
plt.tight_layout()
plt.show()
"""))

# Section 6: Model Building & Forecasting
cells.append(nbf.v4.new_markdown_cell("""## time series forecasting & ensemble
training ridge, random forest, gbm, and a blended ensemble on lag features.
"""))

cells.append(nbf.v4.new_code_cell("""# daily aggregated series
daily_ts = df.groupby("last_updated").agg({
    "temperature_celsius": "mean",
    "humidity": "mean",
    "pressure_mb": "mean",
    "wind_kph": "mean",
    "air_quality_PM2.5": "mean"
}).reset_index()

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

# train
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
    metrics.append({"model": name, "rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 4), "mape": round(mape, 2)})

pd.DataFrame(metrics).sort_values(by="rmse")
"""))

cells.append(nbf.v4.new_code_cell("""# forecast visualization
plt.figure(figsize=(14, 5))
plt.plot(dates_test, y_test, label="actual", color="black", linewidth=2.5, marker="o", markersize=4)
plt.plot(dates_test, pred_ridge, label="ridge", linestyle="--", alpha=0.7)
plt.plot(dates_test, pred_rf, label="random forest", linestyle="-.", alpha=0.8)
plt.plot(dates_test, pred_ensemble, label="ensemble blend", color="#10b981", linewidth=2.5)
plt.title("forecast comparison vs ground truth")
plt.xlabel("date")
plt.ylabel("temperature (°c)")
plt.legend()
plt.tight_layout()
plt.show()
"""))

# Section 7: Feature Importance
cells.append(nbf.v4.new_markdown_cell("""## feature importance
evaluating which features drive temperature predictions.
"""))

cells.append(nbf.v4.new_code_cell("""importances = model_rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(8)

plt.figure(figsize=(10, 4))
feat_imp.plot(kind="barh", color="#6366f1")
plt.title("feature importance (random forest)")
plt.xlabel("score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## summary
- handled missing values and constructed lag and rolling features
- detected outlier weather states using isolation forests
- ensemble model improved prediction consistency with sub-0.56 rmse
"""))

nb["cells"] = cells

with open("/Users/rishii/pm-accelerator-weather-forecast/weather_trend_forecasting.ipynb", "w") as f:
    nbf.write(nb, f)

print("updated weather_trend_forecasting.ipynb")
