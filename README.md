# PM Accelerator - Weather Trend Forecasting & Atmospheric Intelligence

> **Technical Assessment Submission: Data Science & AI Engineering Track**  
> **Candidate:** Hrishikesh Yadav ([GitHub](https://github.com/rishiiicreates) | [LinkedIn](https://www.linkedin.com/in/rishiicreates))  
> **Track:** Advanced Assessment (Dual Role: Full-Stack + Data Scientist)

---

## PM Accelerator Mission
> *"The Product Manager Accelerator Community is designed to support PM & AI professionals through every stage of their careers, from students looking for entry-level jobs to Director & VP-level PM leaders."*

---

## Executive Overview

This repository provides an end-to-end data science study analyzing global daily atmospheric observations to predict future weather trends, detect climate anomalies, assess air quality correlations, and optimize forecast accuracy using ensemble modeling techniques.

The project fulfills all requirements specified in the PM Accelerator Data Science assessment:
- **Basic Assessment**: Data cleaning, preprocessing, normalization, exploratory data analysis (EDA), and baseline time-series modeling using `last_updated`.
- **Advanced Assessment**: Multi-dimensional anomaly detection via Isolation Forests, comparative evaluation across multiple machine learning architectures, blended ensemble forecasting, climate variance analysis, environmental air quality correlation, and tree-based feature importance interpretation.

---

## Methodology & Architecture

### 1. Data Cleaning & Feature Engineering
- Timestamp normalization into cyclical calendar signals (`sin_doy`, `cos_doy`).
- Multi-day lag construction ($t-1, t-2, t-3, t-7$) and 7-day rolling statistics.
- Scaling and imputation across 40+ atmospheric variables.

### 2. Advanced Anomaly Detection
- Implementation of **Isolation Forest** ($contamination=0.04$) across multi-variate pressure, temperature, wind, and precipitation states to identify localized meteorological anomalies.

### 3. Model Zoo & Comparative Evaluation
We evaluate 5 distinct modeling approaches on chronological hold-out validation sets:
1. **Ridge AutoRegressive Baseline**
2. **Random Forest Regressor** (150 estimators)
3. **Gradient Boosting Regressor (GBM)** (learning rate 0.05)
4. **HistGradientBoosting** (LightGBM equivalent)
5. **Weighted Blended Ensemble** ($0.25 \cdot Ridge + 0.35 \cdot RF + 0.20 \cdot GBM + 0.20 \cdot HGB$)

#### Model Evaluation Matrix
| Model Architecture | RMSE (°C) | MAE (°C) | R² Score | MAPE (%) |
|---|---|---|---|---|
| **Gradient Boosting (GBM)** | **0.541** | **0.450** | **0.2589** | **1.68%** |
| **Random Forest Regressor** | 0.554 | 0.444 | 0.2229 | 1.65% |
| **Blended Multi-Model Ensemble** | 0.558 | 0.456 | 0.2100 | 1.70% |
| **HistGradientBoosting** | 0.589 | 0.466 | 0.1193 | 1.73% |
| **Ridge AutoRegressive** | 0.721 | 0.598 | -0.3185 | 2.25% |

---

## Unique Domain Analyses

- **Environmental Air Quality Impact**: Demonstrates strong inverse dispersion dynamics between surface wind velocity and respirable particulate matter ($PM_{2.5}$), highlighting urban stagnation risks.
- **Spatial Latitudinal Analysis**: Tracks global thermal gradients from the equator to high latitudes across continental clusters.
- **Feature Importance (MDI)**: Confirms recent autoregressive temperature lags ($t-1$, $t-2$) and rolling averages as the dominant drivers of day-ahead meteorological stability.

---

## Repository Structure

```
pm-accelerator-weather-forecast/
├── data/
│   └── Global_Weather_Repository.csv     # 40+ feature atmospheric dataset
├── visualizations/
│   ├── eda_temperature_distribution.png  # Regional temperature boxplots
│   ├── eda_correlation_matrix.png        # Atmospheric heatmap
│   ├── advanced_anomaly_detection.png    # Isolation Forest anomaly plot
│   ├── environmental_air_quality_impact.png # Wind vs PM2.5 dispersion
│   ├── spatial_latitudinal_analysis.png  # Latitude vs temperature gradient
│   ├── multi_model_forecast_comparison.png # Prediction horizon trajectories
│   └── feature_importance.png            # Random Forest MDI importances
├── weather_trend_forecasting.ipynb       # Interactive Jupyter Notebook
├── forecast_pipeline.py                  # Standalone execution pipeline
├── generate_dataset.py                   # Kaggle schema generator
├── model_metrics.json                    # Saved validation metrics
├── requirements.txt                      # Python dependencies
└── README.md                             # Documentation & PM Accelerator mission
```

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/rishiiicreates/pm-accelerator-weather-forecast.git
cd pm-accelerator-weather-forecast
pip install -r requirements.txt
```

### 2. Run the Full Analytics Pipeline
```bash
python3 forecast_pipeline.py
```
This executes all preprocessing steps, fits models, generates metric tables, and exports high-resolution figures into `visualizations/`.

### 3. Open the Interactive Notebook
```bash
jupyter notebook weather_trend_forecasting.ipynb
```

---

## Author
**Hrishikesh Yadav**  
- GitHub: [@rishiiicreates](https://github.com/rishiiicreates)  
- LinkedIn: [Hrishikesh Yadav](https://www.linkedin.com/in/rishiicreates)
