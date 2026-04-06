# process.py — Air Quality Pipeline
# Preprocessing + Feature Engineering + ML Prediction
# Saves results to Azure Blob Storage

import pandas as pd
import numpy as np
import os
import io

from azure.storage.blob import BlobServiceClient

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. AZURE CONNECTION
# ─────────────────────────────────────────────
import os
print("AZURE_CONN_STR =", os.getenv("AZURE_CONN_STR"))
conn_str = os.getenv("AZURE_CONN_STR")
client = BlobServiceClient.from_connection_string(conn_str)

def download_blob(container: str, blob_name: str) -> pd.DataFrame:
    blob = client.get_blob_client(container, blob_name)
    data = blob.download_blob().readall()
    return pd.read_csv(io.BytesIO(data))

def upload_blob(df: pd.DataFrame, container: str, blob_name: str):
    blob = client.get_blob_client(container, blob_name)
    blob.upload_blob(df.to_csv(index=False), overwrite=True)
    print(f"  ✓ Uploaded → {container}/{blob_name}")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  AIR QUALITY PIPELINE — STARTING")
print("=" * 55)

df = download_blob("project-container", "input/globalAirQualityINPUT.csv")
print(f"\n[1] Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("\n[2] Preprocessing...")

# Parse timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"]      = df["timestamp"].dt.hour
df["day"]       = df["timestamp"].dt.day
df["dayofweek"] = df["timestamp"].dt.dayofweek   # 0=Mon … 6=Sun
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# Drop duplicates
before = len(df)
df = df.drop_duplicates()
print(f"  Dropped {before - len(df)} duplicate rows")

# Drop rows where all pollutants are NaN (fully empty)
pollutants = ["pm25", "pm10", "no2", "so2", "o3", "co", "aqi"]
df = df.dropna(subset=pollutants, how="all")

# Impute remaining NaNs with per-city median
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df.groupby("city")[numeric_cols].transform(
    lambda x: x.fillna(x.median())
)
# Fallback: global median for cities with no data at all
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print(f"  After cleaning: {df.shape[0]} rows")

# Outlier clipping (IQR × 3 per column — keep real extreme events)
for col in ["pm25", "pm10", "no2", "aqi"]:
    q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=q1, upper=q3)

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3] Feature engineering...")

# Composite pollution index (weighted)
df["pollution_index"] = (
    0.35 * df["pm25"]
    + 0.25 * df["pm10"]
    + 0.15 * df["no2"]
    + 0.10 * df["so2"]
    + 0.10 * df["o3"]
    + 0.05 * df["co"]
).round(3)

# AQI category (US EPA standard)
def aqi_category(aqi):
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"

df["aqi_category"] = df["aqi"].apply(aqi_category)

# Rolling 3-hour average AQI per city (lagged signal for ML)
df = df.sort_values(["city", "timestamp"])
df["aqi_rolling3h"] = (
    df.groupby("city")["aqi"]
    .transform(lambda x: x.rolling(3, min_periods=1).mean())
    .round(2)
)

# Heat index (simplified) — feels-like temperature
df["heat_index"] = (
    df["temperature"] + 0.33 * df["humidity"] / 100 * 6.105
    * np.exp(17.27 * df["temperature"] / (237.7 + df["temperature"]))
    - 0.70 * df["wind_speed"] - 4.0
).round(2)

print(f"  Created features: pollution_index, aqi_category, aqi_rolling3h, heat_index")

# ─────────────────────────────────────────────
# 4. AGGREGATED SUMMARY PER COUNTRY
# ─────────────────────────────────────────────
print("\n[4] Country-level aggregation...")

country_agg = df.groupby("country").agg(
    avg_aqi        = ("aqi", "mean"),
    max_aqi        = ("aqi", "max"),
    avg_pm25       = ("pm25", "mean"),
    avg_pm10       = ("pm10", "mean"),
    avg_no2        = ("no2", "mean"),
    avg_so2        = ("so2", "mean"),
    avg_temp       = ("temperature", "mean"),
    avg_humidity   = ("humidity", "mean"),
    avg_wind       = ("wind_speed", "mean"),
    pollution_idx  = ("pollution_index", "mean"),
    records        = ("aqi", "count"),
).round(2).reset_index()

# Rank countries by pollution
country_agg["aqi_rank"] = country_agg["avg_aqi"].rank(ascending=False).astype(int)
print(f"  Aggregated {len(country_agg)} countries")

# ─────────────────────────────────────────────
# 5. UNSUPERVISED CLUSTERING (K-Means)
# ─────────────────────────────────────────────
print("\n[5] K-Means clustering on countries...")

cluster_features = ["avg_aqi", "avg_pm25", "avg_temp", "avg_humidity", "avg_wind"]
X_cluster = country_agg[cluster_features].fillna(0)
scaler_c = StandardScaler()
X_scaled = scaler_c.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
country_agg["cluster"] = kmeans.fit_predict(X_scaled)

# Label clusters by their average AQI
cluster_means = country_agg.groupby("cluster")["avg_aqi"].mean()
sorted_clusters = cluster_means.sort_values().index.tolist()
label_map = {sorted_clusters[0]: "Low Pollution",
             sorted_clusters[1]: "Moderate Pollution",
             sorted_clusters[2]: "High Pollution"}
country_agg["cluster_label"] = country_agg["cluster"].map(label_map)
print(f"  Cluster distribution:\n{country_agg['cluster_label'].value_counts().to_string()}")

# ─────────────────────────────────────────────
# 6. MACHINE LEARNING — AQI PREDICTION
# ─────────────────────────────────────────────
print("\n[6] Training ML models...")

FEATURES = ["pm25", "pm10", "no2", "so2", "o3", "co",
            "temperature", "humidity", "wind_speed",
            "pollution_index", "aqi_rolling3h",
            "hour", "dayofweek", "is_weekend"]
TARGET = "aqi"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Model 1: Linear Regression (baseline)
lr = LinearRegression()
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)
lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_r2   = r2_score(y_test, lr_pred)
print(f"  Linear Regression → MAE={lr_mae:.2f}, R²={lr_r2:.4f}")

# ── Model 2: Random Forest
rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_r2   = r2_score(y_test, rf_pred)
print(f"  Random Forest     → MAE={rf_mae:.2f}, R²={rf_r2:.4f}")

# ── Model 3: Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_mae  = mean_absolute_error(y_test, gb_pred)
gb_r2   = r2_score(y_test, gb_pred)
print(f"  Gradient Boosting → MAE={gb_mae:.2f}, R²={gb_r2:.4f}")

# ── Best model wins
scores = {"LinearRegression": lr_mae, "RandomForest": rf_mae, "GradientBoosting": gb_mae}
best_name = min(scores, key=scores.get)
best_pred = {"LinearRegression": lr_pred, "RandomForest": rf_pred,
             "GradientBoosting": gb_pred}[best_name]
print(f"\n  🏆 Best model: {best_name} (MAE={scores[best_name]:.2f})")

# Predict on full dataset with best model
if best_name == "LinearRegression":
    df["predicted_aqi"] = lr.predict(scaler.transform(df[FEATURES]))
elif best_name == "RandomForest":
    df["predicted_aqi"] = rf.predict(df[FEATURES])
else:
    df["predicted_aqi"] = gb.predict(df[FEATURES])

df["predicted_aqi"] = df["predicted_aqi"].round(2)
df["prediction_error"] = (df["predicted_aqi"] - df["aqi"]).round(2)

# Feature importance (from RF — always available)
fi = pd.DataFrame({
    "feature": FEATURES,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False).round(4)

# Test set comparison
test_comparison = X_test.copy()
test_comparison["actual_aqi"]    = y_test.values
test_comparison["predicted_aqi"] = rf_pred.round(2)
test_comparison["error"]         = (rf_pred - y_test.values).round(2)

# Model performance summary
model_scores = pd.DataFrame({
    "model": ["LinearRegression", "RandomForest", "GradientBoosting"],
    "mae":   [round(lr_mae, 3), round(rf_mae, 3), round(gb_mae, 3)],
    "r2":    [round(lr_r2, 4),  round(rf_r2, 4),  round(gb_r2, 4)],
})

# ─────────────────────────────────────────────
# 7. HOURLY TREND PER COUNTRY
# ─────────────────────────────────────────────
hourly = df.groupby(["country", "hour"]).agg(
    avg_aqi=("aqi", "mean"),
    avg_pm25=("pm25", "mean"),
).round(2).reset_index()

# ─────────────────────────────────────────────
# 8. UPLOAD ALL RESULTS TO AZURE
# ─────────────────────────────────────────────
print("\n[7] Uploading results to Azure...")

upload_blob(df,               "output", "predictions.csv")
upload_blob(country_agg,      "output", "country_summary.csv")
upload_blob(fi,               "output", "feature_importance.csv")
upload_blob(model_scores,     "output", "model_scores.csv")
upload_blob(test_comparison,  "output", "test_comparison.csv")
upload_blob(hourly,           "output", "hourly_trends.csv")

print("\n" + "=" * 55)
print(f"  ✅ PIPELINE COMPLETE")
print(f"     Rows processed : {len(df):,}")
print(f"     Countries      : {df['country'].nunique()}")
print(f"     Cities         : {df['city'].nunique()}")
print(f"     Best model     : {best_name} (MAE={scores[best_name]:.2f})")
print("=" * 55)
