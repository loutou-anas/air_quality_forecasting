import os
from src.preprocessing.clean_data import load_and_clean_data
from src.preprocessing.feature_engineering import engineer_features

# === Paths ===
RAW_DATA = "E:/air_quality_forecasting/data/AirQualityUCI.csv"
CLEANED_DATA = "E:/air_quality_forecasting/data/clean_air_quality.csv"
PROCESSED_DATA = "E:/air_quality_forecasting/data/processed_air_quality.csv"
MODELS_DIR = "models"

# === Step 1: Clean Data ===
print("🔄 Step 1: Cleaning data...")
df_cleaned = load_and_clean_data(RAW_DATA, CLEANED_DATA)

# === Step 2: Feature Engineering ===
print("🔄 Step 2: Feature engineering...")
df_processed = engineer_features(CLEANED_DATA, PROCESSED_DATA)

# === Step 3: Train Models ===
print("🔄 Step 3: Training models...")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd

df = pd.read_csv(PROCESSED_DATA, parse_dates=["datetime"])
df.set_index("datetime", inplace=True)

targets = ['CO(GT)', 'NO2(GT)', 'NOx(GT)', 'C6H6(GT)']
os.makedirs(MODELS_DIR, exist_ok=True)
results = {}

for target in targets:
    print(f"📌 Training model for: {target}")
    features = df.drop(columns=targets).select_dtypes(include="number").columns
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    results[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    model_path = os.path.join(MODELS_DIR, f"{target.replace('(GT)', '').lower()}_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Saved model to {model_path}")

# === Summary ===
print("\n📊 Model Performance Summary:")
for t, m in results.items():
    print(f"{t}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R2={m['R2']:.3f}")
