import pandas as pd
import numpy as np
import os

def engineer_features(input_path: str, output_path: str):
    # Load cleaned data
    df = pd.read_csv(input_path, parse_dates=['datetime'])

    # Set datetime as index
    df.set_index('datetime', inplace=True)

    # Temporal features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month

    # Rolling means (3-hour window)
    pollutants = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
    for col in pollutants:
        if col in df.columns:
            df[f'{col}_rolling3'] = df[col].rolling(window=3, min_periods=1).mean()

    # Lag features (1 hour before)
    for col in pollutants:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)

    # Drop rows with any NaN values resulting from rolling/lag
    df.dropna(inplace=True)

    # Save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(f"✅ Features engineered and saved to: {output_path}")
    return df


if __name__ == "__main__":
    INPUT_PATH = "E:/air_quality_forecasting/data/clean_air_quality.csv"
    OUTPUT_PATH = "E:/air_quality_forecasting/data/processed_air_quality.csv"
    engineer_features(INPUT_PATH, OUTPUT_PATH)
