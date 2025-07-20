import pandas as pd
import numpy as np
import os

def load_and_clean_data(input_path: str, output_path: str):
    # Load with appropriate separator and encoding
    df = pd.read_csv(input_path, sep=';', decimal=',', encoding='latin1')

    # Drop irrelevant columns
    df.drop(columns=["Unnamed: 15", "Unnamed: 16"], inplace=True, errors='ignore')

    # Combine Date and Time into datetime
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S", errors='coerce')
    df.drop(columns=["Date", "Time"], inplace=True)

    # Move datetime column to the front
    cols = ["datetime"] + [col for col in df.columns if col != "datetime"]
    df = df[cols]

    # Replace -200 values with np.nan (indicates missing data)
    df.replace(-200, np.nan, inplace=True)

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned data saved to: {output_path}")
    return df


if __name__ == "__main__":
    INPUT_PATH = "E:/air_quality_forecasting/data/AirQualityUCI.csv"
    OUTPUT_PATH = "E:/air_quality_forecasting/data/clean_air_quality.csv"
    load_and_clean_data(INPUT_PATH, OUTPUT_PATH)
