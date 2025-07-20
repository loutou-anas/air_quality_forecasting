# 🌫️ Air Quality Impact & Forecasting

An end-to-end data science and analytics project that forecasts air pollutant levels based on the UCI Air Quality dataset. This project demonstrates data engineering, time-series modeling, and dashboard deployment using **Python**, **Streamlit**, and **Scikit-learn**.

---

## 🚀 Features

- 📊 **Visual EDA:** Histograms, time-series plots, and pollutant comparisons
- 📉 **Forecasting (Optional Extension):** Ready for integration with ML models (ARIMA, LSTM, etc.)
- 📌 **Streamlit Dashboard:** Clean, responsive UI using Python and Streamlit
- 🗂️ Modular project structure for scalability

---

## 📁 Project Structure

```
air_quality_forecasting/
├── data/                      # Raw & processed data
├── notebooks/                 # EDA & modeling notebooks
├── src/                       # Python modules (utils, preprocessing, models)
├── models/                    # Saved models (Pickle/Joblib)
├── dashboard/                 # Streamlit dashboard
├── requirements.txt           # Environment dependencies
├── README.md                  # Project overview & setup
└── run.py                     # Main entry point
```

---

## 📦 Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/loutou-anas/air_quality_forecasting.git
   cd air_quality_forecasting
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset**  
   Place [`AirQualityUCI.csv`](https://archive.ics.uci.edu/dataset/360/air+quality) into the `data/` directory.

4. **Run full pipeline**  
   ```bash
   python run.py
   ```

5. **Launch dashboard**  
   ```bash
   streamlit run dashboard/app.py
   ```

---

## 📊 Features

- Data cleaning & transformation
- Feature engineering (lags, rolling means, temporal)
- Forecast models for:
  - `CO(GT)`
  - `NO2(GT)`
  - `NOx(GT)`
  - `C6H6(GT)`
- Streamlit dashboard for visualization
- Modular and reproducible structure

---

## 📚 Dataset Source

- UCI Machine Learning Repository – [Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)

---

# 🌫️ Air Quality Impact & Forecasting

🚀 **Live Demo:** [View Streamlit Dashboard](https://airqualityforecasting-loutou-anas.streamlit.app)

An end-to-end data science and analytics project that forecasts air pollutant levels...

---

## 📌 Author

Made by Anas Loutou.
