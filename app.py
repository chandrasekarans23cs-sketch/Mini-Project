# streamlit_aqi.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

import shap
from lime.lime_tabular import LimeTabularExplainer

# ----------------------------
# Helper Functions
# ----------------------------
def create_sequences(series, window_size=5):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌍 Air Quality Index (AQI) Forecasting")
st.write("Predict AQI using Random Forest + LSTM/GRU with SHAP & LIME interpretability")

uploaded_file = st.file_uploader("Upload CSV with pollutants + environmental features", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Visualization
    st.write("### Pollutant Trends")
    fig, ax = plt.subplots()
    sns.lineplot(x="date", y="PM2.5", data=df, ax=ax, label="PM2.5")
    sns.lineplot(x="date", y="PM10", data=df, ax=ax, label="PM10")
    sns.lineplot(x="date", y="NO2", data=df, ax=ax, label="NO2")
    sns.lineplot(x="date", y="SO2", data=df, ax=ax, label="SO2")
    sns.lineplot(x="date", y="O3", data=df, ax=ax, label="O3")
    plt.legend()
    st.pyplot(fig)

    # Features & Target
    features = ["PM10","NO2","SO2","O3","CO","wind_speed","humidity","traffic"]
    target = "PM2.5"
    X = df[features].dropna()
    y = df[target].loc[X.index]

    # Random Forest
    st.write("### Random Forest Model")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_preds = rf.predict(X)

    st.write("**Performance (Random Forest):**")
    st.write("RMSE:", np.sqrt(mean_squared_error(y, rf_preds)))
    st.write("MAE:", mean_absolute_error(y, rf_preds))
    st.write("R²:", r2_score(y, rf_preds))

    # SHAP
    st.write("### SHAP Feature Importance")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(plt.gcf())

    # LSTM/GRU
    st.write("### LSTM/GRU Model")
    series = y.values
    X_seq, y_seq = create_sequences(series, window_size=10)
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

    model = Sequential([
        GRU(64, activation="relu", input_shape=(X_seq.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq, y_seq, epochs=10, verbose=0)

    lstm_preds = model.predict(X_seq)

    st.write("**Performance (GRU):**")
    st.write("RMSE:", np.sqrt(mean_squared_error(y_seq, lstm_preds)))
    st.write("MAE:", mean_absolute_error(y_seq, lstm_preds))
    st.write("R²:", r2_score(y_seq, lstm_preds))

    # LIME
    st.write("### LIME Explanation (Random Forest)")
    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X),
        feature_names=X.columns,
        mode="regression"
    )
    exp = lime_explainer.explain_instance(X.iloc[0].values, rf.predict)
    st.write(exp.as_list())

    # Health Advisory
    st.write("### Health Advisory")
    latest_aqi = rf_preds[-1]
    if latest_aqi < 50:
        st.success("Air Quality is Good ✅")
    elif latest_aqi < 100:
        st.info("Air Quality is Moderate ⚠️")
    else:
        st.error("Air Quality is Poor ❌ - Limit outdoor activity")
