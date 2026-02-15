# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# =========================
# Helper Functions
# =========================
def create_sequences(series, window_size=5):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# =========================
# Streamlit UI
# =========================
st.title("📈 Time-Series Forecasting System")
st.write("Random Forest + LSTM with SHAP & LIME interpretability")

# Upload data
uploaded_file = st.file_uploader("Upload CSV with 'date' and 'value' columns", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    # Feature engineering
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["lag1"] = df["value"].shift(1)
    df = df.dropna()

    X = df[["dayofweek", "month", "lag1"]]
    y = df["value"]

    # Visualization
    st.write("### Time-Series Plot")
    fig, ax = plt.subplots()
    sns.lineplot(x="date", y="value", data=df, ax=ax)
    st.pyplot(fig)

    # Random Forest
    st.write("### Random Forest Model")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_preds = rf.predict(X)

    st.write("**Performance Metrics (Random Forest):**")
    st.write("RMSE:", np.sqrt(mean_squared_error(y, rf_preds)))
    st.write("MAE:", mean_absolute_error(y, rf_preds))
    st.write("R²:", r2_score(y, rf_preds))

    # SHAP
    st.write("### SHAP Feature Importance (Random Forest)")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    fig_shap = shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig_shap)

    # LSTM
    st.write("### LSTM Model")
    series = df["value"].values
    X_seq, y_seq = create_sequences(series, window_size=5)
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(X_seq.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq, y_seq, epochs=5, verbose=0)

    lstm_preds = model.predict(X_seq)

    st.write("**Performance Metrics (LSTM):**")
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
