# streamlit_india_aqi.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

import shap
from lime.lime_tabular import LimeTabularExplainer

# ----------------------------
# Helper Functions
# ----------------------------
def create_sequences(series, window_size=10):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌍 India AQI Forecasting (2010–2024)")
st.write("Forecast pollutant averages using Random Forest + GRU with SHAP & LIME")

uploaded_file = st.file_uploader("Upload India AQI Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Columns in Dataset")
    st.write(list(df.columns))

    # Use 'last_update' as date if available
    if "last_update" in df.columns:
        df["last_update"] = pd.to_datetime(df["last_update"], errors="coerce", infer_datetime_format=True)
        df = df.dropna(subset=["last_update"])
        df = df.sort_values("last_update")
        date_col = "last_update"
        st.write("✅ Using 'last_update' as the date column")
    else:
        date_col = None
        st.warning("⚠️ No date column detected. Proceeding without time-based plots.")

    st.write("### Data Preview")
    st.dataframe(df.head())

    # Optional: focus on Tamil Nadu (comment out if too restrictive)
    if "state" in df.columns:
        if "Tamil Nadu" in df["state"].unique():
            df = df[df["state"] == "Tamil Nadu"]
        else:
            st.info("ℹ️ No Tamil Nadu rows found, using full dataset instead.")

    # Visualization: pollutant_avg trend
    if date_col and "pollutant_avg" in df.columns:
        # Aggregate daily averages
        df_daily = df.resample("D", on="last_update").mean().reset_index()

        st.write("### Pollutant Trends (Average Values)")
        fig, ax = plt.subplots()
        sns.lineplot(x="last_update", y="pollutant_avg", data=df_daily, ax=ax, label="Pollutant Avg")
        ax.set_title("Pollutant Average Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Pollutant Avg")
        fig.autofmt_xdate()
        st.pyplot(fig)

    # Correlation Heatmap
    st.write("### Correlation Heatmap of Numeric Features")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig_corr, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig_corr)

    # Distribution Plot
    if "pollutant_avg" in df.columns:
        st.write("### Distribution of Pollutant Average")
        fig_dist, ax = plt.subplots()
        sns.histplot(df["pollutant_avg"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of Pollutant Avg")
        ax.set_xlabel("Pollutant Avg")
        st.pyplot(fig_dist)

    # Target = pollutant_avg
    target = "pollutant_avg"
    if target not in df.columns:
        st.error("❌ No 'pollutant_avg' column found in dataset.")
    else:
        # Identify numeric columns automatically
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col != target]

        # Build X and y
        X = df[features].apply(pd.to_numeric, errors="coerce")
        y = df[target].apply(pd.to_numeric, errors="coerce")

        # Drop rows with NaNs in either X or y
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if len(X) == 0:
            st.warning("⚠️ No valid numeric rows found after cleaning. Try removing filters or check dataset.")
        else:
            # Random Forest
            st.write("### Random Forest Model")
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_preds = rf.predict(X)

            st.write("**Performance (Random Forest):**")
            st.write("RMSE:", np.sqrt(mean_squared_error(y, rf_preds)))
            st.write("MAE:", mean_absolute_error(y, rf_preds))
            st.write("R²:", r2_score(y, rf_preds))

            # SHAP Feature Importance
            st.write("### SHAP Feature Importance")
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)

            fig_shap = plt.figure()
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig_shap)

            # GRU Model
            st.write("### GRU Model")
            series = y.values
            X_seq, y_seq = create_sequences(series, window_size=10)
            X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

            model = Sequential([
                GRU(64, activation="relu", input_shape=(X_seq.shape[1], 1)),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_seq, y_seq, epochs=10, verbose=0)

            gru_preds = model.predict(X_seq)

            st.write("**Performance (GRU):**")
            st.write("RMSE:", np.sqrt(mean_squared_error(y_seq, gru_preds)))
            st.write("MAE:", mean_absolute_error(y_seq, gru_preds))
            st.write("R²:", r2_score(y_seq, gru_preds))

            # LIME Explanation
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
