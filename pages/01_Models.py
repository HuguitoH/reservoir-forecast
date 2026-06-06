import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BG       = "#0d0d0b"
CARD_BG  = "#141410"
CARD_BORDER = "#222218"

with open(Path(__file__).parent.parent / "icons" / "chart-spline.svg") as f:
    svg_content = f.read()
svg_content = svg_content.replace("currentColor", "#4a9eff")

st.set_page_config(
    page_title="Model Comparison",
    page_icon=svg_content,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'DM Mono', monospace; }}
    h1, h2, h3, h4 {{ font-family: 'DM Serif Display', serif !important; letter-spacing: -0.02em; }}
    .stApp {{ background-color: {BG}; color: #e8e3d9; }}
    .section-divider {{ border: none; border-top: 1px solid {CARD_BORDER}; margin: 24px 0; }}
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .metric-value {{
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        line-height: 1;
        letter-spacing: -0.02em;
    }}
    .metric-label {{
        font-size: 0.68rem;
        color: #666655;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 6px;
    }}
    .winner {{ border-top: 4px solid #44aa44; }}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent.parent

@st.cache_data
def load_data():
    df = pd.read_csv(
        BASE / "data" / "processed" / "reservoirs_pivot.csv",
        parse_dates=["ds"],
    )
    return df.set_index("ds").asfreq("MS")

@st.cache_data
def load_comparison():
    return pd.read_csv(BASE / "models" / "model_comparison.csv")

@st.cache_resource
def load_xgb():
    with open(BASE / "models" / "xgb_bundle.pkl", "rb") as f:
        return pickle.load(f)

df         = load_data()
comparison = load_comparison()
xgb_bundle = load_xgb()

TRAIN_END  = "2016-12-01"
TEST_START = "2017-01-01"
train      = df.loc[:TRAIN_END, "total_hm3"]
test       = df.loc[TEST_START:, "total_hm3"]

st.markdown("# Model Comparison")
st.markdown(
    "<p style='color:#555544; font-size:0.9rem; margin-top:-12px'>"
    "Four models evaluated on test period Jan 2017 – Mar 2021 · "
    "XGBoost selected for production"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# Winner metrics
best = comparison[comparison["model"] == "XGBoost"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
for col, val, label, color in [
    (c1, f"{best['accuracy']:.1f}%",  "XGBoost accuracy",  "#44aa44"),
    (c2, f"{best['mape']:.1f}%",      "MAPE",              "#4a9eff"),
    (c3, f"{best['r2']:.4f}",         "R²",                "#4a9eff"),
    (c4, f"{best['rmse']:.1f} hm³",   "RMSE",              "#4a9eff"),
]:
    with col:
        cls = "winner" if label == "XGBoost accuracy" else ""
        st.markdown(f"""
        <div class='metric-card {cls}'>
            <div class='metric-value' style='color:{color}'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# Comparison table
st.markdown("### Results — Test Set 2017–2021")


MODEL_STYLES = {
    "Prophet": ("#ff4444", "dot"),
    "SARIMAX": ("#ffaa44", "dash"),
    "LSTM":    ("#4a9eff", "dash"),
    "XGBoost": ("#44aa44", "dash"),
}

styled = comparison.copy()
st.dataframe(
    styled.sort_values("mape")[["model", "r2", "mae", "rmse", "mape", "accuracy"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "model":    st.column_config.TextColumn("Model"),
        "r2":       st.column_config.NumberColumn("R²",       format="%.4f"),
        "mae":      st.column_config.NumberColumn("MAE (hm³)", format="%.1f"),
        "rmse":     st.column_config.NumberColumn("RMSE (hm³)", format="%.1f"),
        "mape":     st.column_config.NumberColumn("MAPE (%)",  format="%.2f"),
        "accuracy": st.column_config.NumberColumn("Accuracy (%)", format="%.2f"),
    }
)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### Forecast vs Actual — Test Period")

# Load model predictions from saved bundles
@st.cache_data
@st.cache_data
def load_all_predictions():
    from src.features import build_xgb_features
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
    from tensorflow.keras import Input

    preds = {}

    # Prophet
    with open(BASE / "models" / "prophet_bundle.pkl", "rb") as f:
        prophet_bundle = pickle.load(f)
    model_prophet = prophet_bundle["model"]
    future        = model_prophet.make_future_dataframe(periods=len(test), freq="MS")
    forecast      = model_prophet.predict(future)
    preds["Prophet"] = forecast[forecast["ds"] >= TEST_START]["yhat"].values

    # SARIMAX
    with open(BASE / "models" / "sarimax_bundle.pkl", "rb") as f:
        sarimax_bundle = pickle.load(f)
    preds["SARIMAX"] = sarimax_bundle["model"].get_forecast(
        steps=len(test)
    ).predicted_mean.values

    # LSTM
    with open(BASE / "models" / "lstm_bundle.pkl", "rb") as f:
        lstm_bundle = pickle.load(f)

    scaler_target = lstm_bundle["scaler_target"]
    scaler_feats  = lstm_bundle["scaler_feats"]
    feature_cols  = lstm_bundle["feature_cols"]
    window_size   = lstm_bundle["window_size"]

    model_lstm = Sequential([
        Input(shape=(window_size, len(feature_cols))),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(1),
    ])
    model_lstm.set_weights(lstm_bundle["model_weights"])

    df_lstm = df[["total_hm3"]].copy()
    df_lstm["month_num"] = df.index.month
    df_lstm["month_sin"] = np.sin(2 * np.pi * df_lstm["month_num"] / 12)
    df_lstm["month_cos"] = np.cos(2 * np.pi * df_lstm["month_num"] / 12)
    df_lstm["total_scaled"] = scaler_target.transform(df_lstm[["total_hm3"]])
    for lag in range(1, 4):
        df_lstm[f"lag_{lag}"] = df_lstm["total_scaled"].shift(lag)
    df_lstm = df_lstm.dropna()
    df_lstm[feature_cols] = scaler_feats.transform(df_lstm[feature_cols])

    features  = df_lstm[feature_cols].values
    targets   = df_lstm["total_scaled"].values
    dates_all = df_lstm.index[window_size:]
    X_all     = np.array([features[i:i+window_size] for i in range(len(targets)-window_size)])
    test_mask = dates_all >= TEST_START
    X_test    = X_all[test_mask]

    y_pred_scaled = model_lstm(X_test, training=False).numpy()
    preds["LSTM"] = scaler_target.inverse_transform(y_pred_scaled).flatten()

    # XGBoost
    model_xgb    = xgb_bundle["model"]
    feature_cols_xgb = xgb_bundle["feature_cols"]
    n_lags       = xgb_bundle["n_lags"]
    df_xgb       = build_xgb_features(df["total_hm3"], n_lags=n_lags)
    X_test_xgb   = df_xgb[df_xgb.index >= TEST_START][feature_cols_xgb]
    preds["XGBoost"] = model_xgb.predict(X_test_xgb)

    return preds

with st.spinner("Loading model predictions..."):
    all_preds = load_all_predictions()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=test.index, y=test.values,
    mode="lines", name="Actual",
    line=dict(color="#ffffff", width=2),
))

for model_name, (color, dash) in MODEL_STYLES.items():
    if model_name in all_preds:
        mape = comparison[comparison["model"] == model_name]["mape"].values[0]
        fig.add_trace(go.Scatter(
            x=test.index,
            y=all_preds[model_name],
            mode="lines",
            name=f"{model_name} (MAPE {mape:.1f}%)",
            line=dict(color=color, width=1.5, dash=dash),
        ))

fig.add_vline(
    x=pd.Timestamp(TEST_START).timestamp() * 1000,
    line=dict(color="#333322", dash="dash", width=1),
)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    height=420,
    hovermode="x unified",
    legend=dict(orientation="h", font=dict(color="#666655", size=11),
                bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor=CARD_BG),
    yaxis=dict(gridcolor=CARD_BG, title="Capacity (hm³)"),
    margin=dict(t=16, b=0),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### XGBoost — Feature Importance")

importance = xgb_bundle["feature_importance"]
imp_series = pd.Series(importance).sort_values(ascending=True).tail(10)

fig2 = go.Figure(go.Bar(
    x=imp_series.values,
    y=imp_series.index,
    orientation="h",
    marker_color="#44aa44",
    opacity=0.85,
    text=[f"{v:.3f}" for v in imp_series.values],
    textposition="outside",
    textfont=dict(color="#666655", size=10),
))
fig2.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    height=360,
    margin=dict(l=0, r=60, t=0, b=0),
    xaxis=dict(gridcolor=CARD_BG, showticklabels=False),
    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    "<p style='color:#333322; font-size:0.75rem'>"
    "rolling_mean_3 and lag_1 account for ~70% of predictive power — "
    "the system has strong short-term momentum. "
    "XGBoost outperforms LSTM with only 279 data points because "
    "well-engineered features beat deep learning on small datasets."
    "</p>",
    unsafe_allow_html=True,
)
