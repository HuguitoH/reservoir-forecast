"""
Madrid Reservoir System — Operational Dashboard
Main page: system overview + forecast + drought risk assessment.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forecast import iterative_forecast, classify_capacity

BG          = "#0d0d0b"
CARD_BG     = "#141410"
CARD_BORDER = "#222218"

with open(Path(__file__).parent / "icons" / "waves-horizontal.svg") as f:
    svg_content = f.read()

svg_content = svg_content.replace("currentColor", "#4a9eff")
svg_content = svg_content.replace("#000000", "#4a9eff")
svg_content = svg_content.replace("#000", "#4a9eff")
svg_content = svg_content.replace('fill="black"', 'fill="#4a9eff"')

st.set_page_config(
    page_title="Madrid Reservoir System",
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
    .status-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}
    .status-normal   {{ border-top: 4px solid #44aa44; }}
    .status-moderate {{ border-top: 4px solid #ffaa44; }}
    .status-severe   {{ border-top: 4px solid #ff4444; }}
    .status-neutral  {{ border-top: 4px solid #4a9eff; }}
    .status-value {{
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        line-height: 1;
        letter-spacing: -0.02em;
    }}
    .status-label {{
        font-size: 0.68rem;
        color: #666655;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 6px;
    }}
    .status-sub {{ font-size: 0.75rem; margin-top: 6px; }}
    .risk-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 6px;
    }}
    .risk-normal   {{ background: rgba(68,170,68,0.15);  color: #44aa44; }}
    .risk-moderate {{ background: rgba(255,170,68,0.15); color: #ffaa44; }}
    .risk-severe   {{ background: rgba(255,68,68,0.15);  color: #ff4444; }}
    div[data-testid="stSelectbox"] label {{
        color: #666655 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent

@st.cache_resource
def load_bundle():
    with open(BASE / "models" / "xgb_bundle.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(
        BASE / "data" / "processed" / "reservoirs_pivot.csv",
        parse_dates=["ds"],
    )
    return df.set_index("ds").asfreq("MS")

@st.cache_data
def load_eda():
    with open(BASE / "data" / "processed" / "eda_summary.json") as f:
        return json.load(f)

bundle = load_bundle()
df     = load_data()
eda    = load_eda()

DROUGHT_SEVERE   = eda["drought_thresholds"]["severe_hm3"]
DROUGHT_MODERATE = eda["drought_thresholds"]["moderate_hm3"]
DROUGHT_GOOD     = eda["drought_thresholds"]["good_hm3"]

@st.cache_data
def run_forecast(n_months: int) -> tuple[pd.DatetimeIndex, list[float]]:
    return iterative_forecast(
        series       = df["total_hm3"],
        model        = bundle["model"],
        feature_cols = bundle["feature_cols"],
        n_lags       = bundle["n_lags"],
        n_months     = n_months,
    )

def color_for(cls: str) -> str:
    return "#44aa44" if cls == "normal" else "#ffaa44" if cls == "moderate" else "#ff4444"

# Header

st.markdown("# Madrid Reservoir System")
st.markdown(
    "<p style='color:#555544; font-size:0.9rem; margin-top:-12px'>"
    "Canal de Isabel II · 13 reservoirs · 1998–2021 · XGBoost forecast"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# Status cards

last_value           = float(df["total_hm3"].iloc[-1])
last_date            = df.index[-1]
last_label, last_cls = classify_capacity(last_value, DROUGHT_SEVERE, DROUGHT_MODERATE)
hist_mean            = float(df["total_hm3"].mean())
pct_mean             = (last_value / hist_mean - 1) * 100
margin               = last_value - DROUGHT_SEVERE

col1, col2, col3, col4 = st.columns(4)

with col1:
    c = color_for(last_cls)
    st.markdown(f"""
    <div class='status-card status-{last_cls}'>
        <div class='status-value' style='color:{c}'>{last_value:,.0f} hm³</div>
        <div class='status-label'>Current capacity · {last_date.strftime('%b %Y')}</div>
        <div class='risk-badge risk-{last_cls}'>{last_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    sign = "+" if pct_mean >= 0 else ""
    sc   = "#44aa44" if pct_mean >= 0 else "#ff4444"
    st.markdown(f"""
    <div class='status-card status-neutral'>
        <div class='status-value' style='color:#e8e3d9'>{hist_mean:,.0f} hm³</div>
        <div class='status-label'>Historical mean</div>
        <div class='status-sub' style='color:{sc}'>{sign}{pct_mean:.1f}% vs mean</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    mc = "#44aa44" if margin > 100 else "#ffaa44" if margin > 0 else "#ff4444"
    st.markdown(f"""
    <div class='status-card status-neutral'>
        <div class='status-value' style='color:#e8e3d9'>{DROUGHT_SEVERE:,.0f} hm³</div>
        <div class='status-label'>Severe drought threshold</div>
        <div class='status-sub' style='color:{mc}'>+{margin:.0f} hm³ margin</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='status-card status-neutral'>
        <div class='status-value' style='color:#4a9eff'>{bundle['metrics']['accuracy']:.1f}%</div>
        <div class='status-label'>Model accuracy</div>
        <div class='status-sub' style='color:#666655'>MAPE {bundle['metrics']['mape']:.1f}% · R²={bundle['metrics']['r2']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# Forecast controls

HORIZON_OPTIONS = {
    "3 months":  3,
    "6 months":  6,
    "12 months": 12,
    "2 years":   24,
    "5 years":   60,
    "12 years":  144,
}

col_ctrl, _ = st.columns([1, 3])
with col_ctrl:
    horizon_label = st.selectbox(
        "Forecast horizon",
        options=list(HORIZON_OPTIONS.keys()),
        index=2,
    )
horizon = HORIZON_OPTIONS[horizon_label]

future_dates, future_preds = run_forecast(horizon)

# Main chart

st.markdown("### System Capacity — Historical & Forecast")

hist_start      = df.index[-1] - pd.DateOffset(years=5)
colors_forecast = [
    "#ff4444" if p < DROUGHT_SEVERE else
    "#ffaa44" if p < DROUGHT_MODERATE else
    "#44aa44"
    for p in future_preds
]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.loc[hist_start:].index,
    y=df.loc[hist_start:, "total_hm3"].values,
    mode="lines", name="Historical",
    line=dict(color="#4a9eff", width=2),
))

fig.add_trace(go.Scatter(
    x=future_dates, y=future_preds,
    mode="lines+markers", name="XGBoost forecast",
    line=dict(color="#44aa44", width=2, dash="dash"),
    marker=dict(color=colors_forecast, size=7,
                line=dict(color=BG, width=1)),
))

for y_val, color, label, pos in [
    (DROUGHT_SEVERE,   "#ff4444", f"Severe  {DROUGHT_SEVERE:.0f} hm³",    "bottom right"),
    (DROUGHT_MODERATE, "#ffaa44", f"Moderate  {DROUGHT_MODERATE:.0f} hm³", "bottom right"),
    (DROUGHT_GOOD,     "#44aa44", f"Good  {DROUGHT_GOOD:.0f} hm³",         "top right"),
]:
    fig.add_hline(y=y_val,
        line=dict(color=color, dash="dot", width=1),
        annotation_text=label,
        annotation_font=dict(color=color, size=10),
        annotation_position=pos,
    )

fig.add_vline(
    x=last_date.timestamp() * 1000,
    line=dict(color="#333322", dash="dash", width=1),
    annotation_text="Forecast start",
    annotation_font=dict(color="#555544", size=10),
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG, plot_bgcolor=BG,
    height=480, hovermode="x unified",
    legend=dict(orientation="h", font=dict(color="#666655", size=11),
                bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor=CARD_BG),
    yaxis=dict(gridcolor=CARD_BG, title="Capacity (hm³)"),
    margin=dict(t=16, b=0, r=120),
)

st.plotly_chart(fig, use_container_width=True)

# Risk summary

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### Drought Risk Assessment")

n_severe   = sum(1 for p in future_preds if p < DROUGHT_SEVERE)
n_moderate = sum(1 for p in future_preds if DROUGHT_SEVERE <= p < DROUGHT_MODERATE)
n_normal   = sum(1 for p in future_preds if p >= DROUGHT_MODERATE)
min_pred   = min(future_preds)
min_date   = future_dates[future_preds.index(min_pred)]
min_label, min_cls = classify_capacity(min_pred, DROUGHT_SEVERE, DROUGHT_MODERATE)

r1, r2, r3, r4 = st.columns(4)

for col, n, label, cls in [
    (r1, n_severe,   "Severe drought months",   "severe"),
    (r2, n_moderate, "Moderate drought months", "moderate"),
    (r3, n_normal,   "Normal / good months",    "normal"),
]:
    c = color_for(cls)
    with col:
        st.markdown(f"""
        <div class='status-card status-{cls}'>
            <div class='status-value' style='color:{c}'>{n}</div>
            <div class='status-label'>{label}</div>
            <div class='risk-badge risk-{cls}'>{n/horizon*100:.0f}% of forecast</div>
        </div>
        """, unsafe_allow_html=True)

c_min = color_for(min_cls)
with r4:
    st.markdown(f"""
    <div class='status-card status-{min_cls}'>
        <div class='status-value' style='color:{c_min}'>{min_pred:,.0f} hm³</div>
        <div class='status-label'>Forecast minimum · {min_date.strftime('%b %Y')}</div>
        <div class='risk-badge risk-{min_cls}'>{min_label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    "<p style='color:#333322; font-size:0.75rem; margin-top:24px'>"
    "XGBoost · rolling_mean_3 + lag features · "
    f"Test MAPE {bundle['metrics']['mape']:.1f}% · R²={bundle['metrics']['r2']} · "
    "Iterative forecast — uncertainty grows with horizon."
    "</p>",
    unsafe_allow_html=True,
)
