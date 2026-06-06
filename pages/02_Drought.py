import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BG          = "#0d0d0b"
CARD_BG     = "#141410"
CARD_BORDER = "#222218"

with open(Path(__file__).parent.parent / "icons" / "droplets.svg") as f:
    svg_content = f.read()
svg_content = svg_content.replace("currentColor", "#4a9eff")

st.set_page_config(
    page_title="Drought Analysis",
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
    .stat-card {{
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
    .stat-value {{
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        line-height: 1;
        letter-spacing: -0.02em;
    }}
    .stat-label {{
        font-size: 0.68rem;
        color: #666655;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 6px;
    }}
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
def load_eda():
    with open(BASE / "data" / "processed" / "eda_summary.json") as f:
        return json.load(f)

df  = load_data()
eda = load_eda()

DROUGHT_SEVERE   = eda["drought_thresholds"]["severe_hm3"]
DROUGHT_MODERATE = eda["drought_thresholds"]["moderate_hm3"]
DROUGHT_GOOD     = eda["drought_thresholds"]["good_hm3"]

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def classify_numeric(value: float) -> int:
    if value < DROUGHT_SEVERE:   return 0  # severe
    if value < DROUGHT_MODERATE: return 1  # moderate
    return 2                                # normal

st.markdown("# Drought Analysis")
st.markdown(
    "<p style='color:#555544; font-size:0.9rem; margin-top:-12px'>"
    "Historical drought classification · Canal de Isabel II · 1998–2021"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# Summary stats
n_severe   = (df["total_hm3"] < DROUGHT_SEVERE).sum()
n_moderate = ((df["total_hm3"] >= DROUGHT_SEVERE) & (df["total_hm3"] < DROUGHT_MODERATE)).sum()
n_normal   = (df["total_hm3"] >= DROUGHT_MODERATE).sum()
n_total    = len(df)

worst_month = df["total_hm3"].idxmin()
best_month  = df["total_hm3"].idxmax()

c1, c2, c3, c4 = st.columns(4)
for col, val, label, color in [
    (c1, f"{n_severe} months",   f"Severe drought · {n_severe/n_total*100:.0f}%",   "#ff4444"),
    (c2, f"{n_moderate} months", f"Moderate drought · {n_moderate/n_total*100:.0f}%", "#ffaa44"),
    (c3, f"{n_normal} months",   f"Normal / good · {n_normal/n_total*100:.0f}%",     "#44aa44"),
    (c4, worst_month.strftime("%b %Y"), f"Worst month · {df['total_hm3'].min():,.0f} hm³", "#ff4444"),
]:
    with col:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value' style='color:{color}'>{val}</div>
            <div class='stat-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### Drought Heatmap — Month × Year")

years  = sorted(df.index.year.unique())
matrix = []
for year in years:
    row = []
    for month in range(1, 13):
        mask = (df.index.year == year) & (df.index.month == month)
        if mask.any():
            val = df.loc[mask, "total_hm3"].values[0]
            row.append(val)
        else:
            row.append(None)
    matrix.append(row)

z_matrix    = [[v if v is not None else float("nan") for v in row] for row in matrix]
text_matrix = [
    [f"{v:,.0f} hm³" if v is not None else "" for v in row]
    for row in matrix
]

fig = go.Figure(go.Heatmap(
    z=z_matrix,
    x=MONTH_LABELS,
    y=[str(y) for y in years],
    text=text_matrix,
    hovertemplate="%{y} %{x}<br>%{text}<extra></extra>",
    colorscale=[
        [0.0,  "#ff4444"],
        [0.35, "#ffaa44"],
        [0.55, "#44aa44"],
        [1.0,  "#4a9eff"],
    ],
    showscale=True,
    colorbar=dict(
        tickvals=[DROUGHT_SEVERE, DROUGHT_MODERATE, DROUGHT_GOOD],
        ticktext=[
            f"Severe {DROUGHT_SEVERE:.0f}",
            f"Moderate {DROUGHT_MODERATE:.0f}",
            f"Good {DROUGHT_GOOD:.0f}",
        ],
        tickfont=dict(color="#666655", size=10),
        outlinewidth=0,
    ),
    zmin=300, zmax=950,
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    height=600,
    margin=dict(t=0, b=0),
    xaxis=dict(side="top"),
    yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### Drought Months per Year")

severe_per_year   = []
moderate_per_year = []
normal_per_year   = []

for year in years:
    year_data = df[df.index.year == year]["total_hm3"]
    severe_per_year.append((year_data < DROUGHT_SEVERE).sum())
    moderate_per_year.append(((year_data >= DROUGHT_SEVERE) & (year_data < DROUGHT_MODERATE)).sum())
    normal_per_year.append((year_data >= DROUGHT_MODERATE).sum())

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=years, y=severe_per_year,
    name="Severe", marker_color="#ff4444", opacity=0.85,
))
fig2.add_trace(go.Bar(
    x=years, y=moderate_per_year,
    name="Moderate", marker_color="#ffaa44", opacity=0.85,
))
fig2.add_trace(go.Bar(
    x=years, y=normal_per_year,
    name="Normal", marker_color="#44aa44", opacity=0.85,
))

fig2.update_layout(
    barmode="stack",
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    height=350,
    hovermode="x unified",
    legend=dict(orientation="h", font=dict(color="#666655", size=11),
                bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor=CARD_BG, dtick=1),
    yaxis=dict(gridcolor=CARD_BG, title="Months", dtick=3),
    margin=dict(t=0, b=0),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    "<p style='color:#333322; font-size:0.75rem; margin-top:8px'>"
    f"Severe drought: capacity below {DROUGHT_SEVERE:.0f} hm³ (p10 historical) · "
    f"Moderate: below {DROUGHT_MODERATE:.0f} hm³ (p25) · "
    f"Good: above {DROUGHT_GOOD:.0f} hm³ (p75). "
    "Worst period: 2005–2006 drought with 16 consecutive months below moderate threshold."
    "</p>",
    unsafe_allow_html=True,
)
