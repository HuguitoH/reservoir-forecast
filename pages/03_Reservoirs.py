from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

BG          = "#0d0d0b"
CARD_BG     = "#141410"
CARD_BORDER = "#222218"

with open(Path(__file__).parent.parent / "icons" / "waves-horizontal.svg") as f:
    svg_content = f.read()
svg_content = svg_content.replace("currentColor", "#4a9eff")

st.set_page_config(
    page_title="Reservoir Explorer",
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
        border-top: 4px solid #4a9eff;
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
        color: #4a9eff;
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
    div[data-testid="stSelectbox"] label {{
        color: #666655 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
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

df = load_data()

RESERVOIR_COLS = [c for c in df.columns if c not in ["year", "month", "total_hm3"]]

RESERVOIR_NAMES = {
    "RioCofio_LaAcena":                    "La Aceña (Río Cofio)",
    "RioGuadalix_Pedrezuela":              "Pedrezuela (Río Guadalix)",
    "RioGuadarrama-Aulencia_LaJorosa":     "La Jarosa (Río Guadarrama)",
    "RioGuadarrama-Aulencia_Navalmedio":   "Navalmedio (Río Guadarrama)",
    "RioGuadarrama-Aulencia_Valmayor":     "Valmayor (Río Guadarrama)",
    "RioJarama_ElVado":                    "El Vado (Río Jarama)",
    "RioLozoya_ElAtazar":                  "El Atazar (Río Lozoya)",
    "RioLozoya_ElVillar":                  "El Villar (Río Lozoya)",
    "RioLozoya_LaPinilla":                 "La Pinilla (Río Lozoya)",
    "RioLozoya_PuentesViejas":             "Puentes Viejas (Río Lozoya)",
    "RioLozoya_Riosequillo":               "Riosequillo (Río Lozoya)",
    "RioManzanares_Navacerrada":           "Navacerrada (Río Manzanares)",
    "RioManzanares_Santillana":            "Santillana (Río Manzanares)",
}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.markdown("# Reservoir Explorer")
st.markdown(
    "<p style='color:#555544; font-size:0.9rem; margin-top:-12px'>"
    "Individual reservoir capacity · Canal de Isabel II · 1998–2021"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

col_sel, _ = st.columns([1, 2])
with col_sel:
    display_names = {RESERVOIR_NAMES.get(c, c): c for c in RESERVOIR_COLS}
    selected_name = st.selectbox(
        "Select reservoir",
        options=list(display_names.keys()),
        index=list(display_names.keys()).index("El Atazar (Río Lozoya)"),
    )
    selected_col = display_names[selected_name]

series = df[selected_col].dropna()
total  = df["total_hm3"]

mean_val   = float(series.mean())
max_val    = float(series.max())
min_val    = float(series.min())
share_pct  = float(series.mean() / total.mean() * 100)
worst_date = series.idxmin()

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, f"{mean_val:,.1f} hm³",      "Mean capacity"),
    (c2, f"{max_val:,.1f} hm³",       f"Maximum · {series.idxmax().strftime('%b %Y')}"),
    (c3, f"{min_val:,.1f} hm³",       f"Minimum · {worst_date.strftime('%b %Y')}"),
    (c4, f"{share_pct:.1f}%",         "Share of system total"),
]:
    with col:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{val}</div>
            <div class='stat-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown(f"### {selected_name} — Historical Capacity")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=series.index, y=series.values,
    mode="lines", name=selected_name,
    line=dict(color="#4a9eff", width=2),
    fill="tozeroy",
    fillcolor="rgba(74,158,255,0.06)",
))
fig.add_trace(go.Scatter(
    x=total.index, y=total.values,
    mode="lines", name="System total",
    line=dict(color="#333322", width=1.5, dash="dot"),
    yaxis="y2",
))
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG, plot_bgcolor=BG,
    height=380,
    hovermode="x unified",
    legend=dict(orientation="h", font=dict(color="#666655", size=11),
                bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor=CARD_BG),
    yaxis=dict(gridcolor=CARD_BG, title=f"{selected_name} (hm³)"),
    yaxis2=dict(overlaying="y", side="right", showgrid=False,
                title="System total (hm³)", tickfont=dict(color="#444433")),
    margin=dict(t=0, b=0),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### Average Monthly Profile")

monthly_avg = series.groupby(series.index.month).mean()

fig2 = go.Figure(go.Bar(
    x=MONTH_LABELS,
    y=[monthly_avg.get(i, 0) for i in range(1, 13)],
    marker_color="#4a9eff",
    opacity=0.85,
))
fig2.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG, plot_bgcolor=BG,
    height=280,
    margin=dict(t=0, b=0),
    xaxis=dict(gridcolor=CARD_BG),
    yaxis=dict(gridcolor=CARD_BG, title="Mean capacity (hm³)"),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("### System Contribution")

reservoir_means = df[RESERVOIR_COLS].mean().sort_values(ascending=True)
colors_bar = [
    "#4a9eff" if c == selected_col else "#222218"
    for c in reservoir_means.index
]

fig3 = go.Figure(go.Bar(
    x=reservoir_means.values,
    y=[RESERVOIR_NAMES.get(c, c) for c in reservoir_means.index],
    orientation="h",
    marker_color=colors_bar,
    opacity=0.9,
    text=[f"{v:.0f} hm³" for v in reservoir_means.values],
    textposition="outside",
    textfont=dict(color="#666655", size=10),
))
fig3.update_layout(
    template="plotly_dark",
    paper_bgcolor=BG, plot_bgcolor=BG,
    height=400,
    margin=dict(l=0, r=80, t=0, b=0),
    xaxis=dict(gridcolor=CARD_BG, showticklabels=False),
    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig3, use_container_width=True)
