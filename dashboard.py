# dashboard.py — Global Air Quality Dashboard
# Reads processed results from Azure Blob Storage
# Run with: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from azure.storage.blob import BlobServiceClient
import os
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML-OPS Global Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark Atmospheric Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1527 50%, #0a1520 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(13, 21, 39, 0.95);
        border-right: 1px solid rgba(99, 220, 255, 0.15);
    }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(145deg, rgba(20, 35, 60, 0.9), rgba(10, 20, 40, 0.9));
        border: 1px solid rgba(99, 220, 255, 0.2);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 16px;
        box-shadow: 0 4px 24px rgba(0, 120, 200, 0.12);
        transition: border-color 0.3s;
    }
    .kpi-card:hover {
        border-color: rgba(99, 220, 255, 0.5);
    }
    .kpi-label {
        color: rgba(180, 210, 255, 0.65);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #63dcff;
        font-size: 36px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }
    .kpi-sub {
        color: rgba(180, 210, 255, 0.45);
        font-size: 12px;
        margin-top: 6px;
    }

    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 18px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin: 28px 0 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(99, 220, 255, 0.15);
    }

    /* AQI badge */
    .badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Hero title */
    .hero {
        text-align: center;
        padding: 10px 0 24px;
    }
    .hero h1 {
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(90deg, #63dcff, #a78bfa, #63dcff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        margin: 0;
    }
    .hero p {
        color: rgba(180, 210, 255, 0.55);
        font-size: 14px;
        margin-top: 6px;
        letter-spacing: 1px;
    }
    @keyframes shimmer {
        0%   { background-position: 0% }
        100% { background-position: 200% }
    }

    /* Plotly chart background */
    .js-plotly-plot .plotly .bg { fill: transparent !important; }

    /* Streamlit metric tweaks */
    div[data-testid="metric-container"] {
        background: rgba(20, 35, 60, 0.7);
        border: 1px solid rgba(99, 220, 255, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COLOUR PALETTE (shared)
# ─────────────────────────────────────────────
CHART_BG   = "rgba(0,0,0,0)"
PAPER_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(99,220,255,0.08)"
FONT_COLOR = "rgba(200,220,255,0.85)"

AQI_COLORS = {
    "Good":                          "#00e676",
    "Moderate":                      "#ffeb3b",
    "Unhealthy for Sensitive Groups": "#ff9800",
    "Unhealthy":                     "#f44336",
    "Very Unhealthy":                "#9c27b0",
    "Hazardous":                     "#b71c1c",
}
CLUSTER_COLORS = {
    "Low Pollution":      "#00e676",
    "Moderate Pollution": "#ff9800",
    "High Pollution":     "#f44336",
}
ACCENT = "#63dcff"
PURPLE = "#a78bfa"

def chart_layout(title="", height=380, **kwargs):
    return dict(
        title=dict(text=title, font=dict(color=FONT_COLOR, size=15, family="Space Grotesk"), x=0.01),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COLOR, family="Space Grotesk"),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        **kwargs,
    )

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    conn_str = os.environ.get("AZURE_CONN_STR", "")

    # ── AZURE path
    client = BlobServiceClient.from_connection_string(conn_str)

    def read(blob_name):
        blob = client.get_blob_client("output", blob_name)
        return pd.read_csv(io.BytesIO(blob.download_blob().readall()))

    df              = read("predictions.csv")
    country_agg     = read("country_summary.csv")
    hourly          = read("hourly_trends.csv")
    fi              = read("feature_importance.csv")
    model_scores    = read("model_scores.csv")
    test_comparison = read("test_comparison.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df, country_agg, hourly, fi, model_scores, test_comparison


with st.spinner("Loading air quality data…"):
    df, country_agg, hourly, fi, model_scores, test_comp = load_data()

# ─────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌐 Filters")
    all_countries = sorted(df["country"].unique().tolist())
    selected_countries = st.multiselect(
        "Countries", all_countries, default=all_countries[:6],
        help="Select one or more countries"
    )
    if not selected_countries:
        selected_countries = all_countries

    aqi_range = st.slider("AQI Range", int(df["aqi"].min()), int(df["aqi"].max()),
                          (int(df["aqi"].min()), int(df["aqi"].max())))

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    best_row = model_scores.loc[model_scores["mae"].idxmin()]
    st.success(f"🏆 **{best_row['model']}**\n\nMAE: `{best_row['mae']:.2f}` | R²: `{best_row['r2']:.4f}`")

    st.markdown("---")
    st.caption("Data refreshes every 5 min")
    st.caption("Pipeline: Azure → Docker → K8s → Jenkins")

# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
mask = df["country"].isin(selected_countries) & df["aqi"].between(*aqi_range)
dff  = df[mask].copy()
cagg = country_agg[country_agg["country"].isin(selected_countries)].copy()

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌍ML-OPS Global Air Quality Intelligence</h1>
    <p>REAL-TIME MONITORING · ML-POWERED PREDICTIONS · CLOUD PIPELINE</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, sub=""):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "Total Records",    f"{len(dff):,}",             f"{dff['country'].nunique()} countries")
kpi(k2, "Avg AQI",          f"{dff['aqi'].mean():.1f}",  "across selection")
kpi(k3, "Avg PM2.5",        f"{dff['pm25'].mean():.1f}", "µg/m³")
kpi(k4, "Avg Temperature",  f"{dff['temperature'].mean():.1f}°C", "")
kpi(k5, "Model R²",
    f"{model_scores.loc[model_scores['mae'].idxmin(),'r2']:.3f}",
    f"Best: {model_scores.loc[model_scores['mae'].idxmin(),'model']}")

# ─────────────────────────────────────────────
# ROW 1 — World Map  +  AQI Distribution
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Geographic Overview</div>', unsafe_allow_html=True)
col_map, col_hist = st.columns([3, 2])

with col_map:
    fig_map = px.choropleth(
        cagg,
        locations="country",
        locationmode="ISO-3",
        color="avg_aqi",
        hover_name="country",
        hover_data={"avg_pm25": ":.1f", "avg_temp": ":.1f", "cluster_label": True, "avg_aqi": ":.1f"},
        color_continuous_scale=["#00e676","#ffeb3b","#ff9800","#f44336","#9c27b0"],
        range_color=[df["avg_aqi"].min() if "avg_aqi" in df.columns else 30,
                     country_agg["avg_aqi"].max()],
        labels={"avg_aqi": "Avg AQI"},
        title="Average AQI by Country",
    )
    fig_map.update_layout(
        **chart_layout(height=380),
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(99,220,255,0.2)",
            showland=True, landcolor="rgba(20,35,60,0.8)",
            showocean=True, oceancolor="rgba(5,15,35,0.9)",
            showcountries=True, countrycolor="rgba(99,220,255,0.15)",
        ),
        coloraxis_colorbar=dict(title="AQI", tickfont=dict(color=FONT_COLOR)),
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_hist:
    aqi_cats = dff["aqi_category"].value_counts().reset_index()
    aqi_cats.columns = ["category", "count"]
    cat_order = list(AQI_COLORS.keys())
    aqi_cats["color"] = aqi_cats["category"].map(AQI_COLORS)
    fig_donut = go.Figure(go.Pie(
        labels=aqi_cats["category"],
        values=aqi_cats["count"],
        hole=0.62,
        marker=dict(colors=[AQI_COLORS.get(c, "#888") for c in aqi_cats["category"]],
                    line=dict(color="#0a0e1a", width=2)),
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Records: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig_donut.add_annotation(
        text=f"<b>{len(dff):,}</b><br>records",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=FONT_COLOR),
    )
    fig_donut.update_layout(**chart_layout("AQI Category Breakdown", height=380))
    st.plotly_chart(fig_donut, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 2 — Country Rankings  +  Cluster Scatter
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Country Analysis & Clustering</div>', unsafe_allow_html=True)
col_bar, col_scatter = st.columns([3, 2])

with col_bar:
    top_n = cagg.sort_values("avg_aqi", ascending=False).head(20)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=top_n["avg_aqi"],
        y=top_n["country"],
        orientation="h",
        marker=dict(
            color=top_n["avg_aqi"],
            colorscale=[[0,"#00e676"],[0.4,"#ffeb3b"],[0.7,"#ff9800"],[1,"#f44336"]],
            line=dict(width=0),
        ),
        text=top_n["avg_aqi"].apply(lambda x: f"{x:.1f}"),
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>Avg AQI: %{x:.1f}<extra></extra>",
    ))
    fig_bar.update_layout(
        **chart_layout("Countries Ranked by Average AQI", height=380),
               xaxis_title="Average AQI",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_scatter:
    cluster_colors_list = [CLUSTER_COLORS.get(str(c), "#888") for c in cagg["cluster_label"]]
    fig_cl = px.scatter(
        cagg, x="avg_pm25", y="avg_aqi",
        color="cluster_label",
        size="records",
        hover_name="country",
        hover_data={"avg_temp": ":.1f", "records": ":,"},
        color_discrete_map=CLUSTER_COLORS,
        title="PM2.5 vs AQI (Clustered)",
        labels={"avg_pm25": "Avg PM2.5", "avg_aqi": "Avg AQI", "cluster_label": "Cluster"},
        size_max=30,
    )
    fig_cl.update_layout(**chart_layout(height=380))
    fig_cl.update_traces(marker=dict(line=dict(width=1, color="rgba(99,220,255,0.3)")))
    st.plotly_chart(fig_cl, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 3 — Hourly Trend  +  Feature Importance
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">⏱️ Temporal Patterns & Model Insights</div>', unsafe_allow_html=True)
col_hourly, col_fi = st.columns([3, 2])

with col_hourly:
    hourly_sel = hourly[hourly["country"].isin(selected_countries[:8])]
    fig_hourly = px.line(
        hourly_sel, x="hour", y="avg_aqi", color="country",
        title="Hourly AQI Pattern by Country",
        labels={"hour": "Hour of Day", "avg_aqi": "Average AQI"},
        markers=False,
    )
    fig_hourly.update_traces(line=dict(width=2.5))
    fig_hourly.update_layout(**chart_layout(height=360))
    fig_hourly.update_xaxes(tickvals=list(range(0,24,3)), ticktext=[f"{h:02d}:00" for h in range(0,24,3)])
    st.plotly_chart(fig_hourly, use_container_width=True)

with col_fi:
    top_fi = fi.head(12)
    fig_fi = go.Figure(go.Bar(
        x=top_fi["importance"],
        y=top_fi["feature"],
        orientation="h",
        marker=dict(
            color=top_fi["importance"],
            colorscale=[[0, PURPLE],[1, ACCENT]],
            line=dict(width=0),
        ),
        text=top_fi["importance"].apply(lambda x: f"{x:.3f}"),
        textposition="auto",
    ))
    fig_fi.update_layout(
        **chart_layout("Feature Importance (Random Forest)", height=360),
           )
    st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 4 — Predicted vs Actual  +  Pollutant Radar
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🤖 ML Predictions & Pollutant Profiles</div>', unsafe_allow_html=True)
col_pred, col_radar = st.columns([3, 2])

with col_pred:
    sample = test_comp.sample(min(1500, len(test_comp)), random_state=1)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=sample["actual_aqi"], y=sample["predicted_aqi"],
        mode="markers",
        marker=dict(color=ACCENT, size=5, opacity=0.5,
                    line=dict(width=0.5, color="rgba(99,220,255,0.3)")),
        name="Predictions",
        hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
    ))
    # Perfect prediction line
    lims = [min(sample["actual_aqi"].min(), sample["predicted_aqi"].min()),
            max(sample["actual_aqi"].max(), sample["predicted_aqi"].max())]
    fig_pred.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color="#ff9800", dash="dash", width=2),
        name="Perfect Fit",
    ))
    fig_pred.update_layout(
        **chart_layout("Actual vs Predicted AQI", height=380),
        xaxis_title="Actual AQI",
        yaxis_title="Predicted AQI",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

with col_radar:
    # Average pollutants per cluster
    if "cluster_label" in cagg.columns:
        radar_data = cagg.groupby("cluster_label")[["avg_pm25","avg_pm10","avg_no2","avg_so2"]].mean()
        cats = ["PM2.5","PM10","NO2","SO2"]
        fig_radar = go.Figure()
        for cluster, row in radar_data.iterrows():
            vals = row.tolist()
            vals += [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]],
                fill="toself", name=str(cluster),
                line=dict(color=CLUSTER_COLORS.get(str(cluster), "#888"), width=2),
                fillcolor="rgba(244,67,54,0.2)"
            ))
        fig_radar.update_layout(
            **chart_layout("Pollutant Profile by Cluster", height=380),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR)),
                angularaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR)),
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ─────────────────────────────────────────────
# ROW 5 — Model Comparison + Correlation Heatmap
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Model Comparison & Correlations</div>', unsafe_allow_html=True)
col_models, col_corr = st.columns([1, 2])

with col_models:
    fig_models = go.Figure()
    fig_models.add_trace(go.Bar(
        name="MAE (lower=better)",
        x=model_scores["model"],
        y=model_scores["mae"],
        marker_color=ACCENT,
        text=model_scores["mae"].apply(lambda x: f"{x:.2f}"),
        textposition="auto",
    ))
    fig_models.add_trace(go.Scatter(
        name="R² × 20 (higher=better)",
        x=model_scores["model"],
        y=model_scores["r2"] * 20,
        mode="markers+lines",
        marker=dict(color=PURPLE, size=10),
        line=dict(color=PURPLE, dash="dot"),
        yaxis="y2",
    ))
    fig_models.update_layout(
        **chart_layout("Model Comparison", height=340),
               yaxis2=dict(title="R² × 20", overlaying="y", side="right", showgrid=False),
           )
    st.plotly_chart(fig_models, use_container_width=True)

with col_corr:
    corr_cols = ["aqi","pm25","pm10","no2","so2","o3","co","temperature","humidity","wind_speed"]
    corr_df = dff[corr_cols].corr().round(2)
    fig_heat = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=corr_cols, y=corr_cols,
        colorscale=[[0,"#f44336"],[0.5,"#0a1520"],[1,ACCENT]],
        zmid=0,
        text=corr_df.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="%{x} × %{y}<br>r = %{z:.2f}<extra></extra>",
        showscale=True,
    ))
    fig_heat.update_layout(**chart_layout("Pollutant Correlation Matrix", height=340))
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
# DATA TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">🗃️ Country Summary Table</div>', unsafe_allow_html=True)
display_cols = ["country","avg_aqi","avg_pm25","avg_pm10","avg_no2","avg_temp","avg_humidity","pollution_idx","cluster_label","aqi_rank"]
show_cols = [c for c in display_cols if c in cagg.columns]
st.dataframe(
    cagg[show_cols].sort_values("avg_aqi", ascending=False).reset_index(drop=True),
    use_container_width=True,
    height=320,
)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:rgba(180,210,255,0.3); font-size:12px;'>"
    "ML-OPS Global Air Quality Intelligence Dashboard · Cloud Data Pipeline · "
    "Azure → Docker → Kubernetes → Jenkins · Built with Streamlit & Plotly"
    "</p>",
    unsafe_allow_html=True,
)
