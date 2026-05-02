import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Rental AI",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #080c18 !important;
    color: #c8d6e8 !important;
}
.stApp { background: #080c18 !important; }
#MainMenu, footer, header { visibility: hidden; }

/* Hide sidebar completely */
[data-testid="stSidebar"] {
    display: none !important;
}

/* Main content area - full width */
.main > div {
    width: 100% !important;
    padding: 0 !important;
}
.block-container {
    max-width: 100% !important;
    padding: 1rem 2rem !important;
}

/* Hide the original radio buttons and columns */
.stRadio, div[data-testid="column"] {
    display: none !important;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(145deg, #111827, #0f1c2e);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 14px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(56,189,248,0.1);
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, linear-gradient(90deg,#14b8a6,#38bdf8));
}
.kpi-label {
    font-size: 0.68rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #3a5472;
    font-weight: 600;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #e8f0fb;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-sub { font-size: 0.74rem; color: #2e4461; }
.kpi-icon { position: absolute; top: 16px; right: 18px; font-size: 1.35rem; opacity: 0.5; }

.sec-title { font-size: 1rem; font-weight: 700; color: #e2ecfb; margin-bottom: 2px; }
.sec-sub { font-size: 0.76rem; color: #3a5472; margin-bottom: 12px; }

.chart-card {
    background: linear-gradient(145deg, #0f1826, #0c1522);
    border: 1px solid rgba(56,189,248,0.09);
    border-radius: 14px;
    padding: 20px 18px 10px;
    transition: transform 0.2s ease;
}
.chart-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}

/* Form styling */
.stSelectbox > div > div, 
.stNumberInput > div > div > input,
.stSlider > div {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding-left: 0 !important;
}

.stSelectbox > div > div:focus, 
.stNumberInput > div > div > input:focus {
    border-bottom: 1px solid #38bdf8 !important;
}

/* Widget styling */
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: #5a7595 !important;
    font-size: 0.75rem !important;
    font-weight: 400 !important;
    margin-bottom: 2px !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(90deg, #14b8a6, #38bdf8) !important;
    color: #040810 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 13px 0 !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
    margin-top: 20px;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { 
    opacity: 0.9 !important; 
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 16px rgba(56,189,248,0.3) !important;
}

/* Result card */
.result-card {
    background: linear-gradient(145deg, #0a1f14, #071510);
    border: 1px solid rgba(20,184,166,0.22);
    border-radius: 14px;
    padding: 30px 22px;
    text-align: center;
}
.result-num {
    font-size: 3.4rem;
    font-weight: 800;
    color: #2dd4bf;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}

/* Full width chart container */
.full-width-chart {
    width: 100%;
    margin-top: 30px;
    margin-bottom: 20px;
}

/* Model selector styling */
.model-selector {
    background: linear-gradient(145deg, #0f1826, #0c1522);
    border: 1px solid rgba(56,189,248,0.09);
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 20px;
}
.model-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 30px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-left: 10px;
    background: rgba(56,189,248,0.1);
    color: #38bdf8;
}
.comparison-table {
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    padding: 15px;
    margin-top: 15px;
}
.comparison-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(56,189,248,0.1);
}
.comparison-row:last-child {
    border-bottom: none;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
hours = list(range(24))
registered = [12,6,4,3,4,18,70,155,185,130,108,112,118,105,112,122,178,192,155,108,84,62,42,22]
casual      = [4, 2,1,1,2, 6,12, 25, 42, 52, 58, 62, 65, 63, 60, 58, 52, 48, 38, 28,22,16,10, 5]
months_label = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
total_rentals= [40200,22000,52000,72000,96000,115000,138000,135000,138000,104000,72000,25000]
avg_temp     = [4,5,9,15,19,24,27,26,21,15,9,5]

C = {
    'teal':'#14b8a6','cyan':'#38bdf8','purple':'#a78bfa',
    'orange':'#fb923c','green':'#4ade80',
    'grid':'rgba(255,255,255,0.04)','muted':'#3a5472','text':'#c8d6e8',
}

BASE_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans', color=C['text'], size=12),
    xaxis=dict(gridcolor=C['grid'], color=C['muted'], showline=False),
    yaxis=dict(gridcolor=C['grid'], color=C['muted'], showline=False),
    legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                yanchor='bottom', y=-0.26, xanchor='center', x=0.5,
                font=dict(size=12, color=C['text'])),
    hovermode='x unified',
    hoverlabel=dict(bgcolor='#1a2840', bordercolor='rgba(56,189,248,0.3)',
                    font=dict(family='Plus Jakarta Sans', color='#e2ecfb', size=13))
)


def kpi(icon, label, value, sub, accent):
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent}">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def sec(title, sub):
    st.markdown(f'<div class="sec-title">{title}</div><div class="sec-sub">{sub}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS FOR NORMALIZATION
# ─────────────────────────────────────────────
def normalize_temperature(temp_celsius):
    min_temp = -5
    max_temp = 40
    clipped_temp = np.clip(temp_celsius, min_temp, max_temp)
    normalized = (clipped_temp - min_temp) / (max_temp - min_temp)
    return normalized

def normalize_humidity(humidity_percent):
    clipped_humidity = np.clip(humidity_percent, 0, 100)
    normalized = clipped_humidity / 100.0
    return normalized

def normalize_windspeed(windspeed_kmh):
    min_wind = 0
    max_wind = 50
    clipped_wind = np.clip(windspeed_kmh, min_wind, max_wind)
    normalized = (clipped_wind - min_wind) / (max_wind - min_wind)
    return normalized

def predict_for_hours(model, base_features, hours_range):
    predictions = []
    for hour in hours_range:
        features = base_features.copy()
        features[8] = hour
        pred_log = model.predict(features.reshape(1, -1))
        pred = int(max(0, np.expm1(pred_log)[0]))
        predictions.append(pred)
    return predictions

def load_available_models():
    """Load all available model files and return a dictionary of models"""
    model_files = {
        'Gradient Boosting': 'bike_model_gradientboosting.pkl',
        'Random Forest': 'bike_model_randomforest.pkl',
        'XGBoost': 'bike_model_xgboost.pkl',
        'CatBoost': 'bike_model_catboost.pkl'
    }
    
    available_models = {}
    for model_name, filename in model_files.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    available_models[model_name] = pickle.load(f)
            except:
                pass
    
    # Also try to load the original model as fallback
    if not available_models and os.path.exists('bike_model.pkl'):
        try:
            with open('bike_model.pkl', 'rb') as f:
                available_models['Gradient Boosting (Original)'] = pickle.load(f)
        except:
            pass
    
    return available_models

def load_ensemble_info():
    """Load ensemble info if available"""
    if os.path.exists('bike_ensemble_info.pkl'):
        try:
            with open('bike_ensemble_info.pkl', 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None


# ─────────────────────────────────────────────
#  SESSION STATE FOR NAVIGATION AND MODEL
# ─────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = {}


# ─────────────────────────────────────────────
#  MAIN TITLE - ABOVE NAVIGATION ITEMS
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; margin-bottom: 20px; margin-top: 10px;">
    <span style="font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #14b8a6, #38bdf8, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em; display: inline-block; filter: drop-shadow(0 4px 10px rgba(56,189,248,0.2));">
        🚲 Bike Rental AI
    </span>
    <div style="font-size: 0.9rem; color: #3a5472; margin-top: 5px; letter-spacing: 0.1em;">
        INTELLIGENT DEMAND FORECASTING SYSTEM
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TOP NAVIGATION BAR WITH CLICKABLE BUTTONS
# ─────────────────────────────────────────────
# Create columns for the navbar buttons
nav_cols = st.columns([1, 1, 1, 1, 2])

with nav_cols[0]:
    if st.button("📊 Dashboard", key="nav_dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"
        st.rerun()
with nav_cols[1]:
    if st.button("🌤 Weather", key="nav_weather", use_container_width=True):
        st.session_state.page = "Weather Forecast"
        st.rerun()
with nav_cols[2]:
    if st.button("🔮 Predict", key="nav_predict", use_container_width=True):
        st.session_state.page = "Predict Demand"
        st.rerun()
with nav_cols[3]:
    if st.button("📈 Analytics", key="nav_analytics", use_container_width=True):
        st.session_state.page = "Analytics"
        st.rerun()
with nav_cols[4]:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; justify-content:flex-end;">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#22c55e; box-shadow:0 0 10px #22c55e; animation:pulse 2s infinite;"></span>
        <span style="color:#7a91b0; font-size:0.85rem;">Active · Ready</span>
    </div>
    """, unsafe_allow_html=True)

# Style the navbar buttons
st.markdown("""
<style>
div[data-testid="column"] button {
    background: transparent !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 30px !important;
    padding: 10px 20px !important;
    color: #5a7595 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}
div[data-testid="column"] button:hover {
    background: rgba(56,189,248,0.1) !important;
    color: #c8d6e8 !important;
    transform: translateY(-1px) !important;
    border-color: rgba(56,189,248,0.3) !important;
}
div[data-testid="column"] button:active {
    background: linear-gradient(90deg, #14b8a6, #38bdf8) !important;
    color: #040810 !important;
}
@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(1.1); }
    100% { opacity: 1; transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# Highlight active page button
if st.session_state.page == "Dashboard":
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-child(1) button {
        background: linear-gradient(90deg, #14b8a6, #38bdf8) !important;
        color: #040810 !important;
        box-shadow: 0 4px 12px rgba(56,189,248,0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)
elif st.session_state.page == "Weather Forecast":
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-child(2) button {
        background: linear-gradient(90deg, #14b8a6, #38bdf8) !important;
        color: #040810 !important;
        box-shadow: 0 4px 12px rgba(56,189,248,0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)
elif st.session_state.page == "Predict Demand":
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-child(3) button {
        background: linear-gradient(90deg, #14b8a6, #38bdf8) !important;
        color: #040810 !important;
        box-shadow: 0 4px 12px rgba(56,189,248,0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)
elif st.session_state.page == "Analytics":
    st.markdown("""
    <style>
    div[data-testid="column"]:nth-child(4) button {
        background: linear-gradient(90deg, #14b8a6, #38bdf8) !important;
        color: #040810 !important;
        box-shadow: 0 4px 12px rgba(56,189,248,0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Add a separator line
st.markdown("<hr style='border-color: rgba(56,189,248,0.1); margin: 20px 0 30px 0;'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE CONTENT
# ─────────────────────────────────────────────
# Dashboard
if st.session_state.page == "Dashboard":
    col_h, col_w = st.columns([5,1])
    with col_h:
        st.markdown('<div style="font-size:1.65rem;font-weight:800;color:#e2ecfb;line-height:1.1;">Bike Rental Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.78rem;color:#3a5472;margin-bottom:18px;">Historical analysis · 365 days · 8,760 hourly records</div>', unsafe_allow_html=True)
    with col_w:
        st.markdown('<div style="text-align:right;padding-top:6px;"><span style="background:linear-gradient(90deg,#fef3c7,#fde68a);color:#92400e;border-radius:30px;padding:7px 16px;font-weight:700;font-size:0.82rem;">☀️ Clear 22°C</span></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4, gap="small")
    with c1: kpi("🚲","Total Rentals","1,058,318","Past 12 months","linear-gradient(90deg,#14b8a6,#38bdf8)")
    with c2: kpi("📈","Avg Daily","2,900","Rentals per day","linear-gradient(90deg,#a78bfa,#818cf8)")
    with c3: kpi("⏰","Peak Hour","17:00","Avg 257 bikes / hr","linear-gradient(90deg,#fb923c,#f59e0b)")
    with c4: kpi("🌦","Weather Effect","13% drop","Clear vs rainy days","linear-gradient(90deg,#4ade80,#22d3ee)")

    st.markdown("<br>", unsafe_allow_html=True)

    # Hourly chart
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    sec("Average Hourly Demand Pattern", "Registered vs Casual riders across 24 hours")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=hours, y=registered, name="Registered",
        mode='lines', fill='tozeroy',
        line=dict(color=C['cyan'], width=2.5, shape='spline'),
        fillcolor='rgba(56,189,248,0.13)',
        hovertemplate='<b>Registered</b>: %{y}'
    ))
    fig1.add_trace(go.Scatter(
        x=hours, y=casual, name="Casual",
        mode='lines', fill='tozeroy',
        line=dict(color=C['purple'], width=2.5, shape='spline'),
        fillcolor='rgba(167,139,250,0.13)',
        hovertemplate='<b>Casual</b>: %{y}'
    ))
    lay1 = {**BASE_LAYOUT}
    lay1['height'] = 295
    lay1['margin'] = dict(l=10,r=10,t=6,b=40)
    lay1['xaxis'] = dict(
        tickvals=hours, ticktext=[f"{h:02d}:00" for h in hours],
        gridcolor=C['grid'], color=C['muted'], showline=False
    )
    lay1['yaxis'] = dict(range=[0,215], gridcolor=C['grid'], color=C['muted'])
    fig1.update_layout(**lay1)
    st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        sec("Weather Impact on Demand", "Average hourly rentals by weather condition")
        fig2 = go.Figure(go.Bar(
            x=['Clear','Cloudy','Light Rain','Heavy Rain'],
            y=[118, 112, 106, 14],
            marker_color=[C['orange'], '#94a3b8', '#60a5fa', C['purple']],
            marker_line_width=0,
            hovertemplate='%{x}: <b>%{y}</b> avg rentals<extra></extra>'
        ))
        lay2 = {**BASE_LAYOUT}
        lay2['height'] = 280
        lay2['margin'] = dict(l=10,r=10,t=6,b=10)
        lay2['yaxis'] = dict(gridcolor=C['grid'], color=C['muted'], range=[0,145])
        lay2['bargap'] = 0.35
        lay2['showlegend'] = False
        fig2.update_layout(**lay2)
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        sec("Seasonal Split", "Avg daily rentals by season")
        fig3 = go.Figure(go.Pie(
            labels=['Spring','Summer','Fall','Winter'],
            values=[27,34,28,12],
            hole=0.0,
            marker=dict(
                colors=[C['green'], C['orange'], '#fb923c', '#60a5fa'],
                line=dict(color='#0c1522', width=2.5)
            ),
            textinfo='label+percent',
            textfont=dict(size=13, color='#e2ecfb'),
            hovertemplate='<b>%{label}</b>: %{value}%<extra></extra>'
        ))
        lay3 = {**BASE_LAYOUT}
        lay3['height'] = 280
        lay3['margin'] = dict(l=10,r=10,t=6,b=10)
        lay3['showlegend'] = False
        fig3.update_layout(**lay3)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)


# Weather Forecast
elif st.session_state.page == "Weather Forecast":
    st.markdown('<div style="font-size:1.65rem;font-weight:800;color:#e2ecfb;margin-bottom:4px;">Weather Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#3a5472;margin-bottom:22px;">7-day outlook and estimated impact on bike demand</div>', unsafe_allow_html=True)

    days    = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    icons   = ["☀️","⛅","🌧️","☀️","☀️","⛅","🌤️"]
    highs   = [22,19,15,24,26,21,23]
    lows    = [14,13,10,16,17,14,15]
    impacts = ["+12%","-5%","-28%","+18%","+22%","+8%","+14%"]
    icolors = ["#4ade80","#fb923c","#f87171","#4ade80","#4ade80","#4ade80","#4ade80"]

    cols = st.columns(7, gap="small")
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{icolors[i]};padding:16px 10px;text-align:center;">
                <div style="font-size:0.65rem;color:#3a5472;letter-spacing:.1em;text-transform:uppercase;">{days[i]}</div>
                <div style="font-size:1.8rem;margin:10px 0;">{icons[i]}</div>
                <div style="font-weight:700;color:#e2ecfb;font-size:0.95rem;">{highs[i]}°C</div>
                <div style="font-size:0.72rem;color:#3a5472;">{lows[i]}°C low</div>
                <div style="margin-top:10px;font-size:0.82rem;font-weight:700;color:{icolors[i]};">{impacts[i]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    sec("Forecasted Demand", "Estimated daily rentals for next 7 days")
    fig_w = go.Figure(go.Bar(
        x=days, y=[3360,2760,1920,3720,4080,3180,3540],
        marker_color=C['cyan'], marker_line_width=0,
        hovertemplate='%{x}: <b>%{y:,}</b> est. rentals<extra></extra>'
    ))
    lay_w = {**BASE_LAYOUT}
    lay_w['height'] = 240
    lay_w['margin'] = dict(l=10,r=10,t=6,b=10)
    lay_w['showlegend'] = False
    lay_w['bargap'] = 0.3
    fig_w.update_layout(**lay_w)
    st.plotly_chart(fig_w, use_container_width=True, config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)


# Predict Demand
elif st.session_state.page == "Predict Demand":
    # Load available models
    available_models = load_available_models()
    ensemble_info = load_ensemble_info()
    
    st.markdown('<div style="font-size:1.65rem;font-weight:800;color:#e2ecfb;margin-bottom:4px;">Predict Demand</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#3a5472;margin-bottom:22px;">Enter conditions to forecast hourly bike rentals</div>', unsafe_allow_html=True)

    # Model selector section
    if available_models:
        st.markdown('<div class="model-selector">', unsafe_allow_html=True)
        
        # Create columns for model selection and info
        model_col1, model_col2 = st.columns([2, 1])
        
        with model_col1:
            model_names = list(available_models.keys())
            
            # Add ensemble option if available
            if ensemble_info:
                model_names.append('Ensemble (Average of All Models)')
            
            selected_model_name = st.selectbox(
                "Select Model for Prediction",
                model_names,
                key="model_selector"
            )
            
            st.session_state.selected_model = selected_model_name
        
        with model_col2:
            # Show model badge and validation score if available
            if selected_model_name != 'Ensemble (Average of All Models)' and ensemble_info:
                if selected_model_name in ensemble_info.get('validation_scores', {}):
                    score = ensemble_info['validation_scores'][selected_model_name]
                    st.markdown(f"""
                    <div style="text-align:right;">
                        <span class="model-badge">RMSLE: {score:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            elif selected_model_name == 'Ensemble (Average of All Models)' and ensemble_info:
                st.markdown(f"""
                <div style="text-align:right;">
                    <span class="model-badge">Ensemble RMSLE: {ensemble_info.get('ensemble_rmsle', 'N/A'):.4f}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No trained models found. Please run Benchmark_solution.ipynb to train and save models first.")
        st.stop()

    col_form, col_out = st.columns([3,2], gap="large")

    with col_form:
        r1, r2 = st.columns(2)
        with r1:
            season = st.selectbox("Season", [1,2,3,4],
                format_func=lambda x:{1:"🌸 Spring",2:"☀️ Summer",3:"🍂 Fall",4:"❄️ Winter"}[x],
                key="season_select")
            yr = st.selectbox("Year", [0,1], format_func=lambda x:"2011" if x==0 else "2012", key="year_select")
            month = st.slider("Month", 1, 12, 6, key="month_slider")
            holiday = st.selectbox("Holiday", [0,1], format_func=lambda x:"No" if x==0 else "Yes", key="holiday_select")
            workingday = st.selectbox("Working Day", [0,1], format_func=lambda x:"No" if x==0 else "Yes", key="working_select")
        
        with r2:
            weather = st.selectbox("Weather", [1,2,3,4],
                format_func=lambda x:{1:"☀️ Clear",2:"⛅ Cloudy",3:"🌧 Light Rain",4:"⛈ Heavy Rain"}[x],
                key="weather_select")
            temp_celsius = st.number_input("Temperature (°C)", -10.0, 45.0, 20.0, 0.5, key="temp_input")
            atemp_celsius = st.number_input("Feels Like (°C)", -10.0, 45.0, 20.0, 0.5, key="atemp_input")
            humidity_percent = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, 1.0, key="humidity_input")
            windspeed_kmh = st.number_input("Wind Speed (km/h)", 0.0, 60.0, 10.0, 0.5, key="wind_input")
        
        hour = st.slider("Hour of Day (0–23)", 0, 23, 17, key="hour_slider")
        day = st.slider("Day of Month", 1, 31, 15, key="day_slider")
        dayofweek = st.selectbox("Day of Week", [0,1,2,3,4,5,6], 
                                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
                                key="dow_select")
        
        # Add compare button
        compare_col1, compare_col2 = st.columns(2)
        with compare_col1:
            predict_btn = st.button("🔮  Predict with Selected Model", key="predict_btn")
        with compare_col2:
            compare_btn = st.button("📊  Compare All Models", key="compare_btn")

    with col_out:
        temp_norm = normalize_temperature(temp_celsius)
        atemp_norm = normalize_temperature(atemp_celsius)
        humidity_norm = normalize_humidity(humidity_percent)
        windspeed_norm = normalize_windspeed(windspeed_kmh)
        
        base_features = np.array([
            season, holiday, workingday, weather,
            temp_norm, atemp_norm, humidity_norm, windspeed_norm,
            hour, day, month, yr, dayofweek
        ])
        
        season_names = {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}
        weather_names = {1:"Clear",2:"Cloudy",3:"Light Rain",4:"Heavy Rain"}

        if predict_btn:
            try:
                if selected_model_name == 'Ensemble (Average of All Models)' and ensemble_info:
                    # Ensemble prediction
                    ensemble_predictions = []
                    for model_name, model in available_models.items():
                        pred_log = model.predict(base_features.reshape(1, -1))
                        ensemble_predictions.append(pred_log[0])
                    pred_log = np.mean(ensemble_predictions)
                else:
                    # Single model prediction
                    model = available_models[selected_model_name]
                    pred_log = model.predict(base_features.reshape(1, -1))
                
                prediction = int(max(0, np.expm1(pred_log)[0] if isinstance(pred_log, np.ndarray) else np.expm1(pred_log)))
                
                # Generate hourly predictions for the selected model/ensemble
                if selected_model_name == 'Ensemble (Average of All Models)' and ensemble_info:
                    # For ensemble, average predictions from all models for each hour
                    all_hours_predictions = []
                    for hour_i in hours:
                        hour_preds = []
                        for model_name, model in available_models.items():
                            temp_features = base_features.copy()
                            temp_features[8] = hour_i
                            pred_log_i = model.predict(temp_features.reshape(1, -1))
                            hour_preds.append(pred_log_i[0])
                        avg_pred_log = np.mean(hour_preds)
                        all_hours_predictions.append(int(max(0, np.expm1(avg_pred_log))))
                else:
                    model = available_models[selected_model_name]
                    all_hours_predictions = predict_for_hours(model, base_features, hours)
                
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:0.65rem;letter-spacing:.14em;text-transform:uppercase;color:#1a4036;margin-bottom:14px;">
                        {selected_model_name} • Estimated Rentals / Hour
                    </div>
                    <div class="result-num">{prediction:,}</div>
                    <div style="color:#1a4036;margin-top:8px;font-size:0.8rem;">bikes per hour</div>
                    <div style="display:flex;gap:10px;margin-top:22px;">
                        <div style="flex:1;background:rgba(20,184,166,0.07);border-radius:10px;padding:12px;text-align:center;">
                            <div style="color:#1a4036;font-size:0.65rem;letter-spacing:.1em;text-transform:uppercase;">Season</div>
                            <div style="color:#e2ecfb;font-weight:600;margin-top:4px;font-size:0.85rem;">{season_names[season]}</div>
                        </div>
                        <div style="flex:1;background:rgba(20,184,166,0.07);border-radius:10px;padding:12px;text-align:center;">
                            <div style="color:#1a4036;font-size:0.65rem;letter-spacing:.1em;text-transform:uppercase;">Hour</div>
                            <div style="color:#e2ecfb;font-weight:600;margin-top:4px;font-size:0.85rem;">{hour:02d}:00</div>
                        </div>
                        <div style="flex:1;background:rgba(20,184,166,0.07);border-radius:10px;padding:12px;text-align:center;">
                            <div style="color:#1a4036;font-size:0.65rem;letter-spacing:.1em;text-transform:uppercase;">Weather</div>
                            <div style="color:#e2ecfb;font-weight:600;margin-top:4px;font-size:0.85rem;">{weather_names[weather]}</div>
                        </div>
                    </div>
                    <div style="display:flex;gap:10px;margin-top:10px;justify-content:center;font-size:0.7rem;color:#3a5472;">
                        <span>{temp_celsius:.1f}°C</span> • <span>{humidity_percent:.0f}%</span> • <span>{windspeed_kmh:.0f} km/h</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="result-card" style="opacity:0.7;">
                    <div style="font-size:2rem;margin-bottom:12px;">⚠️</div>
                    <div style="color:#fb923c;font-weight:600;">Prediction Error</div>
                    <div style="color:#3a5472;font-size:0.8rem;margin-top:8px;">{str(e)}</div>
                </div>
                """, unsafe_allow_html=True)
        elif compare_btn:
            try:
                # Compare all models
                comparison_results = {}
                all_predictions = {}
                
                for model_name, model in available_models.items():
                    pred_log = model.predict(base_features.reshape(1, -1))
                    pred = int(max(0, np.expm1(pred_log)[0]))
                    comparison_results[model_name] = pred
                    all_predictions[model_name] = pred
                
                # Add ensemble if available
                if ensemble_info:
                    ensemble_pred = int(np.mean(list(comparison_results.values())))
                    comparison_results['Ensemble (Average)'] = ensemble_pred
                
                st.session_state.comparison_results = comparison_results
                
                # Display comparison
                st.markdown("""
                <div class="comparison-table">
                    <div style="font-size:1rem;font-weight:700;color:#e2ecfb;margin-bottom:15px;">
                        📊 Model Comparison Results
                    </div>
                """, unsafe_allow_html=True)
                
                for model_name, pred_value in comparison_results.items():
                    st.markdown(f"""
                    <div class="comparison-row">
                        <span style="color:#c8d6e8;">{model_name}</span>
                        <span style="color:#38bdf8;font-weight:600;font-family:'JetBrains Mono';">{pred_value:,} bikes</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Comparison error: {str(e)}")
        else:
            st.markdown("""
            <div class="result-card" style="opacity:0.4;">
                <div style="font-size:2.4rem;margin-bottom:14px;">🔮</div>
                <div style="color:#3a5472;font-size:0.88rem;">Fill in the conditions<br>and click Predict or Compare</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Hourly forecast chart (if prediction was made)
    if predict_btn and 'all_hours_predictions' in locals():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        sec("Hourly Demand Forecast", f"Predicted bike rentals throughout the day ({season_names[season]}, {weather_names[weather]}) using {selected_model_name}")
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(
            x=hours, y=all_hours_predictions,
            name="Predicted Demand",
            mode='lines+markers',
            line=dict(color=C['cyan'], width=4, shape='spline'),
            marker=dict(size=10, color=C['cyan']),
            hovertemplate='Hour %{x:02d}:00<br><b>%{y:,}</b> bikes<extra></extra>'
        ))
        fig_hourly.add_trace(go.Scatter(
            x=[hour], y=[prediction],
            name=f"Selected Hour ({hour:02d}:00)",
            mode='markers',
            marker=dict(size=16, color=C['orange'], line=dict(width=3, color='white')),
            hovertemplate='Selected: %{x:02d}:00<br><b>%{y:,}</b> bikes<extra></extra>'
        ))
        
        lay_hourly = {**BASE_LAYOUT}
        lay_hourly['height'] = 500
        lay_hourly['margin'] = dict(l=80, r=40, t=40, b=100)
        lay_hourly['xaxis'] = dict(
            tickvals=hours, ticktext=[f"{h:02d}:00" for h in hours],
            tickangle=-45, gridcolor=C['grid'], color=C['muted'],
            showline=False, title="Hour of Day", title_font=dict(size=14), tickfont=dict(size=11)
        )
        lay_hourly['yaxis'] = dict(
            gridcolor=C['grid'], color=C['muted'],
            title="Predicted Rentals (bikes per hour)", title_font=dict(size=14), tickfont=dict(size=11)
        )
        lay_hourly['showlegend'] = True
        lay_hourly['legend'] = dict(
            bgcolor='rgba(0,0,0,0)', orientation='h',
            yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(size=12)
        )
        fig_hourly.update_layout(**lay_hourly)
        
        st.plotly_chart(fig_hourly, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show comparison results if available
    elif compare_btn and st.session_state.comparison_results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        sec("Model Comparison Chart", "Visual comparison of predictions from all models")
        
        # Create comparison bar chart
        models_list = list(st.session_state.comparison_results.keys())
        predictions_list = list(st.session_state.comparison_results.values())
        
        colors = [C['cyan'], C['purple'], C['orange'], C['green'], '#94a3b8']
        
        fig_compare = go.Figure(go.Bar(
            x=models_list,
            y=predictions_list,
            marker_color=colors[:len(models_list)],
            marker_line_width=0,
            text=[f"{p:,}" for p in predictions_list],
            textposition='outside',
            textfont=dict(size=12, color=C['text']),
            hovertemplate='%{x}: <b>%{y:,}</b> bikes<extra></extra>'
        ))
        
        lay_compare = {**BASE_LAYOUT}
        lay_compare['height'] = 400
        lay_compare['margin'] = dict(l=60, r=40, t=40, b=100)
        lay_compare['showlegend'] = False
        lay_compare['bargap'] = 0.3
        lay_compare['xaxis'] = dict(
            tickangle=-30,
            gridcolor=C['grid'], 
            color=C['muted'],
            showline=False,
            title="Model",
            title_font=dict(size=14)
        )
        lay_compare['yaxis'] = dict(
            gridcolor=C['grid'], 
            color=C['muted'],
            title="Predicted Rentals (bikes per hour)",
            title_font=dict(size=14)
        )
        fig_compare.update_layout(**lay_compare)
        
        st.plotly_chart(fig_compare, use_container_width=True, config={'displayModeBar':False})
        st.markdown('</div>', unsafe_allow_html=True)


# Analytics
elif st.session_state.page == "Analytics":
    st.markdown('<div style="font-size:1.65rem;font-weight:800;color:#e2ecfb;margin-bottom:4px;">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;color:#3a5472;margin-bottom:22px;">Deep dive into rental patterns and trends</div>', unsafe_allow_html=True)

    # Monthly dual-axis chart
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    sec("Monthly Rental Volume", "Total rentals per month with average temperature")

    fig_m = make_subplots(specs=[[{"secondary_y": True}]])
    fig_m.add_trace(go.Bar(
        x=months_label, y=total_rentals, name="Total Rentals",
        marker_color=C['cyan'], marker_line_width=0,
        hovertemplate='%{x}: <b>%{y:,}</b><extra></extra>'
    ), secondary_y=False)
    fig_m.add_trace(go.Bar(
        x=months_label, y=avg_temp, name="Avg Temp (°C)",
        marker_color=C['orange'], marker_line_width=0,
        hovertemplate='%{x}: <b>%{y}°C</b><extra></extra>'
    ), secondary_y=True)

    fig_m.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans', color=C['text'], size=12),
        margin=dict(l=10,r=10,t=6,b=40), height=310,
        bargap=0.15, bargroupgap=0.05,
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h',
                    yanchor='bottom', y=-0.28, xanchor='center', x=0.5,
                    font=dict(size=12, color=C['text'])),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#1a2840', bordercolor='rgba(56,189,248,0.3)',
                        font=dict(family='Plus Jakarta Sans', color='#e2ecfb', size=13))
    )
    fig_m.update_xaxes(gridcolor=C['grid'], color=C['muted'])
    fig_m.update_yaxes(gridcolor=C['grid'], color=C['muted'], secondary_y=False)
    fig_m.update_yaxes(gridcolor='rgba(0,0,0,0)', color=C['orange'], secondary_y=True)
    st.plotly_chart(fig_m, use_container_width=True, config={'displayModeBar':False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Insight cards
    insights = [
        ("🌡️","Temperature Sweet Spot",
         "Demand peaks at 20–25°C. Below 5°C or above 35°C reduces rentals by up to 60%."),
        ("💧","Humidity Effect",
         "Humidity above 80% correlates with 30–40% fewer rentals compared to dry conditions."),
        ("👥","Commuter Pattern",
         "Registered users dominate weekday peaks at 8am & 5pm. Casual riders peak on weekends."),
        ("🌦️","Weather Sensitivity",
         "Rainy conditions reduce demand by ~45%. Clear sky days see up to 3× more casual riders."),
    ]
    cols = st.columns(4, gap="small")
    for i, (icon, title, body) in enumerate(insights):
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:linear-gradient(90deg,#38bdf8,#14b8a6);">
                <div style="font-size:1.3rem;margin-bottom:10px;">{icon}</div>
                <div style="font-weight:700;color:#e2ecfb;font-size:0.86rem;margin-bottom:8px;">{title}</div>
                <div style="font-size:0.76rem;color:#3a5472;line-height:1.5;">{body}</div>
            </div>
            """, unsafe_allow_html=True)
            