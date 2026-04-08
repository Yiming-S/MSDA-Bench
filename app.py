import streamlit as st
import os
from utils import register_plotly_template

st.set_page_config(
    page_title="MSDA-Bench",
    page_icon="file.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

register_plotly_template()

st.markdown('''
<style>
:root {
    --lab-bg: #F4F7FB;
    --lab-panel: rgba(255, 255, 255, 0.88);
    --lab-panel-strong: rgba(255, 255, 255, 0.96);
    --lab-sidebar: #ECF2F8;
    --lab-border: #D6E2F0;
    --lab-grid: #D7E0EA;
    --lab-ink: #132238;
    --lab-muted: #61758C;
    --lab-accent: #2F6FED;
    --lab-accent-soft: #E7F0FF;
    --lab-shadow: 0 16px 34px rgba(19, 34, 56, 0.07);
    --lab-shadow-soft: 0 10px 24px rgba(19, 34, 56, 0.05);
    --lab-mono: "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
    --lab-sans: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--lab-sans);
    font-variant-numeric: tabular-nums;
}

body {
    background: linear-gradient(180deg, #F7FAFD 0%, #F2F6FB 100%);
    color: var(--lab-ink);
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(47, 111, 237, 0.06), transparent 32%),
        linear-gradient(180deg, #F7FAFD 0%, #F2F6FB 100%);
}

#MainMenu,
footer,
header,
div[data-testid="stDecoration"] {
    visibility: hidden;
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

.block-container {
    padding-top: 1.35rem !important;
    padding-bottom: 1.75rem !important;
    padding-left: 2.15rem !important;
    padding-right: 2.15rem !important;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EEF4FA 0%, #E7EEF6 100%);
    border-right: 1px solid var(--lab-border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 0.8rem;
}

h1, h2, h3 {
    color: var(--lab-ink);
    letter-spacing: -0.03em;
    font-weight: 700 !important;
}

h1 {
    font-size: 2.05rem;
}

p, li {
    color: var(--lab-muted);
    line-height: 1.6;
}

hr {
    border: none;
    border-top: 1px solid var(--lab-border);
    margin: 1.2rem 0;
}

code, pre {
    font-family: var(--lab-mono);
}

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--lab-panel-strong) 0%, rgba(247, 250, 253, 0.95) 100%);
    border: 1px solid var(--lab-border);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: var(--lab-shadow-soft);
}

div[data-testid="stMetricLabel"] {
    color: var(--lab-muted);
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricDelta"] {
    font-family: var(--lab-mono);
    color: var(--lab-ink);
}

div[data-testid="stDataFrame"],
div[data-testid="stTable"],
div[data-testid="stPlotlyChart"] {
    background: var(--lab-panel);
    border: 1px solid var(--lab-border);
    border-radius: 20px;
    box-shadow: var(--lab-shadow);
    overflow: hidden;
}

div[data-testid="stPlotlyChart"] {
    padding: 0.2rem 0.2rem 0.1rem 0.2rem;
}

div[data-testid="stDataFrame"] *,
div[data-testid="stTable"] table,
div[data-testid="stTable"] th,
div[data-testid="stTable"] td {
    font-family: var(--lab-mono) !important;
    font-variant-numeric: tabular-nums;
}

div[data-testid="stTable"] table {
    border-collapse: collapse;
}

div[data-testid="stTable"] th,
div[data-testid="stTable"] td {
    border-bottom: 1px solid #E8EEF5;
}

div[data-testid="stAlert"] {
    border-radius: 18px;
    border: 1px solid var(--lab-border);
    background: rgba(255, 255, 255, 0.82);
}

div.stButton > button,
div[data-testid="stBaseButton-secondary"] {
    border-radius: 999px;
    border: 1px solid var(--lab-border);
    background: linear-gradient(180deg, #FFFFFF 0%, #F4F8FC 100%);
    color: var(--lab-ink);
    box-shadow: var(--lab-shadow-soft);
}

div.stButton > button:hover,
div[data-testid="stBaseButton-secondary"]:hover {
    border-color: rgba(47, 111, 237, 0.32);
    color: var(--lab-accent);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    border-radius: 16px !important;
    border-color: var(--lab-border) !important;
    background: rgba(255, 255, 255, 0.82) !important;
    box-shadow: none !important;
}

div.stSelectbox label,
div.stMultiSelect label,
div.stSlider label,
div.stRadio > label {
    color: var(--lab-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

div.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

div.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.64);
    border: 1px solid var(--lab-border);
    border-radius: 999px;
    padding: 0.45rem 0.95rem;
    height: auto;
    color: var(--lab-muted);
}

div.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--lab-accent-soft);
    border-color: rgba(47, 111, 237, 0.24);
    color: var(--lab-accent);
}

section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

section[data-testid="stSidebar"] div.stRadio > label {
    display: none !important;
}

section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label[data-baseweb="radio"] {
    background: rgba(255, 255, 255, 0.46);
    border: 1px solid transparent;
    border-radius: 16px;
    padding: 0.8rem 0.95rem;
    margin-bottom: 0.45rem;
    transition: all 0.18s ease;
}

section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label[data-baseweb="radio"]:hover {
    background: rgba(255, 255, 255, 0.72);
    border-color: rgba(47, 111, 237, 0.12);
    transform: translateX(2px);
}

section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(180deg, #F6FAFF 0%, #EAF2FF 100%);
    border-color: rgba(47, 111, 237, 0.28);
    box-shadow: inset 0 0 0 1px rgba(47, 111, 237, 0.06);
}

section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label[data-baseweb="radio"] p {
    font-size: 0.98rem;
    font-weight: 650;
    color: #40556E;
    margin: 0;
}

section[data-testid="stSidebar"] div.stRadio div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) p {
    color: var(--lab-accent);
}
</style>
''', unsafe_allow_html=True)

# Import pages
from views import (
    page_1_overview, page_2_benchmark, page_3_stability,
    page_4_config, page_5_subject, page_6_da,
    page_7_mechanism, page_8_target, page_9_error, page_10_efficiency
)

# Sidebar: dataset selector
import base64, os
_svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file.svg")
with open(_svg_path, "rb") as _f:
    _svg_b64 = base64.b64encode(_f.read()).decode()
st.sidebar.markdown(
    f'''
    <div style="
        background: linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(246,250,253,0.94) 100%);
        border: 1px solid #D6E2F0;
        border-radius: 22px;
        padding: 18px 16px;
        margin-top: -6px;
        margin-bottom: 18px;
        box-shadow: 0 16px 34px rgba(19, 34, 56, 0.08);
    ">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
            <div style="
                width:58px;
                height:58px;
                border-radius:18px;
                background:#E7F0FF;
                border:1px solid #C7D8EE;
                display:flex;
                align-items:center;
                justify-content:center;
                flex-shrink:0;
            ">
                <img src="data:image/svg+xml;base64,{_svg_b64}" width="34" height="34" style="opacity:0.95;">
            </div>
            <div>
                <div style="
                    margin:0 0 2px 0;
                    font-size:0.68rem;
                    font-weight:700;
                    letter-spacing:0.14em;
                    text-transform:uppercase;
                    color:#6B7E95;
                ">Cool Light Lab</div>
                <h1 style="
                    margin:0;
                    font-size:1.45rem;
                    font-weight:800;
                    color:#132238;
                    letter-spacing:-0.05em;
                    line-height:1.0;
                ">MSDA-Bench</h1>
            </div>
        </div>
        <p style="
            margin:0;
            font-size:0.78rem;
            font-weight:500;
            color:#5F7088;
            line-height:1.55;
        ">Multi-source EEG session-shift benchmark with matched-pipeline comparison, mechanism inspection, and runtime tradeoff analysis.</p>
    </div>
    ''',
    unsafe_allow_html=True
)

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATA_DIRS = {
    name: os.path.join(_BASE, name)
    for name in sorted(os.listdir(_BASE))
    if os.path.isdir(os.path.join(_BASE, name))
}

dataset = st.sidebar.selectbox("Dataset", list(DATA_DIRS.keys()))

# Load data
from data_loader import get_data_store
store = get_data_store(DATA_DIRS[dataset], dataset)

if st.sidebar.button("Refresh Data"):
    st.cache_resource.clear()
    st.rerun()

# Page navigation using tabs (simpler than st.navigation for compatibility)
pages = {
    "Dataset Overview": page_1_overview,
    "Pipeline Comparison": page_2_benchmark,
    "Selection Sensitivity": page_3_stability,
    "Configuration Effects": page_4_config,
    "Subject-Level Results": page_5_subject,
    "Adaptation Effects": page_6_da,
    "Session Usage Mechanisms": page_7_mechanism,
    "Target Session Difficulty": page_8_target,
    "Error Analysis": page_9_error,
    "Runtime & Efficiency": page_10_efficiency,
}

page_name = st.sidebar.radio("Page", list(pages.keys()))
pages[page_name].render(store, dataset)

# Author info at bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.75rem; color:#6F8096; line-height:1.6;">'
    '<b>Yiming Shen</b> &amp; <b>David Degras</b><br>'
    'Department of Mathematics<br>'
    'University of Massachusetts Boston'
    '</div>',
    unsafe_allow_html=True
)
