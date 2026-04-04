import streamlit as st
import os

st.set_page_config(page_title="MSDA-Bench", page_icon="file.svg", layout="wide")

st.markdown('''
<style>
    /* Modern minimalist padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Enhance metric cards */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4F46E5;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
        color: #64748B;
    }
    
    /* Make sidebar radio options nicer */
    div.row-widget.stRadio > div {
        background: transparent;
        gap: 4px;
    }
    div.row-widget.stRadio > div > label {
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 0;
        transition: all 0.2s ease;
    }
    div.row-widget.stRadio > div > label:hover {
        background-color: #E0E7FF;
        color: #4338CA;
    }
    
    /* Headers typography */
    h1, h2, h3 {
        color: #1E293B;
        font-weight: 700 !important;
    }
    
    /* Dataframes rounded corners & subtle shadow */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
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
    f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">'
    f'<img src="data:image/svg+xml;base64,{_svg_b64}" width="40" height="40">'
    f'<h1 style="margin:0; font-size:1.5rem;">MSDA-Bench</h1>'
    f'</div>',
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

# Show completion in sidebar
completion = store.derived['completion']
n_done = completion.sum().sum()
n_total = completion.size
st.sidebar.metric("Completion", f"{int(n_done)}/{n_total}")
st.sidebar.progress(float(n_done / n_total) if n_total > 0 else 0.0)

if st.sidebar.button("Refresh Data"):
    st.cache_resource.clear()
    st.rerun()

# Page navigation using tabs (simpler than st.navigation for compatibility)
pages = {
    "Overview": page_1_overview,
    "Pipeline Benchmark": page_2_benchmark,
    "Stability & Sensitivity": page_3_stability,
    "Config Explorer": page_4_config,
    "Subject Explorer": page_5_subject,
    "DA Analysis": page_6_da,
    "Mechanism Explorer": page_7_mechanism,
    "Target Session": page_8_target,
    "Prediction Error": page_9_error,
    "Efficiency & Progress": page_10_efficiency,
}

page_name = st.sidebar.radio("Page", list(pages.keys()))
pages[page_name].render(store, dataset)

# Footer: author info
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.75rem; color:#94A3B8; line-height:1.4;">'
    '<b>Yiming Shen</b> &amp; <b>David Degras</b><br>'
    'Department of Mathematics<br>'
    'University of Massachusetts Boston'
    '</div>',
    unsafe_allow_html=True
)
