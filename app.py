import streamlit as st
import os
import base64

st.set_page_config(page_title="MSDA-Bench", page_icon="file.svg", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown('''
<style>
    /* Refined spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    /* Metric cards with gradient */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.08);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        color: #4F46E5;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Headers */
    h1 { color: #1E293B; font-weight: 800 !important; }
    h2 { color: #1E293B; font-weight: 700 !important; letter-spacing: -0.3px; }
    h3 { color: #334155; font-weight: 600 !important; }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }

    /* Alert boxes */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Tab buttons */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Expander headers */
    details summary {
        font-weight: 600 !important;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
</style>
''', unsafe_allow_html=True)

# ── Sidebar branding ──────────────────────────────────────────────────────────
_svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file.svg")
with open(_svg_path, "rb") as _f:
    _svg_b64 = base64.b64encode(_f.read()).decode()

st.sidebar.markdown(
    f'''
    <div style="
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; margin-bottom: 20px; padding-bottom: 16px;
        border-bottom: 1px solid #E2E8F0;
    ">
        <img src="data:image/svg+xml;base64,{_svg_b64}" width="80" height="80"
             style="filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.08)); margin-bottom: 8px;">
        <h1 style="
            margin: 0; font-size: 1.5rem; font-weight: 800;
            background: linear-gradient(135deg, #4F46E5, #7C3AED);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        ">MSDA-Bench</h1>
        <p style="
            margin: 4px 0 0 0; font-size: 0.7rem; font-weight: 600;
            color: #64748B; text-align: center; line-height: 1.3;
        ">Multi-Source Domain Adaptation Benchmark</p>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ── Dataset selector ──────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATA_DIRS = {
    name: os.path.join(_BASE, name)
    for name in sorted(os.listdir(_BASE))
    if os.path.isdir(os.path.join(_BASE, name))
}

dataset = st.sidebar.selectbox("Dataset", list(DATA_DIRS.keys()))

from data_loader import get_data_store
store = get_data_store(DATA_DIRS[dataset], dataset)

if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

# ── Import pages ──────────────────────────────────────────────────────────────
from views import (
    page_1_overview, page_2_benchmark, page_3_stability,
    page_4_config, page_5_subject, page_6_da,
    page_7_mechanism, page_8_target, page_9_error, page_10_efficiency,
    page_11_degradation,
)


def _wrap(render_func):
    """Wrap a render(store, dataset) function for st.navigation."""
    def _run():
        render_func(store, dataset)
    return _run


# ── Grouped Navigation ───────────────────────────────────────────────────────
pg = st.navigation({
    "Overview": [
        st.Page(_wrap(page_1_overview.render),
                title="Dataset Overview", icon=":material/dashboard:",
                url_path="overview"),
    ],
    "Performance": [
        st.Page(_wrap(page_2_benchmark.render),
                title="Pipeline Comparison", icon=":material/compare_arrows:",
                url_path="pipeline-comparison"),
        st.Page(_wrap(page_4_config.render),
                title="Configuration Effects", icon=":material/tune:",
                url_path="config-effects"),
        st.Page(_wrap(page_5_subject.render),
                title="Subject Explorer", icon=":material/person_search:",
                url_path="subject-explorer"),
    ],
    "Adaptation": [
        st.Page(_wrap(page_6_da.render),
                title="Adaptation Effects", icon=":material/auto_fix_high:",
                url_path="adaptation-effects"),
        st.Page(_wrap(page_3_stability.render),
                title="Selection Sensitivity", icon=":material/balance:",
                url_path="selection-sensitivity"),
        st.Page(_wrap(page_7_mechanism.render),
                title="Session Mechanisms", icon=":material/settings_suggest:",
                url_path="session-mechanisms"),
        st.Page(_wrap(page_8_target.render),
                title="Target Difficulty", icon=":material/gps_fixed:",
                url_path="target-difficulty"),
        st.Page(_wrap(page_11_degradation.render),
                title="BDP Degradation", icon=":material/warning:",
                url_path="bdp-degradation"),
    ],
    "Diagnostics": [
        st.Page(_wrap(page_9_error.render),
                title="Error Analysis", icon=":material/bug_report:",
                url_path="error-analysis"),
        st.Page(_wrap(page_10_efficiency.render),
                title="Runtime & Efficiency", icon=":material/speed:",
                url_path="runtime-efficiency"),
    ],
})

pg.run()

# ── Author info ───────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.75rem; color:#94A3B8; line-height:1.5;">'
    '<b>Yiming Shen</b> &amp; <b>David Degras</b><br>'
    'Department of Mathematics<br>'
    'University of Massachusetts Boston'
    '</div>',
    unsafe_allow_html=True,
)
