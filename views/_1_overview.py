import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import os
from utils import COOL_LIGHT_COMPLETION

def render(store, dataset):
    st.header("1. Dataset Overview")
    st.markdown(
        "**MSDA-Bench** (Multi-Source Domain Adaptation Benchmark) is an interactive dashboard "
        "for comparing cross-session EEG classification pipelines. It evaluates how different "
        "source session utilization strategies — uniform pooling (MAP), distance-weighted pooling (DWP), "
        "nearest-source selection (MMP), and hierarchical bridge/far partitioning (BDP) — "
        "affect classification accuracy under session-to-session distribution shift."
    )
    st.markdown(
        "Use the sidebar to switch datasets, then navigate through the pages to explore "
        "pipeline accuracy, config sensitivity, domain adaptation effects, session mechanisms, and timing."
    )

    st.markdown("---")

    df = store.summary_df
    completion = store.derived['completion']
    qc = store.derived['qc_results']

    # Experiment summary
    st.subheader("Experiment Summary")
    subjects = sorted(df['subject'].unique())
    n_sessions = df.groupby('subject')['n_session'].first()
    n_configs = len(df[(df['subject']==subjects[0]) & (df['pipe_short']==df['pipe_short'].iloc[0])]) if len(df) > 0 else 0

    summary_data = {
        'Item': ['Dataset', 'Subjects', 'Sessions/subject', 'Pipelines', 'Configs/pipeline',
                 'Features', 'Classifiers', 'DA methods', 'Total detail records'],
        'Value': [
            dataset,
            f"{len(subjects)} ({', '.join(f'S{s}' for s in subjects[:5])}{'...' if len(subjects)>5 else ''})",
            ', '.join(str(v) for v in sorted(n_sessions.unique())),
            ', '.join(sorted(df['pipe_short'].unique())),
            str(n_configs),
            ', '.join(sorted(df['feature'].unique())),
            ', '.join(sorted(df['classifier'].unique())),
            ', '.join(sorted(df['da'].unique())),
            str(len(store.detail_df)) if store._detail_df is not None else 'Not loaded yet'
        ]
    }
    st.table(pd.DataFrame(summary_data))

    # Completion heatmap
    st.subheader("Completion Matrix")
    st.caption("Blue = completed, pale slate = missing. Each cell represents one subject x pipeline combination.")

    comp_numeric = completion.astype(int)
    fig = px.imshow(comp_numeric,
                    labels=dict(x="Pipeline", y="Subject", color="Complete"),
                    x=completion.columns.tolist(),
                    y=[f"S{s}" for s in completion.index],
                    color_continuous_scale=COOL_LIGHT_COMPLETION,
                    zmin=0, zmax=1,
                    text_auto=True,
                    aspect='auto')
    fig.update_layout(height=max(300, len(subjects)*35))
    st.plotly_chart(fig, use_container_width=True)

    # Matched subjects
    st.subheader("Matched Subject Sets")
    st.caption("For fair pipeline comparison, only subjects with ALL compared pipelines are included.")
    matched_all = store.get_matched_subjects(list(completion.columns))
    st.info(f"All 6 pipelines: {len(matched_all)} subjects -- {', '.join(f'S{s}' for s in matched_all)}")

    # QC Results
    st.subheader("QC Checks")
    if len(qc) > 0:
        for _, row in qc.iterrows():
            icon = "PASS" if row['status'] == 'PASS' else ("WARNING" if row['severity'] == 'WARNING' else "FAIL")
            st.markdown(f"**[{icon}] {row['check_name']}**: {row['message']}")
    else:
        st.success("All QC checks passed.")

    # Data freshness
    st.subheader("Data Freshness")
    try:
        newest = max(os.path.getmtime(os.path.join(store.data_dir, f))
                     for f in os.listdir(store.data_dir) if f.endswith('.pkl'))
        from datetime import datetime
        st.text(f"Last updated: {datetime.fromtimestamp(newest).strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception:
        st.text("Unable to determine data freshness.")
