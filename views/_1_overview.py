"""Page 1: Dataset Overview."""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import os
from utils import PIPE_ORDER, style_figure


def render(store, dataset):
    st.header("Dataset Overview")
    st.markdown(
        "Interactive dashboard for comparing cross-session EEG classification pipelines "
        "under multi-source domain adaptation."
    )

    df = store.summary_df
    completion = store.derived['completion']
    qc = store.derived['qc_results']
    sp = store.derived['subject_pipeline']

    subjects = sorted(df['subject'].unique())
    n_sessions = df.groupby('subject')['n_session'].first()
    pipes_present = sorted(df['pipe_short'].unique())
    n_configs = len(df[(df['subject'] == subjects[0]) & (df['pipe_short'] == df['pipe_short'].iloc[0])]) if len(df) > 0 else 0

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    total_cells = len(subjects) * len(PIPE_ORDER)
    completed_cells = int(completion.sum().sum())
    completion_pct = completed_cells / total_cells * 100 if total_cells > 0 else 0

    # Check if BDP degradation data is available
    deg_df = store.derived.get("degradation", pd.DataFrame())
    has_bdp_degrade = not deg_df.empty

    if has_bdp_degrade:
        c1, c2, c3, c4, c5 = st.columns(5)
    else:
        c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects", len(subjects))
    c2.metric("Pipelines", len(pipes_present))
    c3.metric("Configs / Pipeline", n_configs)
    c4.metric("Completion", f"{completion_pct:.0f}%")
    if has_bdp_degrade:
        mean_deg = deg_df["degraded_ratio"].mean()
        c5.metric("BDP Degrade Rate", f"{mean_deg:.0%}")

    # ── Takeaway ──────────────────────────────────────────────────────────────
    if not sp.empty:
        best = sp.groupby('pipe_short')['M_acc'].mean()
        if not best.empty:
            best_pipe = best.idxmax()
            best_val = best.max()
            st.success(
                f"**{best_pipe}** leads with mean accuracy **{best_val:.4f}** "
                f"across all subjects on M(s,p)."
            )

    st.markdown("")

    # ── Completion Matrix & Matched Subjects ──────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        with st.container(border=True):
            st.subheader("Completion Matrix")
            st.caption("Each cell: one subject x pipeline combination.")
            comp_numeric = completion.astype(int)
            fig = px.imshow(
                comp_numeric,
                labels=dict(x="Pipeline", y="Subject", color="Complete"),
                x=completion.columns.tolist(),
                y=[f"S{s}" for s in completion.index],
                color_continuous_scale=[[0, '#FCA5A5'], [1, '#86EFAC']],
                zmin=0, zmax=1, text_auto=True, aspect='auto',
            )
            style_figure(fig, height=max(300, len(subjects) * 40))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        with st.container(border=True):
            st.subheader("Matched Subjects")
            st.caption("Subjects with data for ALL 6 pipelines.")
            matched_all = store.get_matched_subjects(list(completion.columns))
            st.metric("Full Coverage", f"{len(matched_all)} subjects")
            if matched_all:
                st.markdown(', '.join(f'**S{s}**' for s in matched_all))

        with st.container(border=True):
            st.subheader("Data Freshness")
            try:
                newest = max(
                    os.path.getmtime(os.path.join(store.data_dir, f))
                    for f in os.listdir(store.data_dir) if f.endswith('.pkl')
                )
                from datetime import datetime
                st.markdown(
                    f"Last updated: **{datetime.fromtimestamp(newest).strftime('%Y-%m-%d %H:%M')}**"
                )
            except Exception:
                st.markdown("Unable to determine.")

    # ── Experiment Details (expander) ─────────────────────────────────────────
    with st.expander("Experiment Details"):
        summary_data = {
            'Item': [
                'Dataset', 'Subjects', 'Sessions/subject', 'Pipelines',
                'Configurations/pipeline', 'Features', 'Classifiers', 'DA methods',
                'Total detail records',
            ],
            'Value': [
                dataset,
                f"{len(subjects)} ({', '.join(f'S{s}' for s in subjects[:5])}{'...' if len(subjects) > 5 else ''})",
                ', '.join(str(v) for v in sorted(n_sessions.unique())),
                ', '.join(sorted(df['pipe_short'].unique())),
                str(n_configs),
                ', '.join(sorted(df['feature'].unique())),
                ', '.join(sorted(df['classifier'].unique())),
                ', '.join(sorted(df['da'].unique())),
                str(len(store.detail_df)) if store._detail_df is not None else 'Not loaded yet',
            ],
        }
        st.table(pd.DataFrame(summary_data))

    # ── QC Checks (expander) ──────────────────────────────────────────────────
    with st.expander("QC Checks"):
        if len(qc) > 0:
            for _, row in qc.iterrows():
                severity = row.get('severity', '')
                if row['status'] == 'PASS':
                    icon = "✅"
                elif severity in ('WARNING', 'warning'):
                    icon = "⚠️"
                else:
                    icon = "❌"
                st.markdown(f"{icon} **{row['check_name']}**: {row['message']}")
        else:
            st.success("All QC checks passed.")
