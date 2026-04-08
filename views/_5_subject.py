"""Page 5: Subject Explorer."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, format_acc


def render(store, dataset):
    st.header("5. Subject Explorer")
    st.markdown("Deep-dive into a single subject. See all 24 configurations across all pipelines and understand why this subject behaves differently from the group.")
    try:
        sdf = store.summary_df
        sp = store.derived["subject_pipeline"]
        if sdf.empty or sp.empty:
            st.warning("No data available.")
            return

        subjects = sorted(sdf["subject"].unique())
        subj = st.sidebar.selectbox("Subject", subjects, format_func=lambda s: f"S{s}")

        pipes = [p for p in PIPE_ORDER if p in sdf["pipe_short"].unique()]
        subj_data = sdf[sdf["subject"] == subj]

        # --- Full 24x6 accuracy table ---
        st.subheader(f"Accuracy Table for S{subj}")
        if not subj_data.empty:
            pivot = subj_data.pivot_table(index="config_label", columns="pipe_short",
                                          values="cvMeanAcc", aggfunc="first")
            pivot = pivot.reindex(columns=[p for p in PIPE_ORDER if p in pivot.columns])
            fmt = {c: "{:.4f}" for c in pivot.columns}
            st.dataframe(pivot.style.format(fmt, na_rep="---").background_gradient(
                cmap="RdYlGn", axis=None), width="stretch")
            st.caption("Each row is a configuration (feature/classifier/DA); each column is a pipeline.")
        else:
            st.info(f"No data for subject S{subj}.")

        # --- Pipeline summary bar chart ---
        st.subheader(f"Pipeline Summary for S{subj}")
        sp_subj = sp[sp["subject"] == subj]
        if not sp_subj.empty:
            sp_ord = sp_subj.set_index("pipe_short").reindex(
                [p for p in PIPE_ORDER if p in sp_subj["pipe_short"].values]).reset_index()
            metrics = ["M_acc", "B_acc"]
            fig = go.Figure()
            for m in metrics:
                if m in sp_ord.columns:
                    fig.add_trace(go.Bar(
                        x=sp_ord["pipe_short"], y=sp_ord[m], name=m,
                        text=sp_ord[m].apply(lambda v: f"{v:.4f}"),
                        textposition="outside"))
            fig.update_layout(title=f"S{subj}: Mean vs Best Accuracy",
                              yaxis_title="Accuracy", barmode="group",
                              template="plotly_white")
            st.plotly_chart(fig, width="stretch")
            st.caption("M_acc = mean across all configurations; B_acc = best single configuration (oracle).")

            # Additional metrics table
            disp_cols = [c for c in ["pipe_short", "M_acc", "B_acc", "G_gain", "H_helps",
                                      "n_cfg", "best_feature", "best_classifier", "best_da"]
                        if c in sp_ord.columns]
            rename_map = {"pipe_short": "Pipeline", "M_acc": "Mean Accuracy",
                          "B_acc": "Best Accuracy", "G_gain": "Mean DA Gain",
                          "H_helps": "DA Helps Rate", "n_cfg": "# Configurations",
                          "best_feature": "Best Feature", "best_classifier": "Best Classifier",
                          "best_da": "Best DA"}
            disp_df = sp_ord[disp_cols].rename(columns=rename_map)
            fmt2 = {rename_map[c]: "{:.4f}" for c in disp_cols
                     if c in ("M_acc", "B_acc", "G_gain", "H_helps")}
            st.dataframe(disp_df.style.format(fmt2),
                         hide_index=True, width="stretch")

        # --- Best vs mean gap per pipeline ---
        st.subheader(f"Best vs Mean Gap for S{subj}")
        if not sp_subj.empty:
            sp_gap = sp_subj.copy()
            sp_gap["gap"] = sp_gap["B_acc"] - sp_gap["M_acc"]
            sp_gap = sp_gap[sp_gap["pipe_short"].isin(pipes)]
            sp_gap["order"] = sp_gap["pipe_short"].map({p: i for i, p in enumerate(PIPE_ORDER)})
            sp_gap = sp_gap.sort_values("order")
            fig = px.bar(sp_gap, x="pipe_short", y="gap", color="pipe_short",
                         color_discrete_map=PIPE_COLORS,
                         category_orders={"pipe_short": pipes})
            fig.update_layout(title=f"S{subj}: Oracle Premium (B - M)",
                              yaxis_title="Accuracy Gap", showlegend=False,
                              template="plotly_white")
            st.plotly_chart(fig, width="stretch")
            st.caption("Larger gap means configuration selection matters more for this subject on that pipeline.")

    except Exception as e:
        st.error(f"Error rendering Subject Explorer page: {e}")
