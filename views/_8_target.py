"""Page 8: Target Session Analysis."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS


def render(store, dataset):
    st.header("8. Target Session Analysis")
    st.markdown("Which target sessions are hard to predict? See if early sessions (fewer trials) are systematically harder and whether pipelines differ in session-level strengths.")
    try:
        ddf = store.detail_df
        if ddf.empty:
            st.warning("No detail data available. Ensure detail pkl files are present.")
            return

        pipes = [p for p in PIPE_ORDER if p in ddf["pipe_short"].unique()]

        # Identify target session column
        target_col = None
        for c in ["test_label", "target_session", "test_session"]:
            if c in ddf.columns:
                target_col = c
                break
        if target_col is None:
            st.warning("No target session identifier column found in detail data.")
            return

        # Compute per-target accuracy
        acc_col = "acc_DA" if "acc_DA" in ddf.columns else "cvMeanAcc"
        if acc_col not in ddf.columns:
            st.warning(f"Accuracy column '{acc_col}' not found.")
            return

        target_acc = ddf.groupby(["pipe_short", target_col])[acc_col].mean().reset_index()
        target_acc = target_acc[target_acc["pipe_short"].isin(pipes)]

        # --- Accuracy by target session heatmap ---
        st.subheader("Accuracy by Target Session")
        st.caption("Heatmap of mean accuracy when each session is used as the prediction target. Dark cells = easy sessions, light cells = hard sessions.")
        pivot = target_acc.pivot_table(index=target_col, columns="pipe_short",
                                       values=acc_col, aggfunc="mean")
        pivot = pivot.reindex(columns=[p for p in PIPE_ORDER if p in pivot.columns])
        if not pivot.empty:
            fig = px.imshow(pivot.values,
                            x=pivot.columns.tolist(),
                            y=[str(s) for s in pivot.index.tolist()],
                            color_continuous_scale="RdYlGn",
                            text_auto=".3f", aspect="auto")
            fig.update_layout(title="Mean Accuracy per Target Session x Pipeline",
                              template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each cell shows mean accuracy when that session is the target (test set).")

        # --- Grouped bar per target session ---
        st.subheader("Per-Target Session Comparison")
        st.caption("Grouped bar chart comparing pipelines on each target session. Look for sessions where pipeline rankings differ from the overall ranking.")
        fig = px.bar(target_acc, x=target_col, y=acc_col, color="pipe_short",
                     barmode="group", color_discrete_map=PIPE_COLORS,
                     category_orders={"pipe_short": pipes})
        fig.update_layout(title="Accuracy by Target Session",
                          yaxis_title="Mean Accuracy", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Grouped bars compare pipeline performance for each target session.")

        # --- Session difficulty ranking ---
        st.subheader("Session Difficulty Ranking")
        st.caption("Sessions ranked by mean accuracy across all pipelines. Hardest sessions appear first.")
        sess_diff = ddf.groupby(target_col)[acc_col].agg(["mean", "std", "count"]).reset_index()
        sess_diff.columns = [target_col, "mean_acc", "sd_acc", "n_folds"]
        sess_diff = sess_diff.sort_values("mean_acc", ascending=True)
        st.dataframe(sess_diff.style.format({"mean_acc": "{:.4f}", "sd_acc": "{:.4f}"}),
                     hide_index=True, use_container_width=True)
        st.caption("Sessions sorted by mean accuracy (hardest first). Hard sessions have lower mean accuracy.")

        # --- Pipeline x session interaction line chart ---
        st.subheader("Pipeline x Session Interaction")
        st.caption("Line chart showing each pipeline's accuracy across target sessions. Crossing lines indicate that pipeline rankings change depending on which session is tested.")
        if not target_acc.empty:
            fig = px.line(target_acc, x=target_col, y=acc_col, color="pipe_short",
                          color_discrete_map=PIPE_COLORS, markers=True,
                          category_orders={"pipe_short": pipes})
            fig.update_layout(title="Pipeline Accuracy Across Target Sessions",
                              yaxis_title="Mean Accuracy", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Lines crossing indicate pipelines that excel on different sessions, suggesting complementary strengths.")

    except Exception as e:
        st.error(f"Error rendering Target Session page: {e}")
