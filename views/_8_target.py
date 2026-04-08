"""Page 8: Target Session Analysis."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, style_figure


def render(store, dataset):
    st.header("Target Session Difficulty")
    st.markdown(
        "Which target sessions are hard to predict? See if early sessions are "
        "systematically harder and whether pipelines differ in session-level strengths."
    )
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

        acc_col = "acc_DA" if "acc_DA" in ddf.columns else "cvMeanAcc"
        if acc_col not in ddf.columns:
            st.warning(f"Accuracy column '{acc_col}' not found.")
            return

        target_acc = ddf.groupby(["pipe_short", target_col])[acc_col].mean().reset_index()
        target_acc = target_acc[target_acc["pipe_short"].isin(pipes)]

        # Pre-compute session difficulty for takeaway
        sess_diff = ddf.groupby(target_col)[acc_col].agg(["mean", "std", "count"]).reset_index()
        sess_diff.columns = [target_col, "mean_acc", "sd_acc", "n_folds"]
        sess_diff = sess_diff.sort_values("mean_acc", ascending=True)

        # ── Takeaway ──────────────────────────────────────────────────────────
        if not sess_diff.empty:
            hardest = sess_diff.iloc[0]
            easiest = sess_diff.iloc[-1]
            st.success(
                f"Hardest session: **{hardest[target_col]}** "
                f"(acc: {hardest['mean_acc']:.4f}). "
                f"Easiest: **{easiest[target_col]}** "
                f"(acc: {easiest['mean_acc']:.4f})."
            )

        # ── Heatmap + Grouped bar (side by side) ─────────────────────────────
        col_hm, col_bar = st.columns(2)

        with col_hm:
            with st.container(border=True):
                st.subheader("Accuracy by Target Session")
                st.caption("Dark = easy, light = hard.")
                pivot = target_acc.pivot_table(
                    index=target_col, columns="pipe_short",
                    values=acc_col, aggfunc="mean",
                )
                pivot = pivot.reindex(columns=[p for p in PIPE_ORDER if p in pivot.columns])
                if not pivot.empty:
                    fig = px.imshow(
                        pivot.values,
                        x=pivot.columns.tolist(),
                        y=[str(s) for s in pivot.index.tolist()],
                        color_continuous_scale="RdYlGn",
                        text_auto=".3f", aspect="auto",
                    )
                    fig.update_layout(title="Mean Accuracy per Target x Pipeline")
                    style_figure(fig)
                    st.plotly_chart(fig, use_container_width=True)

        with col_bar:
            with st.container(border=True):
                st.subheader("Per-Target Comparison")
                st.caption("Grouped bars comparing pipelines on each target session.")
                fig = px.bar(
                    target_acc, x=target_col, y=acc_col, color="pipe_short",
                    barmode="group", color_discrete_map=PIPE_COLORS,
                    category_orders={"pipe_short": pipes},
                )
                fig.update_layout(
                    title="Accuracy by Target Session",
                    yaxis_title="Mean Accuracy",
                )
                style_figure(fig)
                st.plotly_chart(fig, use_container_width=True)

        # ── Session difficulty ranking ────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Session Difficulty Ranking")
            st.caption("Sorted by mean accuracy (hardest first).")
            display_diff = sess_diff.rename(columns={
                "mean_acc": "Mean Accuracy", "sd_acc": "SD", "n_folds": "# Folds",
            })
            st.dataframe(
                display_diff.style.format({"Mean Accuracy": "{:.4f}", "SD": "{:.4f}"}),
                hide_index=True, use_container_width=True,
            )

        # ── Pipeline x session interaction line chart ─────────────────────────
        with st.container(border=True):
            st.subheader("Pipeline x Session Interaction")
            st.caption(
                "Crossing lines indicate pipelines that excel on different sessions."
            )
            if not target_acc.empty:
                fig = px.line(
                    target_acc, x=target_col, y=acc_col, color="pipe_short",
                    color_discrete_map=PIPE_COLORS, markers=True,
                    category_orders={"pipe_short": pipes},
                )
                fig.update_layout(
                    title="Pipeline Accuracy Across Target Sessions",
                    yaxis_title="Mean Accuracy",
                )
                style_figure(fig)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering Target Session page: {e}")
