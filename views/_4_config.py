"""Page 4: Config Explorer."""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS


def render(store, dataset):
    st.header("4. Config Explorer")
    st.markdown("Explore all 24 feature/classifier/DA combinations. Find which configs are universally good and which are situationally useful.")
    try:
        cfg = store.derived["config_agg"]
        if cfg.empty:
            st.warning("No config aggregation data available.")
            return

        pipes = [p for p in PIPE_ORDER if p in cfg["pipe_short"].unique()]

        # Controls in main area
        sort_options = {
            "mean_acc": "Mean accuracy (highest first)",
            "mean_gain": "Mean DA gain (highest first)",
            "sd_acc": "Stability / low variance (lowest SD first)",
        }
        col_sort, col_topn = st.columns(2)
        with col_sort:
            sort_by = st.selectbox("Sort configs by", list(sort_options.keys()),
                                   format_func=lambda k: sort_options[k])
        with col_topn:
            n_total = cfg["config_label"].nunique()
            top_n = st.slider("Top N configs (heatmap only)", 5, n_total, min(15, n_total))

        # --- Config x Pipeline heatmap ---
        st.subheader("Config x Pipeline Heatmap")
        st.caption(f"Showing top {top_n} configs sorted by {sort_options[sort_by]}. "
                   "Each cell is the mean accuracy for that config-pipeline pair across all matched subjects.")

        # Get top configs by chosen metric across all pipelines
        cfg_rank = cfg.groupby("config_label")[sort_by].mean().sort_values(
            ascending=(sort_by == "sd_acc")).head(top_n)
        top_configs = cfg_rank.index.tolist()

        filt = cfg[cfg["config_label"].isin(top_configs) & cfg["pipe_short"].isin(pipes)]
        if filt.empty:
            st.info("No data after filtering.")
            return

        pivot = filt.pivot_table(index="config_label", columns="pipe_short",
                                 values="mean_acc", aggfunc="first")
        pivot = pivot.reindex(columns=[p for p in PIPE_ORDER if p in pivot.columns])
        pivot = pivot.reindex(top_configs)

        fig = px.imshow(pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        color_continuous_scale="RdYlGn",
                        text_auto=".3f", aspect="auto",
                        zmin=pivot.values[np.isfinite(pivot.values)].min() - 0.02 if np.any(np.isfinite(pivot.values)) else 0.5,
                        zmax=pivot.values[np.isfinite(pivot.values)].max() + 0.02 if np.any(np.isfinite(pivot.values)) else 1.0)
        fig.update_layout(template="plotly_white", height=max(400, top_n * 25))
        st.plotly_chart(fig, width="stretch")

        # Feature Contribution is shown in Pipeline Benchmark page

        # --- Detailed config table ---
        st.subheader("Config Details")
        st.caption("Full table of config-level statistics. Sort by accuracy, gain, or stability to identify the best candidates.")
        display_cols = [c for c in ["pipe_short", "config_label", "mean_acc", "median_acc",
                                     "sd_acc", "mean_gain", "helps_rate", "n_subject"]
                       if c in cfg.columns]
        sorted_cfg = cfg[cfg["pipe_short"].isin(pipes)].sort_values(sort_by,
                        ascending=(sort_by == "sd_acc")).head(top_n * len(pipes))
        fmt = {c: "{:.4f}" for c in display_cols if c in ("mean_acc", "median_acc", "sd_acc", "mean_gain", "helps_rate")}
        st.dataframe(sorted_cfg[display_cols].style.format(fmt),
                     hide_index=True, width="stretch")
        st.caption(f"Configs sorted by {sort_by}. Shows top entries per pipeline.")

    except Exception as e:
        st.error(f"Error rendering Config Explorer page: {e}")
