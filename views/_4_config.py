"""Page 4: Config Explorer."""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, style_figure


def render(store, dataset):
    st.header("Configuration Effects")
    st.markdown(
        "Explore all 24 feature/classifier/DA combinations. Find which are "
        "universally good and which are situationally useful."
    )
    try:
        cfg = store.derived["config_agg"]
        if cfg.empty:
            st.warning("No configuration aggregation data available.")
            return

        pipes = [p for p in PIPE_ORDER if p in cfg["pipe_short"].unique()]

        # ── Takeaway ──────────────────────────────────────────────────────────
        if not cfg.empty:
            best = cfg.loc[cfg["mean_acc"].idxmax()]
            st.success(
                f"Top configuration: **{best['config_label']}** on "
                f"**{best['pipe_short']}** with accuracy **{best['mean_acc']:.4f}**"
            )

        # Controls
        sort_options = {
            "mean_acc": "Mean accuracy (highest first)",
            "mean_gain": "Mean DA gain (highest first)",
            "sd_acc": "Stability / low variance (lowest SD first)",
        }
        col_sort, col_topn = st.columns(2)
        with col_sort:
            sort_by = st.selectbox(
                "Sort configurations by", list(sort_options.keys()),
                format_func=lambda k: sort_options[k],
            )
        with col_topn:
            n_total = cfg["config_label"].nunique()
            top_n = st.slider(
                "Top N configurations (heatmap only)", 5, n_total, min(15, n_total),
            )

        # ── Config x Pipeline heatmap ─────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Configuration x Pipeline Heatmap")
            st.caption(
                f"Top {top_n} configurations sorted by {sort_options[sort_by]}. "
                "Each cell = mean accuracy for that config-pipeline pair."
            )
            cfg_rank = cfg.groupby("config_label")[sort_by].mean().sort_values(
                ascending=(sort_by == "sd_acc"),
            ).head(top_n)
            top_configs = cfg_rank.index.tolist()

            filt = cfg[cfg["config_label"].isin(top_configs) & cfg["pipe_short"].isin(pipes)]
            if filt.empty:
                st.info("No data after filtering.")
                return

            pivot = filt.pivot_table(
                index="config_label", columns="pipe_short",
                values="mean_acc", aggfunc="first",
            )
            pivot = pivot.reindex(columns=[p for p in PIPE_ORDER if p in pivot.columns])
            pivot = pivot.reindex(top_configs)

            fig = px.imshow(
                pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                color_continuous_scale="RdYlGn", text_auto=".3f", aspect="auto",
                zmin=pivot.values[np.isfinite(pivot.values)].min() - 0.02
                if np.any(np.isfinite(pivot.values)) else 0.5,
                zmax=pivot.values[np.isfinite(pivot.values)].max() + 0.02
                if np.any(np.isfinite(pivot.values)) else 1.0,
            )
            style_figure(fig, height=max(400, top_n * 25))
            st.plotly_chart(fig, use_container_width=True)

        # ── Detailed config table ─────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Configuration Details")
            st.caption("Full table of configuration-level statistics.")
            display_cols = [
                c for c in [
                    "pipe_short", "config_label", "mean_acc", "median_acc",
                    "sd_acc", "mean_gain", "helps_rate", "n_subject",
                ]
                if c in cfg.columns
            ]
            sorted_cfg = cfg[cfg["pipe_short"].isin(pipes)].sort_values(
                sort_by, ascending=(sort_by == "sd_acc"),
            ).head(top_n * len(pipes))
            rename_map = {
                "pipe_short": "Pipeline", "config_label": "Configuration",
                "mean_acc": "Mean Accuracy", "median_acc": "Median Accuracy",
                "sd_acc": "SD", "mean_gain": "Mean DA Gain",
                "helps_rate": "DA Helps Rate", "n_subject": "# Subjects",
            }
            fmt = {
                rename_map.get(c, c): "{:.4f}"
                for c in display_cols
                if c in ("mean_acc", "median_acc", "sd_acc", "mean_gain", "helps_rate")
            }
            st.dataframe(
                sorted_cfg[display_cols].rename(columns=rename_map).style.format(fmt),
                hide_index=True, use_container_width=True,
                height=min(500, max(200, top_n * len(pipes) * 35)),
            )

    except Exception as e:
        st.error(f"Error rendering Configuration Effects page: {e}")
