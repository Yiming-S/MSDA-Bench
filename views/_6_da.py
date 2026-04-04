"""Page 6: DA Analysis."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS


def render(store, dataset):
    st.header("6. DA (Domain Adaptation) Analysis")
    st.markdown("Does domain adaptation actually help? Break down DA gain by pipeline, feature, and DA method to find what works and what harms.")
    try:
        sdf = store.summary_df
        sp = store.derived["subject_pipeline"]
        cfg = store.derived["config_agg"]
        if sdf.empty:
            st.warning("No data available.")
            return

        pipes = [p for p in PIPE_ORDER if p in sdf["pipe_short"].unique()]

        # --- Overall DA gain bar chart per pipeline ---
        st.subheader("Overall DA Gain per Pipeline")
        st.caption("Mean DA lift (acc_DA - baseline) per pipeline, averaged across all configs and subjects. Positive = DA helps on average, negative = DA hurts.")
        if not sp.empty and "G_gain" in sp.columns:
            gain_agg = sp[sp["pipe_short"].isin(pipes)].groupby("pipe_short")["G_gain"].agg(
                ["mean", "std"]).reset_index()
            gain_agg["order"] = gain_agg["pipe_short"].map({p: i for i, p in enumerate(PIPE_ORDER)})
            gain_agg = gain_agg.sort_values("order")
            fig = go.Figure()
            for _, row in gain_agg.iterrows():
                fig.add_trace(go.Bar(
                    x=[row["pipe_short"]], y=[row["mean"]],
                    error_y=dict(type="data", array=[row["std"]], visible=True),
                    marker_color=PIPE_COLORS.get(row["pipe_short"], "#888"),
                    name=row["pipe_short"], showlegend=False,
                    text=[f"{row['mean']:.4f}"], textposition="outside"))
            fig.update_layout(title="Mean DA Gain (cvMeanAcc - baseline)",
                              yaxis_title="Gain", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Positive gain means DA improves over baseline on average. Error bars = 1 SD across subjects.")

        # --- DA gain heatmap: config x pipeline ---
        st.subheader("DA Gain Heatmap (Config x Pipeline)")
        st.caption("Per-config DA gain matrix. Red cells = DA hurts this config, green = DA helps. Identifies dangerous feature+DA combinations (e.g., logvar+coral).")
        if not cfg.empty and "mean_gain" in cfg.columns:
            top_cfgs = cfg.groupby("config_label")["mean_gain"].mean().nlargest(20).index
            filt = cfg[cfg["config_label"].isin(top_cfgs) & cfg["pipe_short"].isin(pipes)]
            pivot = filt.pivot_table(index="config_label", columns="pipe_short",
                                     values="mean_gain", aggfunc="first")
            pivot = pivot.reindex(columns=[p for p in PIPE_ORDER if p in pivot.columns])
            if not pivot.empty:
                fig = px.imshow(pivot.values, x=pivot.columns.tolist(),
                                y=pivot.index.tolist(), color_continuous_scale="RdBu",
                                color_continuous_midpoint=0, text_auto=".3f", aspect="auto")
                fig.update_layout(title="Top 20 Configs: Mean DA Gain",
                                  template="plotly_white", height=max(400, len(top_cfgs) * 22))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Blue = positive gain (DA helps); red = negative (DA hurts). Diverging from zero.")

        # --- Violin distribution of per-subject DA gains ---
        st.subheader("DA Gain Distribution")
        st.caption("Distribution of subject-level mean DA gains per pipeline. Wide violins indicate high variability across subjects.")
        if not sp.empty and "G_gain" in sp.columns:
            violin_data = sp[sp["pipe_short"].isin(pipes)].copy()
            fig = px.violin(violin_data, x="pipe_short", y="G_gain", color="pipe_short",
                            color_discrete_map=PIPE_COLORS, box=True, points="all",
                            category_orders={"pipe_short": pipes})
            fig.update_layout(title="Per-Subject DA Gain Distribution",
                              yaxis_title="G_gain", showlegend=False, template="plotly_white")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each point is one subject. Points below zero indicate DA hurt performance.")

        # --- Feature x DA interaction ---
        st.subheader("Feature x DA Interaction")
        st.caption("Does SA help CSP but hurt TS? This chart shows mean DA gain for each feature+DA combination, revealing interaction effects.")
        if "feature" in sdf.columns and "da" in sdf.columns and "baseline" in sdf.columns:
            sdf_copy = sdf.copy()
            sdf_copy["gain"] = sdf_copy["cvMeanAcc"] - sdf_copy["baseline"]
            feat_da = sdf_copy.groupby(["feature", "da"])["gain"].mean().reset_index()
            fig = px.bar(feat_da, x="feature", y="gain", color="da", barmode="group")
            fig.update_layout(title="Mean Gain by Feature x DA Method",
                              yaxis_title="Mean Gain", template="plotly_white")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Shows which feature-DA combinations yield positive or negative adaptation gains.")

        # --- Harm rate table ---
        st.subheader("Harm Rate by Pipeline")
        st.caption("Fraction of config-fold combinations where DA strictly reduces accuracy. High harm rate suggests DA should be used cautiously with this pipeline.")
        if not cfg.empty and "helps_rate" in cfg.columns:
            harm = cfg[cfg["pipe_short"].isin(pipes)].groupby("pipe_short").agg(
                mean_helps=("helps_rate", "mean"),
                min_helps=("helps_rate", "min"),
                n_configs=("config_label", "count")).reset_index()
            harm["harm_rate"] = 1 - harm["mean_helps"]
            harm["order"] = harm["pipe_short"].map({p: i for i, p in enumerate(PIPE_ORDER)})
            harm = harm.sort_values("order")
            st.dataframe(harm[["pipe_short", "mean_helps", "harm_rate", "min_helps", "n_configs"]].style.format(
                {"mean_helps": "{:.3f}", "harm_rate": "{:.3f}", "min_helps": "{:.3f}"}),
                hide_index=True, use_container_width=True)
            st.caption("Harm rate = fraction of configs where DA reduces accuracy below baseline.")

    except Exception as e:
        st.error(f"Error rendering DA Analysis page: {e}")
