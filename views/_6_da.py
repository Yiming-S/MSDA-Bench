"""Page 6: DA Analysis."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, style_figure


def render(store, dataset):
    st.header("Adaptation Effects")
    st.markdown(
        "Does domain adaptation actually help? Break down DA gain by pipeline, "
        "feature, and DA method to find what works and what harms."
    )
    try:
        sdf = store.summary_df
        sp = store.derived["subject_pipeline"]
        cfg = store.derived["config_agg"]
        if sdf.empty:
            st.warning("No data available.")
            return

        pipes = [p for p in PIPE_ORDER if p in sdf["pipe_short"].unique()]

        # ── Takeaway ──────────────────────────────────────────────────────────
        if not sp.empty and "H_helps" in sp.columns:
            mean_helps = sp["H_helps"].mean()
            mean_gain = sp["G_gain"].mean() if "G_gain" in sp.columns else 0
            st.success(
                f"DA helps in **{mean_helps:.0%}** of configurations on average. "
                f"Mean gain: **{mean_gain:+.4f}**."
            )

        # ── Overall DA gain bar chart ─────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Overall DA Gain per Pipeline")
            st.caption(
                "Mean DA lift (acc_DA - baseline) per pipeline. "
                "Positive = DA helps on average."
            )
            if not sp.empty and "G_gain" in sp.columns:
                gain_agg = sp[sp["pipe_short"].isin(pipes)].groupby("pipe_short")[
                    "G_gain"
                ].agg(["mean", "std"]).reset_index()
                gain_agg["order"] = gain_agg["pipe_short"].map(
                    {p: i for i, p in enumerate(PIPE_ORDER)},
                )
                gain_agg = gain_agg.sort_values("order")
                fig = go.Figure()
                for _, row in gain_agg.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row["pipe_short"]], y=[row["mean"]],
                        error_y=dict(type="data", array=[row["std"]], visible=True),
                        marker_color=PIPE_COLORS.get(row["pipe_short"], "#888"),
                        name=row["pipe_short"], showlegend=False,
                        text=[f"{row['mean']:.4f}"], textposition="outside",
                    ))
                fig.update_layout(
                    title="Mean DA Gain (cvMeanAcc - baseline)",
                    yaxis_title="Gain",
                )
                style_figure(fig)
                st.plotly_chart(fig, use_container_width=True)

        # ── DA gain heatmap + violin (side by side) ───────────────────────────
        col_hm, col_violin = st.columns(2)

        with col_hm:
            with st.container(border=True):
                st.subheader("DA Gain Heatmap")
                st.caption("Config x Pipeline. Blue = helps, red = hurts.")
                if not cfg.empty and "mean_gain" in cfg.columns:
                    top_cfgs = cfg.groupby("config_label")["mean_gain"].mean().nlargest(20).index
                    filt = cfg[cfg["config_label"].isin(top_cfgs) & cfg["pipe_short"].isin(pipes)]
                    pivot = filt.pivot_table(
                        index="config_label", columns="pipe_short",
                        values="mean_gain", aggfunc="first",
                    )
                    pivot = pivot.reindex(columns=[p for p in PIPE_ORDER if p in pivot.columns])
                    if not pivot.empty:
                        fig = px.imshow(
                            pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                            color_continuous_scale="RdBu", color_continuous_midpoint=0,
                            text_auto=".3f", aspect="auto",
                        )
                        fig.update_layout(title="Top 20 Configs: Mean DA Gain")
                        style_figure(fig, height=max(400, len(top_cfgs) * 22))
                        st.plotly_chart(fig, use_container_width=True)

        with col_violin:
            with st.container(border=True):
                st.subheader("DA Gain Distribution")
                st.caption("Per-subject DA gain distribution by pipeline.")
                if not sp.empty and "G_gain" in sp.columns:
                    violin_data = sp[sp["pipe_short"].isin(pipes)].copy()
                    fig = px.violin(
                        violin_data, x="pipe_short", y="G_gain", color="pipe_short",
                        color_discrete_map=PIPE_COLORS, box=True, points="all",
                        category_orders={"pipe_short": pipes},
                    )
                    fig.update_layout(
                        title="Per-Subject DA Gain Distribution",
                        yaxis_title="G_gain", showlegend=False,
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    style_figure(fig, height=max(400, 350))
                    st.plotly_chart(fig, use_container_width=True)

        # ── Feature x DA interaction ──────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Feature x DA Interaction")
            st.caption(
                "Does SA help CSP but hurt TS? Mean DA gain for each "
                "feature+DA combination."
            )
            if "feature" in sdf.columns and "da" in sdf.columns and "baseline" in sdf.columns:
                sdf_copy = sdf.copy()
                sdf_copy["gain"] = sdf_copy["cvMeanAcc"] - sdf_copy["baseline"]
                feat_da = sdf_copy.groupby(["feature", "da"])["gain"].mean().reset_index()
                fig = px.bar(feat_da, x="feature", y="gain", color="da", barmode="group")
                fig.update_layout(
                    title="Mean Gain by Feature x DA Method",
                    yaxis_title="Mean Gain",
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                style_figure(fig)
                st.plotly_chart(fig, use_container_width=True)

        # ── Harm rate table ───────────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Harm Rate by Pipeline")
            st.caption(
                "Fraction of configs where DA strictly reduces accuracy. "
                "High harm rate = use DA cautiously."
            )
            if not cfg.empty and "helps_rate" in cfg.columns:
                harm = cfg[cfg["pipe_short"].isin(pipes)].groupby("pipe_short").agg(
                    mean_helps=("helps_rate", "mean"),
                    min_helps=("helps_rate", "min"),
                    n_configs=("config_label", "count"),
                ).reset_index()
                harm["harm_rate"] = 1 - harm["mean_helps"]
                harm["order"] = harm["pipe_short"].map(
                    {p: i for i, p in enumerate(PIPE_ORDER)},
                )
                harm = harm.sort_values("order")
                st.dataframe(
                    harm[["pipe_short", "mean_helps", "harm_rate", "min_helps", "n_configs"]]
                    .rename(columns={
                        "pipe_short": "Pipeline", "mean_helps": "DA Helps Rate",
                        "harm_rate": "DA Harm Rate", "min_helps": "Min Helps Rate",
                        "n_configs": "# Configurations",
                    })
                    .style.format({
                        "DA Helps Rate": "{:.3f}", "DA Harm Rate": "{:.3f}",
                        "Min Helps Rate": "{:.3f}",
                    }),
                    hide_index=True, use_container_width=True,
                )

    except Exception as e:
        st.error(f"Error rendering DA Analysis page: {e}")
