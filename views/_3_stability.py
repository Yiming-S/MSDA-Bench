"""Page 3: Stability & Sensitivity."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, COOL_LIGHT_SEQUENTIAL, style_figure


def render(store, dataset):
    st.header("Selection Sensitivity")
    st.markdown(
        "How robust are the results? Are pipeline rankings fragile, and how much "
        "does picking the right configuration matter?"
    )
    try:
        sdf = store.summary_df
        sp = store.derived["subject_pipeline"]
        cfg = store.derived["config_agg"]
        if sdf.empty or sp.empty:
            st.warning("No data available for stability analysis.")
            return

        pipes = [p for p in PIPE_ORDER if p in sp["pipe_short"].unique()]

        # ── Takeaway ──────────────────────────────────────────────────────────
        if not sp.empty and "sd_acc" in sp.columns:
            stability = sp.groupby("pipe_short")["sd_acc"].mean()
            if not stability.empty:
                most_stable = stability.idxmin()
                least_stable = stability.idxmax()
                st.success(
                    f"Most stable: **{most_stable}** (mean config SD: {stability.min():.4f}). "
                    f"Most sensitive: **{least_stable}** (SD: {stability.max():.4f})."
                )

        # ── Top-2 Configuration Separation ────────────────────────────────────
        with st.container(border=True):
            st.subheader("Top-2 Configuration Separation")
            st.caption(
                "ECDF: fraction of subjects with a gap <= x between best and "
                "second-best configuration accuracy."
            )
            gaps = []
            for (subj, pipe), grp in sdf.groupby(["subject", "pipe_short"]):
                accs = grp["cvMeanAcc"].dropna().sort_values(ascending=False)
                if len(accs) >= 2:
                    gaps.append({"subject": subj, "pipe_short": pipe,
                                 "gap": accs.iloc[0] - accs.iloc[1]})
            if gaps:
                gap_df = pd.DataFrame(gaps)

                fig = go.Figure()
                for p in pipes:
                    pdata = gap_df[gap_df["pipe_short"] == p]["gap"].sort_values().values
                    if len(pdata) == 0:
                        continue
                    ecdf_y = np.arange(1, len(pdata) + 1) / len(pdata)
                    fig.add_trace(go.Scatter(
                        x=pdata, y=ecdf_y, mode="lines+markers",
                        name=p, line=dict(color=PIPE_COLORS.get(p, "#888"), width=2),
                        marker=dict(size=5),
                    ))
                for thresh, label in [(0.005, "Almost tied"), (0.01, "Weakly separated")]:
                    fig.add_vline(
                        x=thresh, line_dash="dot", line_color="#94A3B8",
                        annotation_text=label, annotation_position="top right",
                        annotation_font_size=10, annotation_font_color="#64748B",
                    )
                fig.update_layout(
                    title="ECDF of Best - 2nd Best Accuracy Gap",
                    xaxis_title="Accuracy Gap (best - 2nd best)",
                    yaxis_title="Cumulative Fraction of Subjects",
                    yaxis=dict(range=[0, 1.05]),
                )
                style_figure(fig)
                st.plotly_chart(fig, use_container_width=True)

                # Subject x Pipeline gap heatmap
                gap_pivot = gap_df.pivot(index="subject", columns="pipe_short", values="gap")
                gap_pivot = gap_pivot[[p for p in PIPE_ORDER if p in gap_pivot.columns]]
                gap_pivot.index = [f"S{s}" for s in gap_pivot.index]

                fig_hm = px.imshow(
                    gap_pivot.values,
                    x=gap_pivot.columns.tolist(), y=gap_pivot.index.tolist(),
                    color_continuous_scale=COOL_LIGHT_SEQUENTIAL,
                    text_auto=".4f", aspect="auto", zmin=0,
                    zmax=float(gap_pivot.values[np.isfinite(gap_pivot.values)].max())
                    if np.any(np.isfinite(gap_pivot.values)) else 0.05,
                )
                fig_hm.update_layout(
                    title="Top-2 Gap by Subject x Pipeline",
                    coloraxis_colorbar_title="Gap",
                )
                style_figure(fig_hm, height=max(300, len(gap_pivot) * 35))
                st.plotly_chart(fig_hm, use_container_width=True)
                st.caption(
                    "Darker = larger gap (decisive winner). Light = nearly tied top-2."
                )

        # ── Config selection premium (B - M) ──────────────────────────────────
        with st.container(border=True):
            st.subheader("Configuration Selection Premium (B - M)")
            st.caption(
                "How much does knowing the best configuration help? "
                "Large premium = pipeline depends heavily on picking the right config."
            )
            sp_copy = sp.copy()
            sp_copy["premium"] = sp_copy["B_acc"] - sp_copy["M_acc"]
            sp_filt = sp_copy[sp_copy["pipe_short"].isin(pipes)]
            fig = px.box(
                sp_filt, x="pipe_short", y="premium", color="pipe_short",
                color_discrete_map=PIPE_COLORS,
                category_orders={"pipe_short": pipes},
            )
            fig.update_layout(
                title="Oracle Selection Premium per Pipeline",
                yaxis_title="B(s,p) - M(s,p)", showlegend=False,
            )
            style_figure(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── Ranking stability across metrics ──────────────────────────────────
        with st.container(border=True):
            st.subheader("Ranking Stability Across Metrics")
            st.caption(
                "Does the pipeline ranking change between mean-over-configurations "
                "and oracle-best?"
            )
            metrics = ["M_acc", "B_acc", "G_gain", "H_helps"]
            rank_rows = []
            for m in metrics:
                agg = sp.groupby("pipe_short")[m].mean()
                ranked = agg.rank(ascending=False)
                for p in pipes:
                    if p in ranked.index:
                        rank_rows.append({"pipe_short": p, "metric": m, "rank": ranked[p]})
            if rank_rows:
                rank_df = pd.DataFrame(rank_rows)
                pivot = rank_df.pivot(index="pipe_short", columns="metric", values="rank")
                pivot = pivot.reindex([p for p in PIPE_ORDER if p in pivot.index])
                fig = px.imshow(
                    pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                    text_auto=".1f", color_continuous_scale="YlOrRd", aspect="auto",
                )
                fig.update_layout(title="Pipeline Rank by Metric (1=best)")
                style_figure(fig)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Consistent ranks = robustly strong/weak pipeline.")

        # ── Config variance per pipeline ──────────────────────────────────────
        with st.container(border=True):
            st.subheader("Configuration Variance per Pipeline")
            st.caption(
                "High within-subject config SD = needs careful tuning. "
                "Low SD = works with most configurations."
            )
            if not sp.empty:
                sp_filt = sp[sp["pipe_short"].isin(pipes)].copy()
                sp_filt["order"] = sp_filt["pipe_short"].map(
                    {p: i for i, p in enumerate(PIPE_ORDER)}
                )
                sp_filt = sp_filt.sort_values("order")

                tab_box, tab_hm, tab_bar = st.tabs(
                    ["Box Plot", "Per-Subject Heatmap", "Summary Bar"]
                )

                with tab_box:
                    fig = px.box(
                        sp_filt, x="pipe_short", y="sd_acc", color="pipe_short",
                        color_discrete_map=PIPE_COLORS,
                        category_orders={"pipe_short": pipes},
                        points="all", hover_data=["subject"],
                    )
                    fig.update_layout(
                        yaxis_title="Within-Subject Configuration SD",
                        showlegend=False,
                    )
                    style_figure(fig)
                    st.plotly_chart(fig, use_container_width=True)

                with tab_hm:
                    pivot_sd = sp_filt.pivot(
                        index="subject", columns="pipe_short", values="sd_acc",
                    )
                    pivot_sd = pivot_sd[[p for p in PIPE_ORDER if p in pivot_sd.columns]]
                    pivot_sd.index = [f"S{s}" for s in pivot_sd.index]
                    fig = px.imshow(
                        pivot_sd.values,
                        x=pivot_sd.columns.tolist(), y=pivot_sd.index.tolist(),
                        color_continuous_scale="YlOrRd",
                        text_auto=".3f", aspect="auto",
                        zmin=float(pivot_sd.values[np.isfinite(pivot_sd.values)].min())
                        if np.any(np.isfinite(pivot_sd.values)) else 0,
                        zmax=float(pivot_sd.values[np.isfinite(pivot_sd.values)].max())
                        if np.any(np.isfinite(pivot_sd.values)) else 0.2,
                    )
                    fig.update_layout(coloraxis_colorbar_title="SD")
                    style_figure(fig, height=max(300, len(pivot_sd) * 35))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Red = high sensitivity. Yellow = low sensitivity.")

                with tab_bar:
                    var_agg = sp_filt.groupby("pipe_short")["sd_acc"].agg(
                        ["mean", "std"]
                    ).reset_index()
                    var_agg["order"] = var_agg["pipe_short"].map(
                        {p: i for i, p in enumerate(PIPE_ORDER)}
                    )
                    var_agg = var_agg.sort_values("order")
                    fig = go.Figure()
                    for _, row in var_agg.iterrows():
                        fig.add_trace(go.Bar(
                            x=[row["pipe_short"]], y=[row["mean"]],
                            error_y=dict(type="data", array=[row["std"]], visible=True),
                            marker_color=PIPE_COLORS.get(row["pipe_short"], "#888"),
                            text=[f'{row["mean"]:.4f}'], textposition="outside",
                            name=row["pipe_short"], showlegend=False,
                        ))
                    fig.update_layout(yaxis_title="Mean Within-Subject Config SD")
                    style_figure(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Lower = more robust to configuration choice.")

        # ── Most stable configurations ────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Most Stable Configurations")
            st.caption("Configurations with lowest cross-subject SD (min 3 subjects).")
            if not cfg.empty and "sd_acc" in cfg.columns:
                stable = cfg[cfg["n_subject"] >= 3].nsmallest(5, "sd_acc")
                display_cols = [
                    c for c in ["pipe_short", "config_label", "mean_acc", "sd_acc", "n_subject"]
                    if c in stable.columns
                ]
                st.dataframe(
                    stable[display_cols]
                    .rename(columns={
                        "pipe_short": "Pipeline", "config_label": "Configuration",
                        "mean_acc": "Mean Accuracy", "sd_acc": "SD",
                        "n_subject": "# Subjects",
                    })
                    .style.format({"Mean Accuracy": "{:.4f}", "SD": "{:.4f}"}),
                    hide_index=True, use_container_width=True,
                )

    except Exception as e:
        st.error(f"Error rendering Stability page: {e}")
