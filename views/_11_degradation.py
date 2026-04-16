"""Page 11: BDP Degradation Explorer."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, DEGRADE_COLORS, style_figure


def render(store, dataset):
    st.header("BDP Degradation Explorer")
    st.markdown(
        "When no valid far-session candidate exists, BDP degrades to MAP-style "
        "all-source training. This page analyzes where, why, and how degradation "
        "affects accuracy."
    )
    try:
        sdf = store.summary_df
        ddf = store.detail_df
        deg_df = store.derived.get("degradation", pd.DataFrame())

        if deg_df.empty:
            st.warning("No BDP degradation data available.")
            return

        bdp_pipes = [p for p in ["BDP_fb", "BDP_bf"] if p in sdf["pipe_short"].unique()]
        if not bdp_pipes:
            st.warning("No BDP pipelines found in data.")
            return

        # ── Read query params for drill-down from Subject Explorer ────────
        qp = st.query_params
        init_subject = int(qp.get("subject", 0)) or None
        init_pipe = qp.get("pipe", bdp_pipes[0])
        if init_pipe not in bdp_pipes:
            init_pipe = bdp_pipes[0]

        subjects = sorted(sdf["subject"].unique())

        # ── Controls ──────────────────────────────────────────────────────
        col_pipe, col_mode = st.columns([1, 2])
        with col_pipe:
            pipe = st.selectbox(
                "BDP Pipeline", bdp_pipes,
                index=bdp_pipes.index(init_pipe) if init_pipe in bdp_pipes else 0,
                key="deg_pipe",
            )
        with col_mode:
            view_mode = st.radio(
                "View", ["Global Overview", "Single-Subject Detail"],
                index=1 if init_subject else 0,
                horizontal=True, key="deg_mode",
            )

        pipe_deg = deg_df[deg_df["pipe_short"] == pipe].copy()
        if pipe_deg.empty:
            st.info(f"No degradation data for {pipe}.")
            return

        if view_mode == "Global Overview":
            _render_global(store, pipe, pipe_deg, subjects)
        else:
            _render_single_subject(store, pipe, pipe_deg, subjects, init_subject)

    except Exception as e:
        st.error(f"Error rendering BDP Degradation Explorer: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def _render_global(store, pipe, pipe_deg, subjects):
    """Full-dataset degradation overview."""

    ddf = store.detail_df

    # ── Takeaway ──────────────────────────────────────────────────────────
    total_pairs = pipe_deg["total_pairs"].sum()
    total_degraded = pipe_deg["degraded_pairs"].sum()
    overall_ratio = total_degraded / total_pairs if total_pairs > 0 else 0
    n_subjects = pipe_deg["subject"].nunique()
    st.success(
        f"**{pipe}**: {overall_ratio:.0%} of {total_pairs:,} pairs degraded to MAP "
        f"across {n_subjects} subjects."
    )

    # ── 1. Heatmap + Stacked bar (side by side) ──────────────────────────
    col_hm, col_bar = st.columns(2)

    with col_hm:
        with st.container(border=True):
            st.subheader("Degradation Rate Heatmap")
            st.caption("Subject x Feature. Darker red = more pairs degraded.")
            if "feature" in pipe_deg.columns:
                pivot = pipe_deg.pivot_table(
                    index="subject", columns="feature",
                    values="degraded_ratio", aggfunc="first",
                )
                pivot.index = [f"S{s}" for s in pivot.index]
                fig = px.imshow(
                    pivot.values,
                    x=pivot.columns.tolist(), y=pivot.index.tolist(),
                    color_continuous_scale=[
                        [0, "#F0FDF4"], [0.5, "#FDE68A"], [1, "#EF4444"],
                    ],
                    zmin=0, zmax=1, text_auto=".0%", aspect="auto",
                )
                fig.update_layout(coloraxis_colorbar_title="Degrade %")
                style_figure(fig, height=max(300, len(pivot) * 28))
                st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        with st.container(border=True):
            st.subheader("Per-Subject Degradation")
            st.caption("Stacked bars: pure vs degraded pairs.")
            subj_agg = pipe_deg.groupby("subject").agg(
                total=("total_pairs", "sum"),
                degraded=("degraded_pairs", "sum"),
            ).reset_index()
            subj_agg["pure"] = subj_agg["total"] - subj_agg["degraded"]
            subj_agg["subject_label"] = subj_agg["subject"].apply(lambda s: f"S{s}")
            subj_agg = subj_agg.sort_values("subject")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=subj_agg["subject_label"], y=subj_agg["pure"],
                name="Pure BDP", marker_color=DEGRADE_COLORS["pure"],
            ))
            fig.add_trace(go.Bar(
                x=subj_agg["subject_label"], y=subj_agg["degraded"],
                name="Degraded", marker_color=DEGRADE_COLORS["full_degrade"],
            ))
            fig.update_layout(
                barmode="stack", yaxis_title="# Pairs",
                xaxis_title="Subject", legend_title="Status",
            )
            style_figure(fig, height=max(300, 350))
            st.plotly_chart(fig, use_container_width=True)

    # ── 2. Degradation by feature ────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Degradation by Feature")
        st.caption("Which features trigger degradation most often?")
        if "feature" in pipe_deg.columns:
            feat_agg = pipe_deg.groupby("feature").agg(
                total=("total_pairs", "sum"),
                degraded=("degraded_pairs", "sum"),
            ).reset_index()
            feat_agg["degraded_ratio"] = feat_agg["degraded"] / feat_agg["total"]
            feat_agg = feat_agg.sort_values("degraded_ratio", ascending=False)

            fig = px.bar(
                feat_agg, x="feature", y="degraded_ratio",
                color="degraded_ratio",
                color_continuous_scale=[
                    [0, "#10B981"], [0.5, "#F59E0B"], [1, "#EF4444"],
                ],
                text=feat_agg["degraded_ratio"].apply(lambda v: f"{v:.0%}"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis_title="Degradation Rate", yaxis=dict(tickformat=".0%"),
                xaxis_title="Feature", coloraxis_showscale=False,
            )
            style_figure(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ── 3. Pure vs Degraded accuracy ─────────────────────────────────────
    with st.container(border=True):
        st.subheader("Pure vs Degraded Accuracy")
        st.caption(
            "Does degradation hurt or help? Compare acc_DA distributions."
        )
        if not ddf.empty and "degraded" in ddf.columns and "acc_DA" in ddf.columns:
            bdp_detail = ddf[ddf["pipe_short"] == pipe].copy()
            bdp_detail["status"] = bdp_detail["degraded"].map(
                {True: "Degraded", False: "Pure BDP"},
            )
            if not bdp_detail.empty:
                col_box, col_stats = st.columns([2, 1])
                with col_box:
                    fig = px.box(
                        bdp_detail, x="status", y="acc_DA", color="status",
                        color_discrete_map={
                            "Pure BDP": DEGRADE_COLORS["pure"],
                            "Degraded": DEGRADE_COLORS["full_degrade"],
                        },
                        points="outliers",
                    )
                    fig.update_layout(
                        yaxis_title="Accuracy (acc_DA)", xaxis_title="",
                        showlegend=False,
                    )
                    style_figure(fig)
                    st.plotly_chart(fig, use_container_width=True)
                with col_stats:
                    stats = bdp_detail.groupby("status")["acc_DA"].agg(
                        ["mean", "median", "std", "count"],
                    ).reset_index()
                    stats.columns = ["Status", "Mean", "Median", "SD", "N Pairs"]
                    st.dataframe(
                        stats.style.format({
                            "Mean": "{:.4f}", "Median": "{:.4f}", "SD": "{:.4f}",
                        }),
                        hide_index=True, use_container_width=True,
                    )

    # ── 4. Subject ranking by degradation severity ───────────────────────
    with st.container(border=True):
        st.subheader("Subject Degradation Ranking")
        st.caption("Sorted by degradation ratio. Click a subject to drill down.")
        ranking = pipe_deg.groupby("subject").agg(
            total=("total_pairs", "sum"),
            degraded=("degraded_pairs", "sum"),
            mean_acc_pure=("acc_pure", "mean"),
            mean_acc_degraded=("acc_degraded", "mean"),
        ).reset_index()
        ranking["degraded_ratio"] = ranking["degraded"] / ranking["total"]
        ranking = ranking.sort_values("degraded_ratio", ascending=False)
        ranking.insert(0, "Subject", ranking["subject"].apply(lambda s: f"S{s}"))

        st.dataframe(
            ranking[["Subject", "total", "degraded", "degraded_ratio",
                      "mean_acc_pure", "mean_acc_degraded"]]
            .rename(columns={
                "total": "Total Pairs", "degraded": "Degraded",
                "degraded_ratio": "Degrade %",
                "mean_acc_pure": "Acc (Pure)", "mean_acc_degraded": "Acc (Degraded)",
            })
            .style.format({
                "Degrade %": "{:.1%}",
                "Acc (Pure)": "{:.4f}", "Acc (Degraded)": "{:.4f}",
            }),
            hide_index=True, use_container_width=True,
            height=min(500, max(200, len(ranking) * 35)),
        )


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-SUBJECT DETAIL
# ══════════════════════════════════════════════════════════════════════════════

def _render_single_subject(store, pipe, pipe_deg, subjects, init_subject):
    """Per-subject degradation drill-down."""

    sdf = store.summary_df
    ddf = store.detail_df

    subj = st.selectbox(
        "Subject", subjects,
        index=subjects.index(init_subject) if init_subject in subjects else 0,
        format_func=lambda s: f"S{s}",
        key="deg_subj",
    )

    subj_deg = pipe_deg[pipe_deg["subject"] == subj]
    if subj_deg.empty:
        st.info(f"No degradation data for S{subj} / {pipe}.")
        return

    # ── Takeaway ──────────────────────────────────────────────────────────
    total = subj_deg["total_pairs"].sum()
    degraded = subj_deg["degraded_pairs"].sum()
    ratio = degraded / total if total > 0 else 0
    st.info(
        f"**S{subj} / {pipe}**: {degraded}/{total} pairs degraded ({ratio:.0%})."
    )

    # ── 1. Per-config degradation status table ───────────────────────────
    with st.container(border=True):
        st.subheader(f"Config Degradation Status for S{subj}")
        st.caption("Each config's degradation status and accuracy comparison.")

        subj_sdf = sdf[
            (sdf["subject"] == subj) & (sdf["pipe_short"] == pipe)
        ].copy()

        if not subj_sdf.empty and "degrade_status" in subj_sdf.columns:
            display = subj_sdf[[
                "config_label", "degrade_status", "cvMeanAcc", "baseline", "score",
            ]].copy()
            display["DA Gain"] = display["cvMeanAcc"] - display["baseline"]
            display = display.rename(columns={
                "config_label": "Config", "degrade_status": "Status",
                "cvMeanAcc": "Accuracy", "baseline": "Baseline",
                "score": "Score Mode",
            })

            def _color_status(val):
                colors = {
                    "pure": "background-color: rgba(16,185,129,0.15);",
                    "partial": "background-color: rgba(245,158,11,0.15);",
                    "full_degrade": "background-color: rgba(239,68,68,0.15);",
                }
                return colors.get(val, "")

            st.dataframe(
                display.style
                    .format({
                        "Accuracy": "{:.4f}", "Baseline": "{:.4f}",
                        "DA Gain": "{:+.4f}",
                    })
                    .map(_color_status, subset=["Status"]),
                hide_index=True, use_container_width=True,
            )

    # ── 2. Per-fold degradation breakdown ────────────────────────────────
    with st.container(border=True):
        st.subheader(f"Fold-Level Degradation for S{subj}")
        st.caption(
            "Each fold (target session): was BDP pure or degraded? "
            "Shows accuracy for each status."
        )

        if not ddf.empty and "degraded" in ddf.columns:
            subj_detail = ddf[
                (ddf["subject"] == subj) & (ddf["pipe_short"] == pipe)
            ].copy()

            if not subj_detail.empty:
                # Filter to first config (method_row 0) for clean fold view
                if "method_row" in subj_detail.columns:
                    subj_detail = subj_detail[subj_detail["method_row"] == 0]

                fold_col = "pair_id" if "pair_id" in subj_detail.columns else None
                target_col = "test_label" if "test_label" in subj_detail.columns else None

                if fold_col:
                    fold_summary = subj_detail.groupby(fold_col).agg(
                        n_total=("degraded", "count"),
                        n_degraded=("degraded", "sum"),
                        mean_acc=("acc_DA", "mean"),
                    ).reset_index()
                    fold_summary["status"] = fold_summary["n_degraded"].apply(
                        lambda d: "Degraded" if d > 0 else "Pure BDP",
                    )

                    if target_col and target_col in subj_detail.columns:
                        target_map = subj_detail.groupby(fold_col)[target_col].first()
                        fold_summary["target"] = fold_summary[fold_col].map(target_map)
                    else:
                        fold_summary["target"] = fold_summary[fold_col].apply(
                            lambda f: f"Fold {f}",
                        )

                    # Bar chart: accuracy per fold, colored by status
                    fig = px.bar(
                        fold_summary, x="target", y="mean_acc",
                        color="status",
                        color_discrete_map={
                            "Pure BDP": DEGRADE_COLORS["pure"],
                            "Degraded": DEGRADE_COLORS["full_degrade"],
                        },
                        text=fold_summary["mean_acc"].apply(lambda v: f"{v:.3f}"),
                    )
                    fig.update_traces(textposition="outside", textfont_size=10)
                    fig.update_layout(
                        xaxis_title="Target Session", yaxis_title="Mean Accuracy",
                        legend_title="Status",
                    )
                    style_figure(fig)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No detail data for this subject/pipeline combination.")

    # ── 3. Per-feature breakdown for this subject ────────────────────────
    with st.container(border=True):
        st.subheader(f"Degradation by Feature for S{subj}")
        if not subj_deg.empty and "feature" in subj_deg.columns:
            feat_data = subj_deg[["feature", "total_pairs", "degraded_pairs",
                                   "degraded_ratio", "acc_pure", "acc_degraded"]].copy()
            feat_data = feat_data.rename(columns={
                "feature": "Feature", "total_pairs": "Total",
                "degraded_pairs": "Degraded", "degraded_ratio": "Degrade %",
                "acc_pure": "Acc (Pure)", "acc_degraded": "Acc (Degraded)",
            })
            st.dataframe(
                feat_data.style.format({
                    "Degrade %": "{:.1%}",
                    "Acc (Pure)": "{:.4f}", "Acc (Degraded)": "{:.4f}",
                }),
                hide_index=True, use_container_width=True,
            )

    # ── 4. Pure vs Degraded accuracy (this subject only) ─────────────────
    if not ddf.empty and "degraded" in ddf.columns and "acc_DA" in ddf.columns:
        subj_detail = ddf[
            (ddf["subject"] == subj) & (ddf["pipe_short"] == pipe)
        ].copy()
        if not subj_detail.empty and subj_detail["degraded"].nunique() > 1:
            with st.container(border=True):
                st.subheader(f"Pure vs Degraded Accuracy for S{subj}")
                subj_detail["status"] = subj_detail["degraded"].map(
                    {True: "Degraded", False: "Pure BDP"},
                )
                fig = px.violin(
                    subj_detail, x="status", y="acc_DA", color="status",
                    color_discrete_map={
                        "Pure BDP": DEGRADE_COLORS["pure"],
                        "Degraded": DEGRADE_COLORS["full_degrade"],
                    },
                    box=True, points="all",
                )
                fig.update_layout(
                    yaxis_title="Accuracy (acc_DA)", xaxis_title="",
                    showlegend=False,
                )
                style_figure(fig)
                st.plotly_chart(fig, use_container_width=True)
