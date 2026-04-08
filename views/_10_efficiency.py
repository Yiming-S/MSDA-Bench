"""Page 10: Efficiency & Progress."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, style_figure


def render(store, dataset):
    st.header("Runtime & Efficiency")
    st.markdown(
        "How long does each pipeline take, and is the extra computation worth it? "
        "Find the best accuracy-to-time tradeoff."
    )
    try:
        ddf = store.detail_df
        sdf = store.summary_df
        if ddf.empty and sdf.empty:
            st.warning("No data available for efficiency analysis.")
            return

        pipes = [p for p in PIPE_ORDER if p in sdf["pipe_short"].unique()] if not sdf.empty else []

        # Check for timing column
        time_col = None
        time_source = None
        for col, src in [
            ("elapsed_sec", "detail"), ("elapsed", "detail"),
            ("elapsed_sec", "summary"), ("time_sec", "summary"),
        ]:
            if src == "detail" and col in ddf.columns:
                time_col, time_source = col, "detail"
                break
            elif src == "summary" and col in sdf.columns:
                time_col, time_source = col, "summary"
                break

        if time_col is None:
            st.warning("No timing column (elapsed_sec) found in data.")
            st.info("Available detail columns: " + ", ".join(ddf.columns.tolist()[:20]))
            return

        src_df = ddf if time_source == "detail" else sdf

        # Pre-compute timing aggregates
        time_agg = src_df[src_df["pipe_short"].isin(pipes)].groupby("pipe_short")[
            time_col
        ].agg(["mean", "std", "median"]).reset_index()
        time_agg["order"] = time_agg["pipe_short"].map(
            {p: i for i, p in enumerate(PIPE_ORDER)},
        )
        time_agg = time_agg.sort_values("order")

        # ── Takeaway ──────────────────────────────────────────────────────────
        if not time_agg.empty:
            fastest = time_agg.loc[time_agg["mean"].idxmin()]
            slowest = time_agg.loc[time_agg["mean"].idxmax()]
            speedup = slowest["mean"] / fastest["mean"] if fastest["mean"] > 0 else 0
            st.success(
                f"Fastest: **{fastest['pipe_short']}** ({fastest['mean']:.1f}s). "
                f"Slowest: **{slowest['pipe_short']}** ({slowest['mean']:.1f}s). "
                f"Speedup: **{speedup:.1f}x**."
            )

        # ── Timing bar + Accuracy-time scatter (side by side) ─────────────────
        col_time, col_scatter = st.columns(2)

        with col_time:
            with st.container(border=True):
                st.subheader("Pipeline Timing")
                st.caption("Mean elapsed time per fold (log scale).")
                fig = go.Figure()
                for _, row in time_agg.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row["pipe_short"]], y=[row["mean"]],
                        marker_color=PIPE_COLORS.get(row["pipe_short"], "#888"),
                        name=row["pipe_short"], showlegend=False,
                        text=[f"{row['mean']:.1f}s"], textposition="outside",
                    ))
                fig.update_layout(
                    title="Mean Elapsed Time per Pipeline",
                    yaxis_title="Seconds (log scale)", yaxis_type="log",
                )
                style_figure(fig)
                st.plotly_chart(fig, use_container_width=True)

        with col_scatter:
            with st.container(border=True):
                st.subheader("Accuracy vs Time")
                st.caption("Top-left = best (high accuracy, low time).")
                acc_col = (
                    "acc_DA" if "acc_DA" in src_df.columns
                    else ("cvMeanAcc" if "cvMeanAcc" in src_df.columns else None)
                )
                if acc_col and time_col:
                    scatter_data = src_df[src_df["pipe_short"].isin(pipes)].copy()
                    scatter_data = scatter_data.dropna(subset=[acc_col, time_col])
                    if not scatter_data.empty:
                        fig = px.scatter(
                            scatter_data, x=time_col, y=acc_col,
                            color="pipe_short", color_discrete_map=PIPE_COLORS,
                            hover_data=(
                                ["config_label"] if "config_label" in scatter_data.columns
                                else None
                            ),
                            category_orders={"pipe_short": pipes}, opacity=0.6,
                        )
                        fig.update_layout(
                            xaxis_title="Time (sec)", yaxis_title="Accuracy",
                        )
                        style_figure(fig)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Accuracy column not found for scatter plot.")

        # ── Zoom controls for scatter ─────────────────────────────────────────
        acc_col = (
            "acc_DA" if "acc_DA" in src_df.columns
            else ("cvMeanAcc" if "cvMeanAcc" in src_df.columns else None)
        )
        if acc_col and time_col:
            with st.expander("Accuracy vs Time -- Zoom Controls"):
                scatter_data = src_df[src_df["pipe_short"].isin(pipes)].copy()
                scatter_data = scatter_data.dropna(subset=[acc_col, time_col])
                if not scatter_data.empty:
                    acc_min_data = float(scatter_data[acc_col].min())
                    acc_max_data = float(scatter_data[acc_col].max())
                    time_min_data = float(scatter_data[time_col].min())
                    time_max_data = float(scatter_data[time_col].max())

                    col_acc, col_time_r = st.columns(2)
                    with col_acc:
                        acc_range = st.slider(
                            "Accuracy range",
                            min_value=round(acc_min_data - 0.02, 2),
                            max_value=round(acc_max_data + 0.02, 2),
                            value=(
                                round(acc_min_data - 0.01, 2),
                                round(acc_max_data + 0.01, 2),
                            ),
                            step=0.01, key="acc_range",
                        )
                    with col_time_r:
                        time_range = st.slider(
                            "Time range (sec)",
                            min_value=0.0,
                            max_value=round(time_max_data * 1.2, 1),
                            value=(0.0, round(time_max_data * 1.1, 1)),
                            step=0.1, key="time_range",
                        )

                    mask = (
                        (scatter_data[acc_col] >= acc_range[0])
                        & (scatter_data[acc_col] <= acc_range[1])
                        & (scatter_data[time_col] >= time_range[0])
                        & (scatter_data[time_col] <= time_range[1])
                    )
                    filtered = scatter_data[mask]

                    if not filtered.empty:
                        fig = px.scatter(
                            filtered, x=time_col, y=acc_col,
                            color="pipe_short", color_discrete_map=PIPE_COLORS,
                            hover_data=(
                                ["config_label"]
                                if "config_label" in filtered.columns else None
                            ),
                            category_orders={"pipe_short": pipes}, opacity=0.6,
                        )
                        fig.update_layout(
                            xaxis_title="Time (sec)", yaxis_title="Accuracy",
                            xaxis=dict(range=[time_range[0], time_range[1]]),
                            yaxis=dict(range=[acc_range[0], acc_range[1]]),
                        )
                        style_figure(fig)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            f"Showing {len(filtered)}/{len(scatter_data)} points "
                            f"within selected range."
                        )
                    else:
                        st.info("No data points in the selected range.")

        # ── Per-subject timing table ──────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Per-Subject Mean Timing")
            st.caption("Mean computation time per subject by pipeline.")
            if "subject" in src_df.columns:
                subj_time = src_df[src_df["pipe_short"].isin(pipes)].groupby(
                    ["subject", "pipe_short"],
                )[time_col].mean().reset_index()
                pivot = subj_time.pivot_table(
                    index="subject", columns="pipe_short",
                    values=time_col, aggfunc="mean",
                )
                pivot = pivot.reindex(columns=[p for p in PIPE_ORDER if p in pivot.columns])
                pivot.index = [f"S{s}" for s in pivot.index]
                st.dataframe(
                    pivot.style.format("{:.1f}", na_rep="---"),
                    use_container_width=True,
                )

        # ── Top 10 slowest configs ────────────────────────────────────────────
        with st.container(border=True):
            st.subheader("Top 10 Slowest Configurations")
            st.caption("Most computationally expensive config-pipeline combinations.")
            if "config_label" in src_df.columns:
                slow = src_df[src_df["pipe_short"].isin(pipes)].groupby(
                    ["pipe_short", "config_label"],
                )[time_col].mean().reset_index()
                slow = slow.sort_values(time_col, ascending=False).head(10)
                slow.columns = ["Pipeline", "Configuration", "Mean Time (s)"]
                st.dataframe(
                    slow.style.format({"Mean Time (s)": "{:.1f}"}),
                    hide_index=True, use_container_width=True,
                )

    except Exception as e:
        st.error(f"Error rendering Efficiency page: {e}")
