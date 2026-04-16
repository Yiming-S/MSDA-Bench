"""Page 5: Subject Explorer."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, format_acc, style_figure


def render(store, dataset):
    st.header("Subject Explorer")
    st.markdown(
        "Deep-dive into a single subject. See all 24 configurations across all "
        "pipelines and understand individual differences."
    )
    try:
        sdf = store.summary_df
        sp = store.derived["subject_pipeline"]
        if sdf.empty or sp.empty:
            st.warning("No data available.")
            return

        subjects = sorted(sdf["subject"].unique())
        subj = st.selectbox("Subject", subjects, format_func=lambda s: f"S{s}")

        pipes = [p for p in PIPE_ORDER if p in sdf["pipe_short"].unique()]
        subj_data = sdf[sdf["subject"] == subj]
        sp_subj = sp[sp["subject"] == subj]

        # ── Takeaway ──────────────────────────────────────────────────────────
        if not sp_subj.empty:
            best = sp_subj.loc[sp_subj["M_acc"].idxmax()]
            st.success(
                f"Best pipeline for S{subj}: **{best['pipe_short']}** "
                f"(M_acc = {best['M_acc']:.4f}, B_acc = {best['B_acc']:.4f})"
            )

        # ── Pipeline summary (side-by-side: chart + table) ───────────────────
        with st.container(border=True):
            st.subheader(f"Pipeline Summary for S{subj}")

            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                if not sp_subj.empty:
                    sp_ord = sp_subj.set_index("pipe_short").reindex(
                        [p for p in PIPE_ORDER if p in sp_subj["pipe_short"].values],
                    ).reset_index()
                    metrics = ["M_acc", "B_acc"]
                    fig = go.Figure()
                    for m in metrics:
                        if m in sp_ord.columns:
                            fig.add_trace(go.Bar(
                                x=sp_ord["pipe_short"], y=sp_ord[m], name=m,
                                text=sp_ord[m].apply(lambda v: f"{v:.4f}"),
                                textposition="outside",
                            ))
                    fig.update_layout(
                        title=f"S{subj}: Mean vs Best Accuracy",
                        yaxis_title="Accuracy", barmode="group",
                    )
                    style_figure(fig)
                    st.plotly_chart(fig, use_container_width=True)

            with col_table:
                if not sp_subj.empty:
                    disp_cols = [
                        c for c in [
                            "pipe_short", "M_acc", "B_acc", "G_gain", "H_helps",
                            "best_feature", "best_classifier", "best_da",
                        ]
                        if c in sp_ord.columns
                    ]
                    rename_map = {
                        "pipe_short": "Pipeline", "M_acc": "Mean Acc",
                        "B_acc": "Best Acc", "G_gain": "DA Gain",
                        "H_helps": "Helps Rate", "best_feature": "Best Feat",
                        "best_classifier": "Best Clf", "best_da": "Best DA",
                    }
                    disp_df = sp_ord[disp_cols].rename(columns=rename_map)
                    fmt2 = {
                        rename_map[c]: "{:.4f}"
                        for c in disp_cols
                        if c in ("M_acc", "B_acc", "G_gain", "H_helps")
                    }
                    st.dataframe(
                        disp_df.style.format(fmt2),
                        hide_index=True, use_container_width=True,
                    )

        # ── Full 24x6 accuracy table ─────────────────────────────────────────
        with st.container(border=True):
            st.subheader(f"Accuracy Table for S{subj}")
            st.caption("Each row = configuration; each column = pipeline.")
            if not subj_data.empty:
                pivot = subj_data.pivot_table(
                    index="config_label", columns="pipe_short",
                    values="cvMeanAcc", aggfunc="first",
                )
                pivot = pivot.reindex(
                    columns=[p for p in PIPE_ORDER if p in pivot.columns],
                )
                fmt = {c: "{:.4f}" for c in pivot.columns}

                # Build degradation status pivot for BDP columns
                has_degrade = "degrade_status" in subj_data.columns
                bdp_cols_in_pivot = [c for c in pivot.columns if c.startswith("BDP")]
                if has_degrade and bdp_cols_in_pivot:
                    deg_pivot = subj_data.pivot_table(
                        index="config_label", columns="pipe_short",
                        values="degrade_status", aggfunc="first",
                    )

                    styled = pivot.style.format(fmt, na_rep="---").background_gradient(
                        cmap="RdYlGn", axis=None,
                    )
                    # Apply degradation markers via cell-level styling
                    deg_style_map = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
                    for col in bdp_cols_in_pivot:
                        if col in deg_pivot.columns:
                            for idx in pivot.index:
                                if idx in deg_pivot.index:
                                    status = deg_pivot.at[idx, col]
                                    if status == "full_degrade":
                                        deg_style_map.at[idx, col] = "border-left: 4px solid #EF4444;"
                                    elif status == "partial":
                                        deg_style_map.at[idx, col] = "border-left: 4px solid #F59E0B;"
                    styled = styled.apply(lambda _: deg_style_map, axis=None)
                    st.dataframe(
                        styled,
                        use_container_width=True,
                        height=min(500, max(200, len(pivot) * 35)),
                    )
                    st.caption(
                        "Red left border = fully degraded to MAP; "
                        "amber left border = partially degraded."
                    )

                    # Drill-down links to BDP Degradation page
                    deg_summary = store.derived.get("degradation", pd.DataFrame())
                    if not deg_summary.empty:
                        for bdp_col in bdp_cols_in_pivot:
                            col_deg = deg_summary[
                                (deg_summary["subject"] == subj)
                                & (deg_summary["pipe_short"] == bdp_col)
                            ]
                            if not col_deg.empty:
                                ratio = (
                                    col_deg["degraded_pairs"].sum()
                                    / col_deg["total_pairs"].sum()
                                )
                                if ratio > 0:
                                    st.page_link(
                                        "views/_11_degradation.py",
                                        label=(
                                            f"S{subj} / {bdp_col}: "
                                            f"{ratio:.0%} degraded — view details"
                                        ),
                                        icon=":material/warning:",
                                    )
                else:
                    st.dataframe(
                        pivot.style.format(fmt, na_rep="---").background_gradient(
                            cmap="RdYlGn", axis=None,
                        ),
                        use_container_width=True,
                        height=min(500, max(200, len(pivot) * 35)),
                    )
            else:
                st.info(f"No data for subject S{subj}.")

        # ── Best vs mean gap per pipeline ─────────────────────────────────────
        with st.container(border=True):
            st.subheader(f"Oracle Premium (B - M) for S{subj}")
            st.caption("Larger gap = configuration selection matters more.")
            if not sp_subj.empty:
                sp_gap = sp_subj.copy()
                sp_gap["gap"] = sp_gap["B_acc"] - sp_gap["M_acc"]
                sp_gap = sp_gap[sp_gap["pipe_short"].isin(pipes)]
                sp_gap["order"] = sp_gap["pipe_short"].map(
                    {p: i for i, p in enumerate(PIPE_ORDER)},
                )
                sp_gap = sp_gap.sort_values("order")
                fig = px.bar(
                    sp_gap, x="pipe_short", y="gap", color="pipe_short",
                    color_discrete_map=PIPE_COLORS,
                    category_orders={"pipe_short": pipes},
                )
                fig.update_layout(
                    title=f"S{subj}: Oracle Premium (B - M)",
                    yaxis_title="Accuracy Gap", showlegend=False,
                )
                style_figure(fig)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering Subject Explorer page: {e}")
