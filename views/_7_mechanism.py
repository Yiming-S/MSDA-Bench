"""Page 7: Mechanism Explorer."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, ROLE_COLORS, DEGRADE_COLORS, style_figure


# ── Lazy utilization computation (cached) ─────────────────────────────────────

@st.cache_data
def compute_utilization(_store, dataset):
    """Compute session utilization per subject x pipeline x fold."""
    sdf = _store.summary_df
    if sdf.empty:
        return pd.DataFrame()

    combos = sdf[["subject", "pipe_short", "n_session"]].drop_duplicates()
    rows = []

    for _, combo in combos.iterrows():
        subj = int(combo["subject"])
        ps = combo["pipe_short"]
        n_sess = int(combo["n_session"])
        n_avail = n_sess - 1

        roles = _store.get_roles(subj, ps)
        if roles.empty:
            continue

        mask = pd.Series(True, index=roles.index)
        if "method_row" in roles.columns:
            mask = mask & (roles["method_row"] == 0)
        if "stage" in roles.columns:
            mask = mask & (roles["stage"] == "final")
        filtered = roles[mask]

        if filtered.empty:
            continue

        fold_col = next(
            (c for c in ["pair_id", "fold", "fold_id"] if c in filtered.columns), None,
        )
        role_col = next(
            (c for c in ["role", "session_role"] if c in filtered.columns), None,
        )
        if fold_col is None or role_col is None:
            continue

        for fold_id, fold_data in filtered.groupby(fold_col):
            n_used = int((fold_data[role_col] == "train").sum())
            rows.append({
                "subject": subj, "pipe_short": ps,
                "pair_id": int(fold_id), "n_used": n_used,
                "n_available": n_avail,
                "utilization": n_used / n_avail if n_avail > 0 else 0.0,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Main render ───────────────────────────────────────────────────────────────

def render(store, dataset):
    st.header("Session Mechanisms")
    st.markdown(
        "Visualize HOW each pipeline uses sessions. See BDP bridge/far assignments, "
        "MMP source selection, and DWP distance weights."
    )
    try:
        sdf = store.summary_df
        if sdf.empty:
            st.warning("No data available.")
            return

        subjects = sorted(sdf["subject"].unique())
        mech_pipes = ["BDP_fb", "BDP_bf", "MMP_mta", "MMP_moe"]
        avail_pipes = [p for p in mech_pipes if p in sdf["pipe_short"].unique()]

        if not avail_pipes:
            st.warning("No mechanism-based pipelines (BDP/MMP) found in data.")
            return

        pipe = st.selectbox("Pipeline", avail_pipes, key="mech_pipe")

        # ── Takeaway ──────────────────────────────────────────────────────────
        util_df = compute_utilization(store, dataset)
        if not util_df.empty:
            pipe_util = util_df[util_df["pipe_short"] == pipe]
            if not pipe_util.empty:
                mean_util = pipe_util["utilization"].mean()
                st.info(
                    f"**{pipe}** uses **{mean_util:.0%}** of available sessions on average."
                )

        # ── Tabs ──────────────────────────────────────────────────────────────
        deg_df = store.derived.get("degradation", pd.DataFrame())
        has_degradation = not deg_df.empty and pipe.startswith("BDP")

        if has_degradation:
            tab_util, tab_roles, tab_degrade = st.tabs(
                ["Session Utilization", "Session Roles", "Degradation Analysis"],
            )
        else:
            tab_util, tab_roles = st.tabs(["Session Utilization", "Session Roles"])

        # ================================================================
        # TAB 1: Session Utilization
        # ================================================================
        with tab_util:
            if util_df.empty:
                st.warning("No roles data available for utilization analysis.")
            else:
                subj_util = util_df.groupby(["subject", "pipe_short"]).agg(
                    mean_used=("n_used", "mean"),
                    sd_used=("n_used", lambda x: x.std(ddof=1) if len(x) > 1 else 0.0),
                    min_used=("n_used", "min"),
                    max_used=("n_used", "max"),
                    mean_ratio=("utilization", "mean"),
                    n_folds=("pair_id", "count"),
                ).reset_index()
                avail_map = util_df[["subject", "n_available"]].drop_duplicates()
                subj_util = subj_util.merge(avail_map, on="subject", how="left")

                all_subjs = sorted(subj_util["subject"].unique())
                sel_subjs = st.multiselect(
                    "Subjects to include", all_subjs, default=all_subjs,
                    format_func=lambda s: f"S{s}", key="util_subj_filter",
                )
                if not sel_subjs:
                    st.info("Select at least one subject.")
                else:
                    filt = subj_util[subj_util["subject"].isin(sel_subjs)]

                    # Box plot + Heatmap side by side
                    col_box, col_hm = st.columns(2)

                    with col_box:
                        with st.container(border=True):
                            st.subheader("Utilization Distribution")
                            ordered_pipes = [
                                p for p in PIPE_ORDER if p in filt["pipe_short"].unique()
                            ]
                            fig = px.box(
                                filt, x="pipe_short", y="mean_ratio",
                                color="pipe_short", color_discrete_map=PIPE_COLORS,
                                category_orders={"pipe_short": ordered_pipes},
                                points="all",
                                hover_data={
                                    "subject": True, "mean_used": ":.1f",
                                    "n_available": True, "mean_ratio": ":.1%",
                                },
                            )
                            fig.update_layout(
                                yaxis_title="Mean Utilization Ratio",
                                yaxis=dict(range=[-0.05, 1.1], tickformat=".0%"),
                                xaxis_title="Pipeline", showlegend=False,
                            )
                            style_figure(fig)
                            st.plotly_chart(fig, use_container_width=True)

                    with col_hm:
                        with st.container(border=True):
                            st.subheader("Utilization Heatmap")
                            pivot = filt.pivot_table(
                                index="subject", columns="pipe_short",
                                values="mean_ratio", aggfunc="first",
                            )
                            pivot = pivot[[p for p in PIPE_ORDER if p in pivot.columns]]
                            pivot.index = [f"S{s}" for s in pivot.index]
                            fig = px.imshow(
                                pivot.values,
                                x=pivot.columns.tolist(), y=pivot.index.tolist(),
                                color_continuous_scale="YlOrRd_r",
                                zmin=0, zmax=1, text_auto=".0%", aspect="auto",
                            )
                            fig.update_layout(coloraxis_colorbar_title="Utilization")
                            style_figure(fig, height=max(300, len(pivot) * 30))
                            st.plotly_chart(fig, use_container_width=True)

                    # Per-subject table
                    with st.container(border=True):
                        st.subheader("Per-Subject Utilization Summary")
                        display = filt[[
                            "subject", "pipe_short", "n_available",
                            "mean_used", "mean_ratio", "sd_used",
                            "min_used", "max_used", "n_folds",
                        ]].copy()
                        display["subject"] = display["subject"].apply(lambda s: f"S{s}")
                        display = display.rename(columns={
                            "subject": "Subject", "pipe_short": "Pipeline",
                            "n_available": "Available", "mean_used": "Mean #Used",
                            "mean_ratio": "Mean Util.", "sd_used": "SD",
                            "min_used": "Min", "max_used": "Max", "n_folds": "Folds",
                        }).sort_values(["Subject", "Pipeline"])
                        st.dataframe(
                            display.style.format({
                                "Mean #Used": "{:.1f}", "Mean Util.": "{:.1%}",
                                "SD": "{:.2f}",
                            }),
                            hide_index=True, use_container_width=True,
                        )

        # ================================================================
        # TAB 2: Session Roles
        # ================================================================
        with tab_roles:
            _render_session_roles(store, dataset, subjects, pipe, avail_pipes)

        # ================================================================
        # TAB 3: Degradation Analysis (BDP only)
        # ================================================================
        if has_degradation:
            with tab_degrade:
                _render_degradation_tab(store, deg_df, pipe, subjects)

    except Exception as e:
        st.error(f"Error rendering Mechanism Explorer page: {e}")


# ── Session Roles tab ─────────────────────────────────────────────────────────

def _render_session_roles(store, dataset, subjects, pipe, avail_pipes):
    """Session roles visualization."""
    sample_roles = store.get_roles(subjects[0], pipe)
    fold_col = next(
        (c for c in ["pair_id", "fold", "fold_id"] if c in sample_roles.columns), None,
    )
    role_col = next(
        (c for c in ["role", "session_role", "type"] if c in sample_roles.columns), None,
    )
    dist_col = next(
        (c for c in ["dist_est", "distance", "dist", "mmd"] if c in sample_roles.columns),
        None,
    )
    sess_col = next(
        (c for c in ["session_label", "session", "session_id"] if c in sample_roles.columns),
        None,
    )
    stage_col = "stage" if "stage" in sample_roles.columns else None

    # ── All-Subjects Overview ─────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Session Roles -- All Subjects")
        st.caption("Each subject's session distance-to-target colored by role.")

        selected_subjects = st.multiselect(
            "Subjects to display", subjects, default=subjects,
            format_func=lambda s: f"S{s}", key="mech_subj_multi",
        )

        if not selected_subjects:
            st.info("Select at least one subject.")
            return

        all_roles_data = []
        for s in selected_subjects:
            rdf = store.get_roles(s, pipe)
            if rdf.empty:
                continue
            if stage_col:
                if pipe.startswith("BDP"):
                    rdf = rdf[rdf[stage_col] == "selection"]
                elif pipe.startswith("MMP"):
                    rdf = rdf[rdf[stage_col] == "main"]
            if "method_row" in rdf.columns:
                rdf = rdf[rdf["method_row"] == 0]
            if fold_col and fold_col in rdf.columns:
                first_fold = rdf[fold_col].min()
                rdf = rdf[rdf[fold_col] == first_fold]
            if role_col and sess_col:
                rdf = rdf[rdf[role_col] != "target"].copy()
                rdf["subject_label"] = f"S{s}"
                all_roles_data.append(rdf)

        if all_roles_data and dist_col and role_col and sess_col:
            combined = pd.concat(all_roles_data, ignore_index=True)
            fig = px.scatter(
                combined, x=dist_col, y="subject_label",
                color=role_col, color_discrete_map=ROLE_COLORS,
                hover_data=[sess_col], text=sess_col,
                category_orders={
                    "subject_label": [f"S{s}" for s in selected_subjects],
                },
            )
            fig.update_traces(textposition="top center", marker=dict(size=10))
            fig.update_layout(
                xaxis_title="Distance to Target (MMD)", yaxis_title="Subject",
            )
            style_figure(fig, height=max(300, len(selected_subjects) * 50))
            st.plotly_chart(fig, use_container_width=True)
        elif all_roles_data and role_col:
            combined = pd.concat(all_roles_data, ignore_index=True)
            counts = combined.groupby(["subject_label", role_col]).size().reset_index(name="count")
            fig = px.bar(
                counts, x="subject_label", y="count", color=role_col,
                color_discrete_map=ROLE_COLORS, barmode="stack",
            )
            fig.update_layout(xaxis_title="Subject", yaxis_title="Count")
            style_figure(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No role/distance data available for the selected pipeline.")

    # ── Single-Subject Detail ─────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Single-Subject Detail")
        st.caption("Fold-by-fold session assignments.")

        subj = st.selectbox(
            "Subject for detail view",
            selected_subjects if selected_subjects else subjects,
            format_func=lambda s: f"S{s}", key="mech_subj_detail",
        )
        roles_df = store.get_roles(subj, pipe)
        if roles_df.empty:
            st.warning(f"No role data found for S{subj} / {pipe}.")
            return

        if stage_col and stage_col in roles_df.columns:
            if pipe.startswith("BDP"):
                roles_df = roles_df[roles_df[stage_col] == "selection"]
            elif pipe.startswith("MMP"):
                roles_df = roles_df[roles_df[stage_col] == "main"]
        if "method_row" in roles_df.columns:
            roles_df = roles_df[roles_df["method_row"] == 0]

        folds = sorted(roles_df[fold_col].unique()) if fold_col and fold_col in roles_df.columns else [0]
        fold = st.slider(
            "Fold", min_value=int(min(folds)), max_value=int(max(folds)),
            value=int(min(folds)), key="mech_fold_detail",
        )
        fold_data = roles_df[roles_df[fold_col] == fold] if fold_col else roles_df

        # Session role scatter
        if role_col and dist_col and sess_col:
            plot_data = fold_data[fold_data[role_col] != "target"].copy()
            if not plot_data.empty:
                fig = px.scatter(
                    plot_data, x=dist_col,
                    y=np.random.default_rng(42).uniform(0.8, 1.2, len(plot_data)),
                    color=role_col, color_discrete_map=ROLE_COLORS,
                    hover_data=[sess_col], text=sess_col,
                )
                fig.update_traces(textposition="top center", marker=dict(size=12))
                fig.update_layout(
                    xaxis_title="Distance to Target (MMD)",
                    yaxis=dict(visible=False),
                    title=f"S{subj} / {pipe} / Fold {fold}",
                )
                style_figure(fig, height=250)
                st.plotly_chart(fig, use_container_width=True)
        elif role_col and sess_col:
            role_counts = fold_data[role_col].value_counts().reset_index()
            role_counts.columns = ["role", "count"]
            fig = px.bar(
                role_counts, x="role", y="count", color="role",
                color_discrete_map=ROLE_COLORS,
            )
            fig.update_layout(title=f"Role Counts (Fold {fold})")
            style_figure(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Session assignment table
        display_cols = [
            c for c in [
                sess_col, role_col, dist_col, "dist_lwr", "dist_upr",
                "weight", "partition_mode", "is_best",
            ]
            if c and c in fold_data.columns
        ]
        if display_cols:
            show_data = (
                fold_data[fold_data[role_col] != "target"][display_cols]
                if role_col else fold_data[display_cols]
            )
            st.dataframe(show_data, hide_index=True, use_container_width=True)
        else:
            st.dataframe(fold_data, hide_index=True, use_container_width=True)

    # ── All-Folds Summary ─────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("All-Folds Summary")
        if fold_col and role_col and sess_col:
            summary_rows = []
            for f in folds:
                f_data = roles_df[roles_df[fold_col] == f]
                f_data = f_data[f_data[role_col] != "target"]
                row_dict = {"Fold": f}
                for role in sorted(f_data[role_col].unique()):
                    sessions = f_data[f_data[role_col] == role][sess_col].tolist()
                    row_dict[role] = ", ".join(str(s) for s in sessions)
                if "partition_mode" in f_data.columns:
                    modes = f_data["partition_mode"].unique()
                    row_dict["Mode"] = modes[0] if len(modes) == 1 else ", ".join(str(m) for m in modes)
                summary_rows.append(row_dict)
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, hide_index=True, use_container_width=True)
                st.caption("Consistent patterns suggest stable mechanism behavior.")

    # ── Bridge count distribution (BDP only) ──────────────────────────────────
    if pipe.startswith("BDP"):
        with st.container(border=True):
            st.subheader("Bridge Session Count Distribution")
            if role_col and fold_col:
                bridge_counts = (
                    roles_df[roles_df[role_col] == "bridge"]
                    .groupby(fold_col).size().reset_index(name="n_bridge")
                )
                if not bridge_counts.empty:
                    fig = px.histogram(
                        bridge_counts, x="n_bridge",
                        nbins=int(max(bridge_counts["n_bridge"].max(), 5)),
                    )
                    fig.update_layout(
                        title="Bridge Sessions per Fold",
                        xaxis_title="N Bridge", yaxis_title="Count",
                    )
                    style_figure(fig)
                    st.plotly_chart(fig, use_container_width=True)


# ── Degradation Analysis tab ─────────────────────────────────────────────────

def _render_degradation_tab(store, deg_df, pipe, subjects):
    """Render the BDP degradation analysis tab."""

    pipe_deg = deg_df[deg_df["pipe_short"] == pipe].copy()
    if pipe_deg.empty:
        st.info(f"No degradation data for {pipe}.")
        return

    # ── Takeaway ──────────────────────────────────────────────────────────
    overall_ratio = (
        pipe_deg["degraded_pairs"].sum() / pipe_deg["total_pairs"].sum()
        if pipe_deg["total_pairs"].sum() > 0 else 0
    )
    n_full = (pipe_deg["degraded_ratio"] == 1.0).sum()
    n_pure = (pipe_deg["degraded_ratio"] == 0.0).sum()
    st.info(
        f"**{pipe}**: {overall_ratio:.0%} of all pairs degraded to MAP. "
        f"{n_pure} subject-feature combos fully pure, {n_full} fully degraded."
    )

    # ── 1. Heatmap: subject x feature degradation ratio ──────────────────
    col_hm, col_bar = st.columns(2)

    with col_hm:
        with st.container(border=True):
            st.subheader("Degradation Rate Heatmap")
            st.caption("Subject x Feature. Darker red = more pairs degraded to MAP.")
            if "feature" in pipe_deg.columns:
                pivot = pipe_deg.pivot_table(
                    index="subject", columns="feature",
                    values="degraded_ratio", aggfunc="first",
                )
                pivot.index = [f"S{s}" for s in pivot.index]
                fig = px.imshow(
                    pivot.values,
                    x=pivot.columns.tolist(), y=pivot.index.tolist(),
                    color_continuous_scale=[[0, "#F0FDF4"], [0.5, "#FDE68A"], [1, "#EF4444"]],
                    zmin=0, zmax=1, text_auto=".0%", aspect="auto",
                )
                fig.update_layout(coloraxis_colorbar_title="Degrade %")
                style_figure(fig, height=max(300, len(pivot) * 28))
                st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        with st.container(border=True):
            st.subheader("Per-Subject Degradation")
            st.caption("Stacked bars: pure vs degraded pairs per subject.")
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

    # ── 2. Pure vs Degraded accuracy comparison ──────────────────────────
    with st.container(border=True):
        st.subheader("Pure vs Degraded Accuracy")
        st.caption(
            "Box plot comparing acc_DA for pure BDP pairs vs degraded (MAP-style) pairs."
        )
        ddf = store.detail_df
        if not ddf.empty and "degraded" in ddf.columns and "acc_DA" in ddf.columns:
            bdp_detail = ddf[ddf["pipe_short"] == pipe].copy()
            bdp_detail["status"] = bdp_detail["degraded"].map(
                {True: "Degraded", False: "Pure BDP"},
            )
            if not bdp_detail.empty:
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

                # Summary stats table
                stats = bdp_detail.groupby("status")["acc_DA"].agg(
                    ["mean", "median", "std", "count"],
                ).reset_index()
                stats.columns = ["Status", "Mean", "Median", "SD", "N Pairs"]
                st.dataframe(
                    stats.style.format({"Mean": "{:.4f}", "Median": "{:.4f}", "SD": "{:.4f}"}),
                    hide_index=True, use_container_width=True,
                )
        else:
            st.info("Detail data with degradation info not available.")

    # ── 3. Degradation rate by feature ───────────────────────────────────
    if "feature" in pipe_deg.columns:
        with st.container(border=True):
            st.subheader("Degradation by Feature")
            st.caption("Which features trigger degradation most often?")
            feat_agg = pipe_deg.groupby("feature").agg(
                total=("total_pairs", "sum"),
                degraded=("degraded_pairs", "sum"),
            ).reset_index()
            feat_agg["degraded_ratio"] = feat_agg["degraded"] / feat_agg["total"]
            feat_agg = feat_agg.sort_values("degraded_ratio", ascending=False)

            fig = px.bar(
                feat_agg, x="feature", y="degraded_ratio",
                color="degraded_ratio",
                color_continuous_scale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#EF4444"]],
                text=feat_agg["degraded_ratio"].apply(lambda v: f"{v:.0%}"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis_title="Degradation Rate", yaxis=dict(tickformat=".0%"),
                xaxis_title="Feature", coloraxis_showscale=False,
            )
            style_figure(fig)
            st.plotly_chart(fig, use_container_width=True)
