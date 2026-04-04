"""Page 7: Mechanism Explorer."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, ROLE_COLORS


def render(store, dataset):
    st.header("7. Mechanism Explorer")
    st.markdown("Visualize HOW each pipeline uses sessions. "
                "See BDP bridge/far assignments, MMP source selection, and DWP distance weights fold by fold.")
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

        # Pipeline selector in sidebar (global for page)
        pipe = st.sidebar.selectbox("Pipeline (Mechanism)", avail_pipes, key="mech_pipe")
        fold_default = 0

        # --- Detect columns from first available roles file ---
        sample_roles = store.get_roles(subjects[0], pipe)
        fold_col = next((c for c in ["pair_id", "fold", "fold_id"] if c in sample_roles.columns), None)
        role_col = next((c for c in ["role", "session_role", "type"] if c in sample_roles.columns), None)
        dist_col = next((c for c in ["dist_est", "distance", "dist", "mmd"] if c in sample_roles.columns), None)
        sess_col = next((c for c in ["session_label", "session", "session_id"] if c in sample_roles.columns), None)
        stage_col = "stage" if "stage" in sample_roles.columns else None

        # ============================================================
        # SECTION 1: All-Subjects Session Roles Overview
        # ============================================================
        st.subheader("Session Roles — All Subjects Overview")
        st.caption("Each subplot shows one subject's session distance-to-target colored by role. "
                   "Select which subjects to display below.")

        # Subject multi-select in main area
        selected_subjects = st.multiselect(
            "Subjects to display",
            subjects,
            default=subjects,
            format_func=lambda s: f"S{s}",
            key="mech_subj_multi"
        )

        if not selected_subjects:
            st.info("Select at least one subject.")
            return

        # Collect role data for all selected subjects (first config, first fold)
        all_roles_data = []
        for s in selected_subjects:
            rdf = store.get_roles(s, pipe)
            if rdf.empty:
                continue

            # Filter: first config, determine stage
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
                hover_data=[sess_col],
                text=sess_col,
                category_orders={"subject_label": [f"S{s}" for s in selected_subjects]}
            )
            fig.update_traces(textposition="top center", marker=dict(size=10))
            fig.update_layout(
                xaxis_title="Distance to Target (MMD)",
                yaxis_title="Subject",
                template="plotly_white",
                height=max(300, len(selected_subjects) * 50),
            )
            st.plotly_chart(fig, use_container_width=True)
        elif all_roles_data and role_col:
            # Fallback: bar chart of role counts per subject
            combined = pd.concat(all_roles_data, ignore_index=True)
            counts = combined.groupby(["subject_label", role_col]).size().reset_index(name="count")
            fig = px.bar(counts, x="subject_label", y="count", color=role_col,
                         color_discrete_map=ROLE_COLORS, barmode="stack")
            fig.update_layout(template="plotly_white", xaxis_title="Subject", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No role/distance data available for the selected pipeline.")

        # ============================================================
        # SECTION 2: Single-Subject Detail (below the overview)
        # ============================================================
        st.markdown("---")
        st.subheader("Single-Subject Detail")
        st.caption("Select a subject below to see fold-by-fold session assignments and detailed role tables.")

        # Subject selector in main area
        subj = st.selectbox(
            "Subject for detail view",
            selected_subjects if selected_subjects else subjects,
            format_func=lambda s: f"S{s}",
            key="mech_subj_detail"
        )

        roles_df = store.get_roles(subj, pipe)
        if roles_df.empty:
            st.warning(f"No role data found for S{subj} / {pipe}.")
            return

        # Filter by stage
        if stage_col and stage_col in roles_df.columns:
            if pipe.startswith("BDP"):
                roles_df = roles_df[roles_df[stage_col] == "selection"]
            elif pipe.startswith("MMP"):
                roles_df = roles_df[roles_df[stage_col] == "main"]

        if "method_row" in roles_df.columns:
            roles_df = roles_df[roles_df["method_row"] == 0]

        # Fold selector
        folds = sorted(roles_df[fold_col].unique()) if fold_col and fold_col in roles_df.columns else [0]
        fold = st.slider("Fold", min_value=int(min(folds)),
                          max_value=int(max(folds)), value=int(min(folds)),
                          key="mech_fold_detail")

        fold_data = roles_df[roles_df[fold_col] == fold] if fold_col else roles_df

        # --- Session role scatter for this subject/fold ---
        st.subheader(f"Session Roles: S{subj} / {pipe} / Fold {fold}")

        if role_col and dist_col and sess_col:
            plot_data = fold_data[fold_data[role_col] != "target"].copy()
            if not plot_data.empty:
                fig = px.scatter(plot_data, x=dist_col,
                                 y=np.random.default_rng(42).uniform(0.8, 1.2, len(plot_data)),
                                 color=role_col, color_discrete_map=ROLE_COLORS,
                                 hover_data=[sess_col], text=sess_col)
                fig.update_traces(textposition="top center", marker=dict(size=12))
                fig.update_layout(
                    xaxis_title="Distance to Target (MMD)",
                    yaxis=dict(visible=False),
                    template="plotly_white", height=250,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Each point is one training session. Position = distance to target. Color = assigned role.")
        elif role_col and sess_col:
            role_counts = fold_data[role_col].value_counts().reset_index()
            role_counts.columns = ["role", "count"]
            fig = px.bar(role_counts, x="role", y="count", color="role",
                         color_discrete_map=ROLE_COLORS)
            fig.update_layout(title=f"Role Counts (Fold {fold})", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # --- Session assignment table ---
        st.subheader(f"Session Assignments (Fold {fold})")
        display_cols = [c for c in [sess_col, role_col, dist_col, "dist_lwr", "dist_upr",
                                     "weight", "partition_mode", "is_best"]
                       if c and c in fold_data.columns]
        if display_cols:
            show_data = fold_data[fold_data[role_col] != "target"][display_cols] if role_col else fold_data[display_cols]
            st.dataframe(show_data, hide_index=True, use_container_width=True)
        else:
            st.dataframe(fold_data, hide_index=True, use_container_width=True)
        st.caption("Full table of session assignments for the selected fold.")

        # --- All-folds comparison ---
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
                st.caption("Session assignments across all folds. Consistent patterns suggest stable mechanism behavior.")

        # --- Bridge count / weight distribution ---
        if pipe.startswith("BDP"):
            st.subheader("Bridge Session Count Distribution")
            if role_col and fold_col:
                bridge_counts = roles_df[roles_df[role_col] == "bridge"].groupby(fold_col).size().reset_index(name="n_bridge")
                if not bridge_counts.empty:
                    fig = px.histogram(bridge_counts, x="n_bridge", nbins=max(bridge_counts["n_bridge"].max(), 5))
                    fig.update_layout(title="Bridge Sessions per Fold", xaxis_title="N Bridge",
                                      yaxis_title="Count", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Distribution of how many sessions are classified as bridge across all folds.")

    except Exception as e:
        st.error(f"Error rendering Mechanism Explorer page: {e}")
