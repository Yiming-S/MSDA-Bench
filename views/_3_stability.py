"""Page 3: Stability & Sensitivity."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, COOL_LIGHT_INTENSITY, COOL_LIGHT_SEQUENTIAL_REVERSED


def render(store, dataset):
    st.header("3. Selection Sensitivity")
    st.markdown("How robust are the results? Are pipeline rankings fragile, and how much does picking the right config matter?")
    try:
        sdf = store.summary_df
        sp = store.derived["subject_pipeline"]
        cfg = store.derived["config_agg"]
        if sdf.empty or sp.empty:
            st.warning("No data available for stability analysis.")
            return

        pipes = [p for p in PIPE_ORDER if p in sp["pipe_short"].unique()]

        # --- Best vs 2nd-best gap histogram ---
        st.subheader("Best vs 2nd-Best Config Gap")
        st.caption("Gap between the best and second-best config for each subject x pipeline. Small gaps mean the winner could easily change with a different fold split.")
        gaps = []
        for (subj, pipe), grp in sdf.groupby(["subject", "pipe_short"]):
            accs = grp["cvMeanAcc"].dropna().sort_values(ascending=False)
            if len(accs) >= 2:
                gaps.append({"subject": subj, "pipe_short": pipe,
                             "gap": accs.iloc[0] - accs.iloc[1]})
        if gaps:
            gap_df = pd.DataFrame(gaps)
            fig = px.histogram(gap_df, x="gap", color="pipe_short", barmode="overlay",
                               color_discrete_map=PIPE_COLORS, nbins=30,
                               category_orders={"pipe_short": pipes})
            fig.update_layout(title="Best - 2nd Best Accuracy Gap", xaxis_title="Gap",
                              yaxis_title="Count", template="plotly_white")
            st.plotly_chart(fig, width="stretch")
            st.caption("Smaller gaps mean the top two configs perform similarly, suggesting less sensitivity to config choice.")

        # --- Config selection premium (B - M) box plot ---
        st.subheader("Config Selection Premium (B - M)")
        st.caption("How much does knowing the best config help? B(s,p) - M(s,p) per pipeline. A large premium means the pipeline depends heavily on picking the right config.")
        sp_copy = sp.copy()
        sp_copy["premium"] = sp_copy["B_acc"] - sp_copy["M_acc"]
        sp_filt = sp_copy[sp_copy["pipe_short"].isin(pipes)]
        fig = px.box(sp_filt, x="pipe_short", y="premium", color="pipe_short",
                     color_discrete_map=PIPE_COLORS,
                     category_orders={"pipe_short": pipes})
        fig.update_layout(title="Oracle Selection Premium per Pipeline",
                          yaxis_title="B(s,p) - M(s,p)", showlegend=False,
                          template="plotly_white")
        st.plotly_chart(fig, width="stretch")
        st.caption("Higher premium means picking the right config matters more for that pipeline.")

        # --- Ranking stability across metrics ---
        st.subheader("Ranking Stability Across Metrics")
        st.caption("Does the pipeline ranking change when you switch from mean-over-configs to oracle-best? Highlighted cells show rank shifts of 2+ positions.")
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
            fig = px.imshow(pivot.values, x=pivot.columns.tolist(),
                            y=pivot.index.tolist(), text_auto=".1f",
                            color_continuous_scale=COOL_LIGHT_SEQUENTIAL_REVERSED, aspect="auto")
            fig.update_layout(title="Pipeline Rank by Metric (1=best)", template="plotly_white")
            st.plotly_chart(fig, width="stretch")
            st.caption("Darker cells indicate stronger ranks. Consistent patterns across metrics indicate a robustly strong (or weak) pipeline.")

        # --- Config variance per pipeline ---
        st.subheader("Config Variance per Pipeline")
        st.caption("How sensitive is each pipeline to config choice? "
                   "A pipeline with high within-subject config SD needs careful tuning; one with low SD works with most configs.")

        if not sp.empty:
            sp_filt = sp[sp["pipe_short"].isin(pipes)].copy()
            sp_filt["order"] = sp_filt["pipe_short"].map({p: i for i, p in enumerate(PIPE_ORDER)})
            sp_filt = sp_filt.sort_values("order")

            tab_box, tab_hm, tab_bar = st.tabs(["Box Plot", "Per-Subject Heatmap", "Summary Bar"])

            # --- Box plot: distribution of sd_acc across subjects ---
            with tab_box:
                fig = px.box(sp_filt, x="pipe_short", y="sd_acc", color="pipe_short",
                             color_discrete_map=PIPE_COLORS,
                             category_orders={"pipe_short": pipes},
                             points="all",
                             hover_data=["subject"])
                fig.update_layout(
                    yaxis_title="Within-Subject Config SD",
                    showlegend=False, template="plotly_white",
                )
                st.plotly_chart(fig, width="stretch")
                st.caption("Each dot is one subject. The box shows the distribution across subjects. "
                           "A pipeline with a high box = sensitive to config choice for many subjects.")

            # --- Heatmap: Subject x Pipeline, color = sd_acc ---
            with tab_hm:
                pivot_sd = sp_filt.pivot(index="subject", columns="pipe_short", values="sd_acc")
                pivot_sd = pivot_sd[[p for p in PIPE_ORDER if p in pivot_sd.columns]]
                pivot_sd.index = [f"S{s}" for s in pivot_sd.index]

                fig = px.imshow(
                    pivot_sd.values,
                    x=pivot_sd.columns.tolist(),
                    y=pivot_sd.index.tolist(),
                    color_continuous_scale=COOL_LIGHT_INTENSITY,
                    text_auto=".3f", aspect="auto",
                    zmin=float(pivot_sd.values[np.isfinite(pivot_sd.values)].min()) if np.any(np.isfinite(pivot_sd.values)) else 0,
                    zmax=float(pivot_sd.values[np.isfinite(pivot_sd.values)].max()) if np.any(np.isfinite(pivot_sd.values)) else 0.2,
                )
                fig.update_layout(
                    template="plotly_white",
                    height=max(300, len(pivot_sd) * 35),
                    coloraxis_colorbar_title="SD",
                )
                st.plotly_chart(fig, width="stretch")
                st.caption("Deeper tones indicate higher config sensitivity for that subject-pipeline pair. "
                           "Lighter cells indicate lower sensitivity (works with most configs). "
                           "Compare columns to see which pipeline is most stable overall.")

            # --- Summary bar: mean SD per pipeline ---
            with tab_bar:
                var_agg = sp_filt.groupby("pipe_short")["sd_acc"].agg(["mean", "std"]).reset_index()
                var_agg["order"] = var_agg["pipe_short"].map({p: i for i, p in enumerate(PIPE_ORDER)})
                var_agg = var_agg.sort_values("order")

                fig = go.Figure()
                for _, row in var_agg.iterrows():
                    fig.add_trace(go.Bar(
                        x=[row["pipe_short"]],
                        y=[row["mean"]],
                        error_y=dict(type="data", array=[row["std"]], visible=True),
                        marker_color=PIPE_COLORS.get(row["pipe_short"], "#888"),
                        text=[f'{row["mean"]:.4f}'],
                        textposition="outside",
                        name=row["pipe_short"],
                        showlegend=False,
                    ))
                fig.update_layout(
                    yaxis_title="Mean Within-Subject Config SD",
                    template="plotly_white",
                )
                st.plotly_chart(fig, width="stretch")
                st.caption("Mean config SD across subjects, with error bars (+/-1 SD). "
                           "Lower = more robust to config choice.")

        # --- Stable configs table ---
        st.subheader("Most Stable Configs (Lowest Cross-Subject SD)")
        st.caption("Configs that work reliably across all subjects. Low SD means consistent performance regardless of subject.")
        if not cfg.empty and "sd_acc" in cfg.columns:
            stable = cfg[cfg["n_subject"] >= 3].nsmallest(5, "sd_acc")
            display_cols = [c for c in ["pipe_short", "config_label", "mean_acc", "sd_acc", "n_subject"]
                           if c in stable.columns]
            st.dataframe(stable[display_cols].style.format(
                {c: "{:.4f}" for c in display_cols if c in ("mean_acc", "sd_acc")}),
                hide_index=True, width="stretch")
            st.caption("Configs with at least 3 subjects, ranked by lowest standard deviation.")

    except Exception as e:
        st.error(f"Error rendering Stability page: {e}")
