"""Page 9: Prediction Error Analysis."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, style_figure


def render(store, dataset):
    st.header("Error Analysis")
    st.markdown(
        "Go beyond accuracy numbers. See which classes are harder, where classifiers "
        "disagree, and which subjects/sessions are universally difficult."
    )
    try:
        ddf = store.detail_df
        sdf = store.summary_df
        if ddf.empty:
            st.warning("No detail data available.")
            return

        pipes = [p for p in PIPE_ORDER if p in ddf["pipe_short"].unique()]
        subjects = sorted(ddf["subject"].unique()) if "subject" in ddf.columns else []

        # ── Takeaway ──────────────────────────────────────────────────────────
        if not sdf.empty and "cvMeanAcc" in sdf.columns:
            subj_acc = sdf.groupby("subject")["cvMeanAcc"].mean()
            if not subj_acc.empty:
                hardest_subj = subj_acc.idxmin()
                easiest_subj = subj_acc.idxmax()
                st.info(
                    f"Hardest subject: **S{hardest_subj}** (acc: {subj_acc.min():.4f}). "
                    f"Easiest: **S{easiest_subj}** (acc: {subj_acc.max():.4f})."
                )

        # ── Per-class accuracy (if y_true / y_pred available) ─────────────────
        has_preds = "y_true" in ddf.columns and "y_pred" in ddf.columns

        if has_preds:
            with st.container(border=True):
                st.subheader("Per-Class Accuracy")
                st.caption(
                    "Accuracy by class across all pipelines. "
                    "A large gap reveals systematic class bias."
                )
                col_subj, col_pipe = st.columns(2)
                with col_subj:
                    sel_subj = st.selectbox(
                        "Subject", subjects,
                        format_func=lambda s: f"S{s}", key="err_subj",
                    )
                with col_pipe:
                    sel_pipe = st.selectbox("Pipeline", pipes, key="err_pipe")

                class_acc_rows = []
                for pipe in pipes:
                    pipe_data = ddf[ddf["pipe_short"] == pipe]
                    if pipe_data.empty:
                        continue
                    all_yt, all_yp = [], []
                    for _, row in pipe_data.iterrows():
                        yt = row.get("y_true")
                        yp = row.get("y_pred")
                        if yt is not None and yp is not None:
                            all_yt.extend(np.asarray(yt).ravel())
                            all_yp.extend(np.asarray(yp).ravel())
                    if not all_yt:
                        continue
                    all_yt = np.array(all_yt)
                    all_yp = np.array(all_yp)
                    for cls in sorted(np.unique(all_yt)):
                        mask = all_yt == cls
                        correct = float(np.mean(all_yp[mask] == cls))
                        class_acc_rows.append({
                            "pipe_short": pipe, "class": str(cls),
                            "accuracy": correct, "n": int(mask.sum()),
                        })
                if class_acc_rows:
                    ca_df = pd.DataFrame(class_acc_rows)
                    fig = px.bar(
                        ca_df, x="class", y="accuracy", color="pipe_short",
                        barmode="group", color_discrete_map=PIPE_COLORS,
                        category_orders={"pipe_short": pipes},
                    )
                    fig.update_layout(
                        title="Per-Class Accuracy by Pipeline",
                        yaxis_title="Accuracy",
                    )
                    style_figure(fig)
                    st.plotly_chart(fig, use_container_width=True)

            # ── Confusion matrix ──────────────────────────────────────────────
            with st.container(border=True):
                st.subheader(f"Confusion Matrix: S{sel_subj} / {sel_pipe}")
                sel_data = ddf[
                    (ddf["subject"] == sel_subj) & (ddf["pipe_short"] == sel_pipe)
                ]
                if not sel_data.empty:
                    all_yt, all_yp = [], []
                    for _, row in sel_data.iterrows():
                        yt = row.get("y_true")
                        yp = row.get("y_pred")
                        if yt is not None and yp is not None:
                            all_yt.extend(np.asarray(yt).ravel())
                            all_yp.extend(np.asarray(yp).ravel())
                    if all_yt:
                        all_yt = np.array(all_yt)
                        all_yp = np.array(all_yp)
                        classes = sorted(np.unique(all_yt))
                        cm = np.zeros((len(classes), len(classes)), dtype=int)
                        cls_to_idx = {c: i for i, c in enumerate(classes)}
                        for yt_val, yp_val in zip(all_yt, all_yp):
                            if yt_val in cls_to_idx and yp_val in cls_to_idx:
                                cm[cls_to_idx[yt_val], cls_to_idx[yp_val]] += 1
                        fig = px.imshow(
                            cm,
                            x=[str(c) for c in classes],
                            y=[str(c) for c in classes],
                            color_continuous_scale="Blues",
                            text_auto=True, aspect="auto",
                        )
                        fig.update_layout(
                            title=f"Confusion Matrix (S{sel_subj}, {sel_pipe})",
                            xaxis_title="Predicted", yaxis_title="True",
                        )
                        style_figure(fig)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Diagonal = correct. Off-diagonal = misclassifications.")
                    else:
                        st.info("No prediction arrays for selected subject/pipeline.")
                else:
                    st.info("No data for selected subject/pipeline.")
        else:
            st.info(
                "Prediction columns (y_true, y_pred) not found. "
                "Showing accuracy-based analysis only."
            )
            sel_subj = st.selectbox(
                "Subject", subjects,
                format_func=lambda s: f"S{s}", key="err_subj",
            )

        # ── Hard sessions & hard subjects (side by side) ──────────────────────
        col_sess, col_subj_hard = st.columns(2)

        with col_sess:
            with st.container(border=True):
                st.subheader("Hard Sessions")
                st.caption("Target sessions with lowest mean accuracy.")
                acc_col = "acc_DA" if "acc_DA" in ddf.columns else None
                target_col = None
                for c in ["test_label", "target_session", "pair_id"]:
                    if c in ddf.columns:
                        target_col = c
                        break
                if acc_col and target_col:
                    sess_acc = ddf.groupby([target_col, "pipe_short"])[acc_col].mean().reset_index()
                    hard = sess_acc.groupby(target_col)[acc_col].mean().reset_index()
                    hard = hard.sort_values(acc_col).head(10)
                    hard.columns = ["Session", "Mean Accuracy"]
                    st.dataframe(
                        hard.style.format({"Mean Accuracy": "{:.4f}"}),
                        hide_index=True, use_container_width=True,
                    )
                else:
                    st.info("Insufficient columns for session-level analysis.")

        with col_subj_hard:
            with st.container(border=True):
                st.subheader("Hard Subjects")
                st.caption("Subjects where all pipelines struggle.")
                if not sdf.empty and "cvMeanAcc" in sdf.columns:
                    subj_acc_df = sdf.groupby("subject")["cvMeanAcc"].mean().reset_index()
                    subj_acc_df = subj_acc_df.sort_values("cvMeanAcc").head(10)
                    subj_acc_df.columns = ["Subject", "Mean Accuracy"]
                    subj_acc_df["Subject"] = subj_acc_df["Subject"].apply(lambda s: f"S{s}")
                    st.dataframe(
                        subj_acc_df.style.format({"Mean Accuracy": "{:.4f}"}),
                        hide_index=True, use_container_width=True,
                    )

    except Exception as e:
        st.error(f"Error rendering Prediction Error page: {e}")
