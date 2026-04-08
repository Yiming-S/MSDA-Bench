"""Page 9: Prediction Error Analysis."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS


def render(store, dataset):
    st.header("9. Prediction Error Analysis")
    st.markdown("Go beyond accuracy numbers. See which classes are harder, where classifiers disagree, and which subjects/sessions are universally difficult.")
    try:
        ddf = store.detail_df
        sdf = store.summary_df
        if ddf.empty:
            st.warning("No detail data available.")
            return

        pipes = [p for p in PIPE_ORDER if p in ddf["pipe_short"].unique()]
        subjects = sorted(ddf["subject"].unique()) if "subject" in ddf.columns else []

        # --- Per-class accuracy (if y_true / y_pred available) ---
        has_preds = "y_true" in ddf.columns and "y_pred" in ddf.columns

        if has_preds:
            st.subheader("Per-Class Accuracy")
            st.caption("Accuracy broken down by class (left hand vs right hand). A large gap reveals systematic class bias in the classifier.")
            # Sidebar: select subject and pipeline for detailed view
            col_subj, col_pipe = st.columns(2)
            with col_subj:
                sel_subj = st.selectbox("Subject", subjects,
                                        format_func=lambda s: f"S{s}", key="err_subj")
            with col_pipe:
                sel_pipe = st.selectbox("Pipeline", pipes, key="err_pipe")

            # Expand y_true/y_pred arrays into flat lists per pipeline
            # Each row has y_true=array, y_pred=array — need to iterate rows
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
                    class_acc_rows.append({"pipe_short": pipe, "class": str(cls),
                                           "accuracy": correct, "n": int(mask.sum())})
            if class_acc_rows:
                ca_df = pd.DataFrame(class_acc_rows)
                fig = px.bar(ca_df, x="class", y="accuracy", color="pipe_short",
                             barmode="group", color_discrete_map=PIPE_COLORS,
                             category_orders={"pipe_short": pipes})
                fig.update_layout(title="Per-Class Accuracy by Pipeline",
                                  yaxis_title="Accuracy", template="plotly_white")
                st.plotly_chart(fig, width="stretch")
                st.caption("Accuracy broken down by true class label. Imbalance reveals class-specific weaknesses.")

            # --- Confusion matrix for selected subject/pipeline ---
            st.subheader(f"Confusion Matrix: S{sel_subj} / {sel_pipe}")
            sel_data = ddf[(ddf["subject"] == sel_subj) & (ddf["pipe_short"] == sel_pipe)]
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
                    fig = px.imshow(cm, x=[str(c) for c in classes],
                                    y=[str(c) for c in classes],
                                    color_continuous_scale="Blues", text_auto=True,
                                    aspect="auto")
                    fig.update_layout(title=f"Confusion Matrix (S{sel_subj}, {sel_pipe})",
                                      xaxis_title="Predicted", yaxis_title="True",
                                      template="plotly_white")
                    st.plotly_chart(fig, width="stretch")
                    st.caption("Diagonal = correct predictions. Off-diagonal = misclassifications.")
                else:
                    st.info("No prediction arrays for selected subject/pipeline.")
            else:
                st.info("No data for selected subject/pipeline.")
        else:
            st.info("Prediction columns (y_true, y_pred) not found. Showing accuracy-based analysis.")
            sel_subj = st.selectbox("Subject", subjects,
                                    format_func=lambda s: f"S{s}", key="err_subj")

        # --- Hard sessions table ---
        st.subheader("Hard Sessions")
        st.caption("Target sessions with the lowest mean accuracy across all pipelines and configs. These sessions are inherently difficult regardless of method choice.")
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
            st.dataframe(hard.style.format({"Mean Accuracy": "{:.4f}"}),
                         hide_index=True, width="stretch")
            st.caption("Sessions with lowest mean accuracy across all pipelines and subjects.")

        # --- Hard subjects table ---
        st.subheader("Hard Subjects")
        st.caption("Subjects where all pipelines struggle. May indicate poor signal quality or atypical brain patterns.")
        if not sdf.empty and "cvMeanAcc" in sdf.columns:
            subj_acc = sdf.groupby("subject")["cvMeanAcc"].mean().reset_index()
            subj_acc = subj_acc.sort_values("cvMeanAcc").head(10)
            subj_acc.columns = ["Subject", "Mean Accuracy"]
            subj_acc["Subject"] = subj_acc["Subject"].apply(lambda s: f"S{s}")
            st.dataframe(subj_acc.style.format({"Mean Accuracy": "{:.4f}"}),
                         hide_index=True, width="stretch")
            st.caption("Subjects with lowest mean accuracy, indicating inherently difficult EEG patterns.")

    except Exception as e:
        st.error(f"Error rendering Prediction Error page: {e}")
