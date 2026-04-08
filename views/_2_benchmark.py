import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PIPE_ORDER, PIPE_COLORS, format_acc, make_heatmap

def render(store, dataset):
    st.header("2. Pipeline Benchmark")
    st.markdown("Compare all pipelines head-to-head on matched subjects. Switch between metrics (mean-over-configurations vs oracle best) to see how rankings change.")

    sp = store.derived['subject_pipeline']
    completion = store.derived['completion']
    all_pipes = [p for p in PIPE_ORDER if p in sp['pipe_short'].unique()]

    # Metric selector — in main area, not sidebar
    metric_options = {
        'M(s,p) — Mean over all configurations': 'M_acc',
        'B(s,p) — Best configuration (oracle)': 'B_acc',
        'G(s,p) — Mean DA gain': 'G_gain',
        'H(s,p) — DA helps rate': 'H_helps',
    }

    metric_descriptions = {
        'M(s,p) — Mean over all configurations':
            'Primary metric. For each subject, average accuracy across all 24 configurations. '
            'No configuration selection bias — represents pipeline performance you would get '
            'without knowing which configuration is best.',
        'B(s,p) — Best configuration (oracle)':
            'Oracle metric. For each subject, the single best configuration accuracy. '
            'This is an upper bound — in practice you cannot know the best configuration '
            'without labeled test data. Use as a supplement to M(s,p).',
        'G(s,p) — Mean DA gain':
            'Mean domain adaptation lift. For each subject, average of (acc_DA - baseline) '
            'across all configurations. Positive = DA helps on average, negative = DA hurts.',
        'H(s,p) — DA helps rate':
            'Fraction of configurations where DA improves accuracy (acc_DA > baseline). '
            'A value of 0.6 means DA helps in 60% of configurations for that subject.',
    }

    col_subj, col_pipe, col_metric = st.columns([1, 2, 2])
    with col_subj:
        subject_mode = st.radio("Subject mode", ["Matched only", "All available"])
    with col_pipe:
        visible_pipes = st.multiselect("Pipelines", all_pipes, default=all_pipes)
    with col_metric:
        metric = st.selectbox("Metric", list(metric_options.keys()))

    if not visible_pipes:
        st.warning("Select at least one pipeline.")
        return

    st.info(metric_descriptions[metric])

    col_name = metric_options[metric]

    # Determine subjects
    if subject_mode == "Matched only":
        subjects = store.get_matched_subjects(visible_pipes)
        if not subjects:
            st.warning("No subjects have all selected pipelines completed.")
            return
        st.caption(f"Matched comparison on {len(subjects)} subjects: {', '.join(f'S{s}' for s in subjects)}")
    else:
        subjects = sorted(sp['subject'].unique())
        st.warning("Unmatched comparison — pipeline sample sizes may differ.")

    # Filter data
    mask = sp['subject'].isin(subjects) & sp['pipe_short'].isin(visible_pipes)
    data = sp[mask].copy()

    # --- Bar chart with error bars ---
    st.subheader("Pipeline Comparison")
    st.caption("Bar chart of each pipeline's average performance with error bars (+/-1 SD across subjects). "
               "The metric selected above determines what is being compared.")

    agg = data.groupby('pipe_short')[col_name].agg(['mean','std','median','count']).reset_index()
    agg = agg[agg['pipe_short'].isin(visible_pipes)]
    # Maintain order
    agg['order'] = agg['pipe_short'].map({p:i for i,p in enumerate(PIPE_ORDER)})
    agg = agg.sort_values('order')

    fig = go.Figure()
    for _, row in agg.iterrows():
        p = row['pipe_short']
        color = PIPE_COLORS.get(p, '#888')
        ci = 1.96 * row['std'] / np.sqrt(row['count']) if row['count'] > 1 else 0
        # CI line
        fig.add_trace(go.Scatter(
            x=[row['mean'] - ci, row['mean'] + ci],
            y=[p, p],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo='skip',
        ))
        # IQR whiskers
        vals = data[data['pipe_short'] == p][col_name].dropna().values
        if len(vals) >= 4:
            q25, q75 = np.percentile(vals, [25, 75])
            fig.add_trace(go.Scatter(
                x=[q25, q75],
                y=[p, p],
                mode='lines',
                line=dict(color=color, width=8),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip',
            ))
        # Mean dot
        fig.add_trace(go.Scatter(
            x=[row['mean']],
            y=[p],
            mode='markers+text',
            marker=dict(color=color, size=12, line=dict(color='white', width=1.5)),
            text=[f"{row['mean']:.4f}"],
            textposition='middle right',
            textfont=dict(size=11),
            showlegend=False,
            hovertemplate=f"<b>{p}</b><br>Mean: {row['mean']:.4f}<br>SD: {row['std']:.4f}<br>n={int(row['count'])}<extra></extra>",
        ))

    x_min = float(agg['mean'].min() - agg['std'].max() * 1.5)
    x_max = float(agg['mean'].max() + agg['std'].max() * 1.5)
    fig.update_layout(
        xaxis_title=metric.split('—')[0].strip(),
        height=max(250, len(agg) * 55),
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(categoryorder='array', categoryarray=list(reversed(agg['pipe_short'].tolist()))),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Dot = mean across subjects. Thick band = IQR (P25–P75). Thin line = 95% CI of the mean.")

    # --- Per-subject table ---
    st.subheader("Per-Subject Values")
    st.caption("Each row is one subject, each column is one pipeline. The green cell marks the winner for that subject. Use this to see which subjects drive the overall ranking.")
    pivot = data.pivot(index='subject', columns='pipe_short', values=col_name)
    pivot = pivot[[p for p in PIPE_ORDER if p in pivot.columns]]
    pivot.index = [f"S{s}" for s in pivot.index]

    # Add winner column
    pivot['Winner'] = pivot.idxmax(axis=1)

    st.dataframe(pivot.style.format({c: "{:.4f}" for c in pivot.columns if c != 'Winner'})
                 .highlight_max(axis=1, subset=[c for c in pivot.columns if c != 'Winner'],
                               props='background-color: #90EE90;'),
                 width="stretch")

    # --- Summary statistics ---
    st.subheader("Summary Statistics")
    st.caption("Descriptive statistics computed across subjects (not across configurations). "
               "Each subject contributes one number per pipeline, then Mean/Median/SD/percentiles are computed on those.")

    stats_rows = []
    for p in [pp for pp in PIPE_ORDER if pp in visible_pipes]:
        vals = data[data['pipe_short']==p][col_name].dropna().values
        if len(vals) == 0:
            continue
        stats_rows.append({
            'Pipeline': p, 'Mean': np.mean(vals), 'Median': np.median(vals),
            'SD': np.std(vals, ddof=1) if len(vals)>1 else 0,
            'P5': np.percentile(vals, 5), 'P25': np.percentile(vals, 25),
            'P75': np.percentile(vals, 75), 'Min': np.min(vals), 'Max': np.max(vals),
            '# Subjects': len(vals)
        })
    stats_df = pd.DataFrame(stats_rows)
    st.dataframe(stats_df.style.format({c: "{:.4f}" for c in stats_df.columns if c not in ('Pipeline','# Subjects')}),
                 width="stretch")

    # --- Paired comparison ---
    if len(visible_pipes) >= 2:
        st.subheader("Paired Comparison")
        st.caption("Head-to-head: for each pair of pipelines, count how many subjects one beats the other. The delta heatmap shows the average accuracy difference (green = row is better).")

        # Build W/T/L and delta matrices
        wtl_data = []
        delta_data = []

        for p1 in visible_pipes:
            wtl_row = {}
            delta_row = {}
            for p2 in visible_pipes:
                if p1 == p2:
                    wtl_row[p2] = '---'
                    delta_row[p2] = 0.0
                    continue
                w, t, l = 0, 0, 0
                deltas = []
                for s in subjects:
                    v1 = data[(data['subject']==s) & (data['pipe_short']==p1)][col_name]
                    v2 = data[(data['subject']==s) & (data['pipe_short']==p2)][col_name]
                    if len(v1)==0 or len(v2)==0:
                        continue
                    a1, a2 = v1.values[0], v2.values[0]
                    deltas.append(a1 - a2)
                    if a1 > a2 + 0.0001: w += 1
                    elif a2 > a1 + 0.0001: l += 1
                    else: t += 1
                wtl_row[p2] = f"{w}/{t}/{l}"
                delta_row[p2] = np.mean(deltas) if deltas else 0
            wtl_data.append(wtl_row)
            delta_data.append(delta_row)

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Win/Tie/Loss (row beats column)")
            wtl_df = pd.DataFrame(wtl_data, index=visible_pipes)
            st.dataframe(wtl_df)

        with col2:
            st.caption("Mean delta (row - column)")
            delta_df = pd.DataFrame(delta_data, index=visible_pipes, columns=visible_pipes)

            fig = px.imshow(delta_df.values,
                           x=visible_pipes, y=visible_pipes,
                           color_continuous_scale='RdYlGn',
                           zmin=-0.02, zmax=0.02,
                           text_auto='.4f',
                           aspect='auto')
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")

    # --- Winning configs (only for B metric) ---
    if col_name == 'B_acc':
        st.subheader("Winning Configurations")
        st.caption("Which feature/classifier/DA combination achieves the highest accuracy most often. "
                   "Only shown when the Oracle Best metric is selected.")

        best_configs = store.summary_df[
            store.summary_df['subject'].isin(subjects) &
            store.summary_df['pipe_short'].isin(visible_pipes)
        ].copy()
        best_configs['acc'] = pd.to_numeric(best_configs['cvMeanAcc'], errors='coerce')

        idx = best_configs.groupby(['subject','pipe_short'])['acc'].idxmax()
        winners = best_configs.loc[idx]

        win_pivot = winners.groupby(['config_label','pipe_short']).agg(
            count=('subject','count'),
            subjects=('subject', lambda x: ', '.join(f'S{s}' for s in sorted(x)))
        ).reset_index()

        win_table = win_pivot.pivot(index='config_label', columns='pipe_short', values='subjects').fillna('---')
        win_table = win_table[[p for p in PIPE_ORDER if p in win_table.columns]]
        st.dataframe(win_table, width="stretch")

    # --- Feature Contribution (always shown) ---
    st.subheader("Feature Contribution")
    st.caption("Mean accuracy broken down by feature type (CSP, logvar, TS) for each pipeline. "
               "Shows which feature extraction method each pipeline benefits from most, using all configurations.")

    cfg = store.derived.get('config_agg')
    if cfg is not None and not cfg.empty and 'feature' in cfg.columns:
        feat_agg = cfg.groupby(['pipe_short', 'feature'])['mean_acc'].mean().reset_index()
        feat_agg = feat_agg[feat_agg['pipe_short'].isin(visible_pipes)]
        if not feat_agg.empty:
            # Two views: heatmap (primary) + bar chart (secondary)
            tab_hm, tab_bar = st.tabs(["Heatmap", "Bar Chart"])

            # --- Heatmap view (better for small differences) ---
            with tab_hm:
                pivot_feat = feat_agg.pivot(index='feature', columns='pipe_short', values='mean_acc')
                pivot_feat = pivot_feat[[p for p in PIPE_ORDER if p in pivot_feat.columns]]

                fig_hm = px.imshow(
                    pivot_feat.values,
                    x=pivot_feat.columns.tolist(),
                    y=pivot_feat.index.tolist(),
                    color_continuous_scale='RdYlGn',
                    text_auto='.4f',
                    aspect='auto',
                    zmin=float(pivot_feat.values.min()) - 0.01,
                    zmax=float(pivot_feat.values.max()) + 0.01,
                )
                fig_hm.update_layout(
                    template='plotly_white', height=250,
                    coloraxis_colorbar_title='Acc',
                )
                st.plotly_chart(fig_hm, width="stretch")
                st.caption("Heatmap amplifies small differences. Green = higher accuracy. "
                           "Compare within each row (which pipeline uses this feature best) "
                           "and within each column (which feature this pipeline benefits from most).")

            # --- Bar chart view (zoomed y-axis) ---
            with tab_bar:
                y_min = float(feat_agg['mean_acc'].min())
                y_max = float(feat_agg['mean_acc'].max())
                y_pad = max((y_max - y_min) * 0.3, 0.01)

                fig_feat = px.bar(feat_agg, x='feature', y='mean_acc', color='pipe_short',
                                  barmode='group', color_discrete_map=PIPE_COLORS,
                                  category_orders={'pipe_short': [p for p in PIPE_ORDER if p in visible_pipes]},
                                  text=feat_agg['mean_acc'].apply(lambda v: f'{v:.4f}'))
                fig_feat.update_traces(textposition='outside', textfont_size=9)
                fig_feat.update_layout(
                    yaxis_title='Mean Accuracy',
                    yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),
                    template='plotly_white',
                )
                st.plotly_chart(fig_feat, width="stretch")
                st.caption("Bar chart with zoomed y-axis to amplify differences. "
                           "Note: y-axis does NOT start at 0.")
