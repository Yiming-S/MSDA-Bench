"""
Utility constants and helper functions for the EEG Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Pipeline name mappings
# ---------------------------------------------------------------------------

# Maps internal/code pipeline names to short display names
PIPE_MAP = {
    "MAP": "MAP",
    "DWP": "DWP",
    "MMP_merge_then_adapt": "MMP_mta",
    "MMP_moe": "MMP_moe",
    "BDP": "BDP_fb",
    "BDP_bridge_to_far": "BDP_bf",
}

# Reverse mapping: display name -> file/code name
PIPE_FILE_MAP = {v: k for k, v in PIPE_MAP.items()}

# Canonical display order
PIPE_ORDER = ["MAP", "DWP", "MMP_mta", "MMP_moe", "BDP_fb", "BDP_bf"]

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

PIPE_COLORS = {
    "MAP": "#3B82F6",       # blue
    "DWP": "#F59E0B",       # amber
    "MMP_mta": "#10B981",   # emerald
    "MMP_moe": "#EF4444",   # red
    "BDP_fb": "#8B5CF6",    # violet
    "BDP_bf": "#F43F5E",    # rose
}

ROLE_COLORS = {
    "bridge": "#3B82F6",
    "far": "#F59E0B",
    "dropped": "#EF4444",
    "target": "#10B981",
    "s_star": "#8B5CF6",
    "selected": "#06B6D4",
    "overlap_only": "#EAB308",
    "not_used": "#94A3B8",
}

DEGRADE_COLORS = {
    "pure": "#10B981",          # emerald – normal BDP
    "partial": "#F59E0B",       # amber – some pairs degraded
    "full_degrade": "#EF4444",  # red – fully degraded to MAP
    "n/a": "#94A3B8",           # slate – non-BDP pipelines
}

TABLE_HIGHLIGHT = "background-color: rgba(47, 111, 237, 0.14); color: #183A63; font-weight: 700;"

COOL_LIGHT_SEQUENTIAL = [
    [0.00, "#F7FAFD"],
    [0.20, "#E6EEF8"],
    [0.40, "#CCDDF4"],
    [0.60, "#A8C0E8"],
    [0.80, "#6F97DB"],
    [1.00, "#2F6FED"],
]

COOL_LIGHT_SEQUENTIAL_REVERSED = list(reversed(COOL_LIGHT_SEQUENTIAL))

COOL_LIGHT_INTENSITY = [
    [0.00, "#F7FAFD"],
    [0.25, "#E7EEF7"],
    [0.50, "#CCD7E6"],
    [0.75, "#92A9C6"],
    [1.00, "#4D678D"],
]

COOL_LIGHT_DIVERGING = [
    [0.00, "#C97A66"],
    [0.20, "#E8BFB4"],
    [0.48, "#F8F9FB"],
    [0.52, "#F8F9FB"],
    [0.80, "#ABC3EA"],
    [1.00, "#2F6FED"],
]

COOL_LIGHT_COMPLETION = [
    [0.00, "#E5ECF3"],
    [0.49, "#E5ECF3"],
    [0.50, "#7EA2D9"],
    [1.00, "#7EA2D9"],
]

COOL_LIGHT_PANDAS_CMAP = "Blues"

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

METRIC_DEFS = {
    "M(s,p)": "Mean accuracy of pipeline p on subject s across folds.",
    "B(s,p)": "Best single-fold accuracy of pipeline p on subject s.",
    "G(s,p)": "Gain of pipeline p over the baseline for subject s.",
    "H(s,p)": "Harmonic mean of precision and recall (F1-like) for pipeline p on subject s.",
}

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_DIRS = {
    "bnci004": "/Users/yiming/Documents/WORK/Ubuntu result/new/bnci004/",
    "stieger2021": "/Users/yiming/Documents/WORK/Ubuntu result/new/stieger2021/",
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def format_acc(val, decimals=4):
    """Format an accuracy value to a fixed number of decimal places.

    Returns '---' for NaN / None values.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    return f"{val:.{decimals}f}"


def highlight_winner(s, props=""):
    """Pandas Styler function that highlights the maximum value in each row.

    Parameters
    ----------
    s : pd.Series
        A row from a DataFrame (applied via ``Styler.apply(..., axis=1)``).
    props : str
        CSS property string applied to the winning cell,
        e.g. ``'background-color: #d4edda'``.

    Returns
    -------
    list[str]
        A list of CSS strings, one per cell.
    """
    if props == "":
        props = "background-color: #d4edda; font-weight: bold"
    numeric = pd.to_numeric(s, errors="coerce")
    is_max = numeric == numeric.max()
    return [props if v else "" for v in is_max]


def make_wtl_matrix(best_dict, subjects, pipelines):
    """Build Win / Tie / Loss and mean-delta matrices.

    Parameters
    ----------
    best_dict : dict[str, dict[str, float]]
        ``best_dict[subject][pipeline]`` = accuracy value.
    subjects : list[str]
        List of subject identifiers.
    pipelines : list[str]
        List of pipeline display names to compare pairwise.

    Returns
    -------
    wtl_df : pd.DataFrame
        DataFrame indexed and columned by *pipelines* where each cell is a
        string ``"W / T / L"`` showing how the row pipeline compares against
        the column pipeline across all subjects.
    delta_df : pd.DataFrame
        Same shape, but each cell is the mean accuracy delta
        (row pipeline minus column pipeline).
    """
    n = len(pipelines)
    wtl_data = [[None] * n for _ in range(n)]
    delta_data = [[0.0] * n for _ in range(n)]

    for i, p_row in enumerate(pipelines):
        for j, p_col in enumerate(pipelines):
            if i == j:
                wtl_data[i][j] = "---"
                delta_data[i][j] = 0.0
                continue
            wins, ties, losses = 0, 0, 0
            deltas = []
            for subj in subjects:
                val_row = best_dict.get(subj, {}).get(p_row, np.nan)
                val_col = best_dict.get(subj, {}).get(p_col, np.nan)
                if np.isnan(val_row) or np.isnan(val_col):
                    continue
                diff = val_row - val_col
                deltas.append(diff)
                if diff > 1e-9:
                    wins += 1
                elif diff < -1e-9:
                    losses += 1
                else:
                    ties += 1
            wtl_data[i][j] = f"{wins} / {ties} / {losses}"
            delta_data[i][j] = float(np.mean(deltas)) if deltas else 0.0

    wtl_df = pd.DataFrame(wtl_data, index=pipelines, columns=pipelines)
    delta_df = pd.DataFrame(delta_data, index=pipelines, columns=pipelines)
    return wtl_df, delta_df


def make_bar_with_error(data_df, x_col, y_col, error_col, color_col,
                        title, yaxis_title):
    """Create a Plotly bar chart with error bars.

    Parameters
    ----------
    data_df : pd.DataFrame
        Source data containing the columns referenced below.
    x_col : str
        Column name for the x-axis categories.
    y_col : str
        Column name for the bar heights.
    error_col : str
        Column name for the symmetric error bar values.
    color_col : str
        Column name whose values map to ``PIPE_COLORS``.
    title : str
        Chart title.
    yaxis_title : str
        Y-axis label.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    for _, row in data_df.iterrows():
        color = PIPE_COLORS.get(row[color_col], "#333333")
        fig.add_trace(go.Bar(
            x=[row[x_col]],
            y=[row[y_col]],
            error_y=dict(type="data", array=[row[error_col]], visible=True),
            marker_color=color,
            name=row[color_col],
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        xaxis_title="",
        template="plotly_white",
        bargap=0.3,
    )
    return fig


def make_heatmap(z, x_labels, y_labels, title,
                 colorscale="RdYlGn", zmin=None, zmax=None,
                 text_auto=True):
    """Create a Plotly heatmap figure.

    Parameters
    ----------
    z : array-like (2-D)
        Matrix of values.
    x_labels : list[str]
        Column labels.
    y_labels : list[str]
        Row labels.
    title : str
        Chart title.
    colorscale : str
        Plotly colour scale name.
    zmin, zmax : float or None
        Explicit colour range bounds.
    text_auto : bool
        Whether to annotate cells with their values.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    text = None
    if text_auto:
        z_arr = np.asarray(z, dtype=float)
        text = [[format_acc(v, 4) for v in row] for row in z_arr]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        text=text,
        texttemplate="%{text}" if text_auto else None,
        hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis=dict(side="bottom"),
    )
    return fig


def style_figure(fig, height=None):
    """Apply unified dashboard styling to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=13),
        title_font=dict(size=16, color="#1E293B"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=50, b=50),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, system-ui, sans-serif",
            bordercolor="#E2E8F0",
        ),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig
