"""
Utility constants and helper functions for the EEG Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

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
    "MAP": "#2F6FED",       # cobalt
    "DWP": "#0F8C94",       # teal
    "MMP_mta": "#4C8B6F",   # sea green
    "MMP_moe": "#7B7DD6",   # muted indigo
    "BDP_fb": "#5E7FA6",    # steel blue
    "BDP_bf": "#AF7287",    # dusty rose
}

ROLE_COLORS = {
    "bridge": "#4B7FDC",
    "far": "#9AA9BC",
    "dropped": "#C97A66",
    "target": "#2E8B7D",
    "s_star": "#2F6FED",
    "selected": "#5A95E6",
    "overlap_only": "#BAC7D8",
    "not_used": "#D8E1EC",
}

PLOTLY_TEMPLATE = "cool_light_lab"
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


def register_plotly_template():
    """Register the Cool Light Lab Plotly template and make it the default."""
    if PLOTLY_TEMPLATE in pio.templates:
        pio.templates.default = PLOTLY_TEMPLATE
        # Preserve existing page code that still passes template="plotly_white".
        pio.templates["plotly_white"] = pio.templates[PLOTLY_TEMPLATE]
        return

    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(
            family="Avenir Next, Segoe UI, Helvetica Neue, Arial, sans-serif",
            size=14,
            color="#132238",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=list(PIPE_COLORS.values()),
        title=dict(
            x=0.02,
            xanchor="left",
            font=dict(size=21, color="#132238"),
        ),
        margin=dict(l=16, r=16, t=56, b=16),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#D6E2F0",
            font=dict(color="#132238", size=12),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="#D6E2F0",
            borderwidth=1,
            font=dict(size=12, color="#40556E"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#C8D5E4",
            tickcolor="#C8D5E4",
            ticks="outside",
            title_font=dict(color="#40556E"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#D7E0EA",
            gridwidth=1,
            zeroline=False,
            showline=True,
            linecolor="#C8D5E4",
            tickcolor="#C8D5E4",
            ticks="outside",
            title_font=dict(color="#40556E"),
        ),
        coloraxis=dict(
            colorbar=dict(
                outlinewidth=0,
                tickfont=dict(size=11, color="#40556E"),
                titlefont=dict(size=12, color="#40556E"),
            )
        ),
    )
    template.data.bar = [
        go.Bar(marker_line_width=0, opacity=0.92),
    ]
    template.data.box = [
        go.Box(marker_opacity=0.8, line_width=1.4),
    ]
    template.data.scatter = [
        go.Scatter(marker=dict(line=dict(width=0.8, color="rgba(255,255,255,0.85)"))),
    ]
    template.data.heatmap = [
        go.Heatmap(colorbar=dict(outlinewidth=0)),
    ]

    pio.templates[PLOTLY_TEMPLATE] = template
    pio.templates.default = PLOTLY_TEMPLATE
    # Alias the house style onto plotly_white so existing update_layout calls
    # adopt the same look without page-by-page template churn.
    pio.templates["plotly_white"] = template

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
        e.g. ``'background-color: rgba(47, 111, 237, 0.14)'``.

    Returns
    -------
    list[str]
        A list of CSS strings, one per cell.
    """
    if props == "":
        props = TABLE_HIGHLIGHT
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
        template=PLOTLY_TEMPLATE,
        bargap=0.3,
    )
    return fig


def make_heatmap(z, x_labels, y_labels, title,
                 colorscale=COOL_LIGHT_SEQUENTIAL, zmin=None, zmax=None,
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
        template=PLOTLY_TEMPLATE,
        xaxis=dict(side="bottom"),
    )
    return fig
