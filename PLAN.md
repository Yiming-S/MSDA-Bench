# Cross-Session EEG Dashboard — Streamlit Development Plan (v2)

## Positioning

This dashboard is an **analysis companion + report generator**, not a replacement for the paper.

- The paper/report retains fixed-caliber tables with documented statistical definitions.
- The dashboard handles exploration, metric switching, drill-down, and export of the current view.
- Every number shown on screen can be traced back to a precise formula and a defined subject set.

---

## Statistical Definitions (locked)

All downstream pages derive from these definitions. They are displayed in the sidebar as a reference and enforced in code.

### Cell-Level (one per subject × pipeline × config)

| Symbol | Formula | Description |
|---|---|---|
| `acc_cfg(s,p,c)` | `cvMeanAcc` from summary pkl | Accuracy for subject s, pipeline p, config c |
| `base_cfg(s,p,c)` | `baseline` from summary pkl | No-DA baseline accuracy |
| `gain_cfg(s,p,c)` | `acc_cfg - base_cfg` | DA lift for this cell |

### Subject-Pipeline Level (one per subject × pipeline)

| Symbol | Formula | Description |
|---|---|---|
| `M(s,p)` | `mean_c acc_cfg(s,p,c)` | **Primary metric.** Mean accuracy over all matched configs. Represents pipeline's average performance on subject s without config selection bias. |
| `B(s,p)` | `max_c acc_cfg(s,p,c)` | **Oracle metric.** Best possible accuracy with perfect config selection. Supplementary only — not achievable in practice. |
| `G(s,p)` | `mean_c gain_cfg(s,p,c)` | Mean DA lift across all configs |
| `H(s,p)` | `proportion_c[acc_cfg > base_cfg]` | Fraction of configs where DA helps |

### Group-Level Summary

All `Mean / Median / SD / P5 / P25 / P75 / Min / Max` are computed on **subject-level values** (e.g., `M(1,p), M(2,p), ..., M(9,p)`), never on raw cells. This ensures subject is the independent unit.

### Matched Subject Sets

When comparing pipelines, only subjects with **all compared pipelines completed** are included. The matched set is displayed explicitly on every comparison view.

```
matched_subjects(pipeline_set) = {s : all p in pipeline_set have data for s}
```

---

## Data Layer Design

### Raw DataFrames

**`summary_df`** — one row per (dataset, subject, pipeline, config)

| Column | Type | Source |
|---|---|---|
| dataset | str | from filename |
| subject | int | from filename |
| pipeline | str | original name (MAP, BDP, etc.) |
| pipe_short | str | display name (MAP, BDP_fb, etc.) |
| method_row | int | config index 0-23 |
| feature | str | CSP / logvar / TS |
| classifier | str | lda / svm_linear |
| da | str | none / sa / pt / coral |
| config_label | str | "{feature}/{classifier}/{da}" |
| cvMeanAcc | float | accuracy (primary) |
| baseline | float | no-DA baseline |
| n_valid_pairs | int | should equal n_folds |
| n_session | int | sessions for this subject |
| session_labels | str | comma-separated session names |

**`detail_df`** — one row per (dataset, subject, pipeline, config, fold)

| Column | Type | Source |
|---|---|---|
| dataset, subject, pipeline, pipe_short | | same as summary |
| method_row | int | config index |
| pair_id | int | fold index |
| feature, classifier, da, config_label | | same as summary |
| acc_DA | float | fold-level accuracy with DA |
| baseline | float | fold-level no-DA baseline |
| elapsed_sec | float | wall-clock time for this fold |
| y_pred | np.array | predicted labels |
| y_true | np.array | true labels |
| test_label | str | target session name |
| proxy_acc | float | proxy tuning accuracy (BDP/MMP) |

Loaded **lazily** — only columns needed for the active page are materialized. `y_pred`/`y_true` arrays are only loaded on the Prediction Error page.

**`roles_df`** — one row per (dataset, subject, pipeline, config, fold, stage, session)

| Column | Type | Source |
|---|---|---|
| dataset, subject, pipeline, pipe_short | | from filename |
| method_row, pair_id | int | config and fold |
| stage | str | selection / proxy / final (BDP) or main / anchor / final (MMP) |
| session_label | str | e.g., "0train", "3test" |
| session_abs_idx | int | global session index |
| role | str | bridge / far / dropped / target / s_star / selected / ... |
| is_best | bool | is this the s* session |
| weight | float | session weight (DWP/MMP) |
| dist_est | float | MMD distance point estimate |
| dist_lwr, dist_upr | float | bootstrap CI bounds |
| partition_mode | str | normal / fallback_bridge (BDP only) |
| proxy_direction | str | far_to_bridge / bridge_to_far (BDP only) |

Loaded **on demand** — only when Mechanism Explorer or Session Tracking page is active.

### Pre-computed DataFrames

**`subject_pipeline_df`** — one row per (dataset, subject, pipeline)

| Column | Formula |
|---|---|
| M_acc | `mean_c acc_cfg(s,p,c)` |
| B_acc | `max_c acc_cfg(s,p,c)` |
| median_acc | `median_c acc_cfg(s,p,c)` |
| sd_acc | `sd_c acc_cfg(s,p,c)` |
| G_gain | `mean_c gain_cfg(s,p,c)` |
| H_helps | `proportion_c[acc_cfg > base_cfg]` |
| n_cfg | count of configs |
| best_feature | feature of B_acc config |
| best_classifier | classifier of B_acc config |
| best_da | da of B_acc config |
| best_da_gain | `B_acc - base_cfg of best config` |

**`config_agg_df`** — one row per (dataset, pipeline, config)

| Column | Formula |
|---|---|
| config_label | "feature/classifier/da" |
| mean_acc | `mean_s acc_cfg(s,p,c)` over matched subjects |
| median_acc | `median_s acc_cfg(s,p,c)` |
| sd_acc | `sd_s acc_cfg(s,p,c)` |
| mean_gain | `mean_s gain_cfg(s,p,c)` |
| helps_rate | `mean_s I[acc_cfg > base_cfg]` |
| n_subject | count of subjects with this pipeline |

**`matched_sets_df`** — tracks which subjects are in each comparison

| Column | Description |
|---|---|
| pipeline_set | frozenset of pipe_short names |
| matched_subjects | list of subject IDs |
| n_matched | count |

---

## QC Checks (run before any analysis)

These checks run at data load time. Failures are shown as warnings on the Overview page and block affected analyses.

| Check | Logic | Severity |
|---|---|---|
| Duplicate keys | `summary_df` has unique `(dataset, subject, pipeline, method_row)` | ERROR |
| Detail-summary alignment | Every `method_row` in summary has matching detail records | ERROR |
| y_pred/y_true length | `len(y_pred) == len(y_true)` for every detail record | ERROR |
| Fold completeness | Each `method_row` has all expected `pair_id` values (0 to n_folds-1) | WARNING |
| Roles coverage | BDP roles CSV has `selection` stage; MMP has `main` stage | WARNING |
| DWP weight range | All weights in [0, 1] and sum to ~1.0 | WARNING |
| MMP weight sum | Selected weights sum to ~1.0 | WARNING |
| BDP disjointness | `bridge ∩ far = ∅` in every fold | ERROR |
| Partial coverage | Flag which pipelines cannot be compared directly | INFO |

```python
def run_qc(summary_df, detail_df) -> pd.DataFrame:
    """Returns DataFrame with columns: check_name, status, message, severity"""
```

---

## Page Structure (10 pages)

### Page 1: Overview & QC

**Purpose:** Entry point. Shows what data exists, what's missing, and whether it's trustworthy.

**Sidebar:** Dataset selector only.

**Content:**

| Component | Description |
|---|---|
| Experiment summary table | Auto-generated from data: dataset, subjects, sessions, configs, total folds |
| Completion heatmap | Subject (rows) × Pipeline (columns). Green=done, Yellow=in-progress (.ckpt exists), Red=missing |
| QC badges | One badge per check. Green=PASS, Yellow=WARNING, Red=ERROR. Click to expand details |
| Matched subject panel | For each pipeline combination, show which subjects can be fairly compared. E.g., "All 6 pipelines: S3, S4, S5, S6, S9 (n=5)" |
| Data freshness | Last-modified timestamp of newest result file. [Refresh] button clears cache |

**Key design:** The "matched subject panel" is critical. It determines which subjects appear in all subsequent comparison pages. This panel should be prominently displayed and referenced by other pages.

---

### Page 2: Pipeline Benchmark

**Purpose:** Core comparison page. Answers "which pipeline is best?"

**Sidebar:**
- Dataset selector
- Metric selector: `M(s,p)` (primary) / `B(s,p)` (oracle) / `G(s,p)` (DA gain) / `H(s,p)` (helps%)
- Subject mode: `Matched only` (default, enforced) / `All available` (with warning banner)
- Pipeline visibility: checkboxes for each pipeline

**Content:**

| Component | Description |
|---|---|
| **Bar chart** | Mean of selected metric per pipeline, with SD error bars. Horizontal reference line at chance (0.5). |
| **Per-subject table** | Rows=subjects, Cols=pipelines. Values=selected metric. Winner cell highlighted green. Shows matched subject set label. |
| **Summary stats table** | Pipeline × (Mean, Median, SD, P5, P25, P75, Min, Max, n_subj). Computed on subject-level values only. |
| **Paired comparison** | Two sub-panels: (1) W/T/L matrix heatmap. (2) Mean delta heatmap with diverging colormap. |
| **Winning configs** (only when metric=B) | Table: Config × Pipeline, showing which configs win most often. |

**Fairness enforcement:** When `Matched only` is active, the header shows "Comparing on N subjects: S3, S4, S5, S6, S9". When `All available` is active, a yellow banner warns "Unmatched comparison — pipeline sample sizes differ."

---

### Page 3: Stability & Sensitivity

**Purpose:** Answer "how robust are the results?" Three angles: ranking stability across metrics, config selection premium, and pipeline sensitivity to config choice.

**Sidebar:**
- Dataset, subject mode, pipeline visibility
- Reference metric for ranking comparison

**Content:**

| Component | Description |
|---|---|
| **Best vs second-best gap** | Table: Subject × Pipeline → gap = B(s,p) - 2nd_best_acc. Small gaps mean the winner is fragile. Histogram of all gaps with median line. Answers "could a different fold split change the winner?" |
| **Config selection premium** | Per pipeline: distribution of `B(s,p) - M(s,p)` across subjects. Box plot. A large premium means the pipeline depends heavily on picking the right config; a small premium means it works with most configs. |
| **Ranking stability across metrics** | Table: under M(s,p), B(s,p), G(s,p), and median — does the pipeline ranking change? Heatmap: rows = metrics, columns = pipelines, values = rank (1-6). Cells highlighted when rank shifts by >=2 positions. |
| **Config variance per pipeline** | Bar chart: `mean_s[sd_c(acc_cfg(s,p,c))]` — the average within-subject config variance for each pipeline. Higher = more sensitive. Answers "which pipeline needs you to pick the right config?" |
| **Config sensitivity scatter** | Per pipeline: x = M(s,p), y = sd_c(acc_cfg(s,p,c)). Each dot = one subject. Shows whether high-performing subjects are also more config-sensitive. |
| **Stable configs** | Table: configs ranked by cross-subject SD (lowest SD = most stable). Shows the top 5 configs that work reliably regardless of subject. |

**Key insight:** This page reveals whether pipeline comparisons are fragile. If B(s,p) - M(s,p) is large (e.g., 0.07), the "best config" result is not representative of real-world deployment where you can't cherry-pick.

---

### Page 4: Config Explorer

**Purpose:** Understand which configs work, which don't, and which are universally good vs situationally good.

**Sidebar:**
- Dataset, subject mode, pipeline visibility
- Sort by: `mean_acc` / `mean_gain` / `sd_acc` (stability) / `helps_rate`
- Top-N slider (5 to 24)

**Content:**

| Component | Description |
|---|---|
| **Config heatmap** | 24 configs (rows, sorted) × pipelines (columns). Color = mean_acc across matched subjects. Annotated with values. Diverging colormap: red (0.50) → white (0.70) → green (0.90). |
| **Config sensitivity bar** | Per pipeline: SD of config accuracies. Higher = more sensitive to config choice. Pipelines that "need the right config" vs "work with anything." |
| **Click-to-expand** | Click a cell → shows all subjects' accuracy for that config × pipeline. Bar chart + table. |
| **Config stability ranking** | Table: Config × (mean_acc, sd_acc, worst_subject_acc, best_subject_acc). Identifies configs that are universally good (high mean, low sd) vs situationally good (high mean, high sd). |
| **Config dominance analysis** | For each pipeline: what fraction of total accuracy is contributed by the top-3 configs vs the bottom-3? Stacked bar. Answers "does this pipeline have a few star configs or a broad portfolio?" |
| **Feature contribution** | Grouped bar: mean accuracy per feature (CSP/logvar/TS) within each pipeline. Shows which features each pipeline benefits from most. |

---

### Page 5: Subject Explorer

**Purpose:** Deep-dive into individual subjects. Explains why group-level and subject-level conclusions differ.

**Sidebar:**
- Dataset selector
- Subject selector (single subject dropdown)

**Content:**

| Component | Description |
|---|---|
| **Subject profile** | N sessions, session labels, trials per session. Overall accuracy range. |
| **All-configs table** | 24 configs × 6 pipelines. Full accuracy matrix for this subject. Sortable by any column. |
| **Pipeline summary** for this subject | M(s,p), B(s,p), G(s,p), H(s,p) per pipeline. Bar chart. |
| **Best vs mean gap** | Per pipeline: B(s,p) - M(s,p). Shows how much config selection matters for this subject. |
| **Subject heterogeneity** | Radar chart or bar: how much this subject's pipeline ranking differs from the group average. If very different, flag as "atypical subject." |

---

### Page 5: DA Analysis

**Purpose:** Answer "does domain adaptation actually help?"

**Sidebar:**
- Dataset, subject mode, pipeline visibility
- Feature filter: CSP / logvar / TS / All
- DA filter: sa / pt / coral (checkboxes)

**Content:**

| Component | Description |
|---|---|
| **Overall DA gain** | Per pipeline: mean G(s,p), median, SD. Bar chart with 0-line. |
| **DA method comparison** | Paired vs `none`: for each DA method, show mean delta, positive rate, harm rate. Grouped bar. |
| **DA gain heatmap** | Config (rows) × Pipeline (columns). Color = mean gain. Red = harmful, green = helpful. Annotated. |
| **DA gain distribution** | Violin plot per pipeline, showing the spread of per-fold gains. With 0-line. |
| **Feature × DA interaction** | Grouped bar: for each (feature, DA) combo, show mean gain and harm rate. Answers "does SA help CSP but hurt TS?" |
| **Harm rate table** | Per DA method: fraction of cells where DA strictly hurts (gain < -0.001). Highlights dangerous combos. |

---

### Page 6: Mechanism Explorer

**Purpose:** Understand HOW each pipeline uses sessions. BDP bridge/far, MMP selection, DWP weighting.

**Sidebar:**
- Dataset selector
- Subject selector (single)
- Pipeline selector: BDP_fb / BDP_bf / MMP_mta / MMP_moe / DWP
- Config selector (method_row dropdown)
- Fold selector (slider)

**Content:**

| Component | Description |
|---|---|
| **Session role diagram** | Scatter: x=distance, y=jitter. Color=role. Size=1/distance. Labels=session names. Target at x=0 as red star. Shows partition_mode badge. |
| **Session table** | For selected fold: Session, Role, Distance, CI [lwr, upr], Weight (if applicable). |
| **All-folds comparison** | Compact table: Fold × (Bridge set, Far set, Dropped, Mode). Shows role stability across folds. |
| **Fold stability analysis** | Per session: how often it appears in each role across folds. Heatmap: session (rows) × fold (columns), color=role. |
| **Distance-performance scatter** | x = mean distance (or bridge-far gap), y = fold accuracy. Per fold. Shows whether distance predicts performance. |
| **Bridge count distribution** (BDP only) | Bar chart: how often 1/2/3/4/... sessions are bridge. |
| **Weight distribution** (DWP only) | Bar chart per fold showing session weights. neff metric displayed. |

---

### Page 8: Target Session Analysis

**Purpose:** Understand which target sessions are hard/easy, and whether different pipelines have different session-level strengths. Directly corresponds to report Section 4d.

**Sidebar:**
- Dataset selector
- Subject mode (matched only / all available)
- Pipeline visibility checkboxes
- Metric: `mean acc_DA` / `median acc_DA` / `sd acc_DA`

**Content:**

| Component | Description |
|---|---|
| **Accuracy by target session heatmap** | Rows = target sessions (0train, 1train, ..., 4test), Columns = pipelines. Color = mean accuracy across all configs and matched subjects. Annotated with values. Clearly shows which sessions are hard. |
| **Accuracy by target session grouped bar** | Same data as heatmap but as grouped bar chart. Each group = one target session, bars = pipelines. Error bars = SD across subjects. |
| **Session difficulty ranking** | Table: Target Session, Mean Acc (all pipelines), SD, Hardest Pipeline, Easiest Pipeline. Sorted by mean acc ascending (hardest first). |
| **Pipeline × session interaction** | Line chart: x = target session (ordered), y = accuracy, one line per pipeline. Shows whether pipelines have different strengths on different sessions. Crossing lines = interaction. |
| **Session-level pipeline ranking stability** | Per target session: what is the pipeline ranking? Table: Target Session × Pipeline → rank. Highlight sessions where ranking differs from overall ranking. |
| **Early vs late session analysis** | Compare accuracy on early sessions (0train, 1train) vs late sessions (2train+). These often differ due to trial count (120 vs 160 in BNCI) or learning effects. Paired bar chart. |
| **Trial count effect** | If sessions have different trial counts, scatter: x = n_trials in target session, y = mean accuracy. Shows whether accuracy scales with test set size. |

**Key insight:** In BNCI2014-004, 0train/1train (120 trials) are consistently harder (~0.63) than 2train/3test/4test (160 trials, ~0.70). This page makes that pattern interactive and reveals whether some pipelines handle early sessions better.

---

### Page 9: Prediction Error Analysis

**Purpose:** Go beyond accuracy. Understand what the classifiers get wrong.

**Sidebar:**
- Dataset, subject selector (single or multi), pipeline selector

**Content:**

| Component | Description |
|---|---|
| **Per-class accuracy** | Bar: left_hand vs right_hand per pipeline. Bias = |left - right|. |
| **Confusion matrix** | 2×2 heatmap for selected subject × pipeline × config. Normalized by true class. |
| **Error consistency** | For same subject, across pipelines: do different pipelines misclassify the same trials? Jaccard overlap of error sets. |
| **Hard sessions** | Per target session: mean accuracy across all pipelines. Identifies sessions that are universally difficult. |
| **Hard subjects** | Per subject: mean accuracy across all pipelines and configs. Identifies subjects where all methods struggle. |

**Note:** This page loads `y_pred`/`y_true` arrays, which are heavier. Only load on demand.

---

### Page 10: Efficiency & Progress

**Purpose:** Timing analysis and dataset growth tracking.

**Sidebar:**
- Dataset selector
- Subject selector (multi)

**Content:**

| Component | Description |
|---|---|
| **Pipeline timing bar** | Mean sec/fold per pipeline with median marker. |
| **Accuracy-time frontier** | Scatter: x=mean time/fold, y=mean accuracy. Each dot=one config×pipeline. Pareto frontier highlighted. Ideal=top-left. |
| **Per-subject timing table** | Subject × Pipeline total time (minutes). Heatmap coloring. |
| **Top 10 slowest configs** | Table with pipeline, config, mean time/fold. |
| **Progress tracker** (for growing datasets) | Line chart: as subjects are added, how does the pipeline ranking change? Shows stability of conclusions. x=number of subjects included, y=mean M(s,p) for each pipeline. |
| **Completion forecast** | Based on completed timing, estimate remaining time for incomplete subjects. |

---

## Analysis Lenses (cross-cutting, not separate pages)

These lenses are available as toggles or filters on relevant pages:

| Lens | Where Applied | What It Does |
|---|---|---|
| **Fairness** | Pages 2-5 | Toggle between `matched only` and `all available`. Show warning banner when unmatched. |
| **Robustness** | Pages 2-3 | Show not just mean but median, IQR, P5. Flag if mean and median diverge. |
| **Sensitivity** | Pages 2-3 | Show SD of configs within each pipeline. Highlight pipelines that need careful config selection. |
| **Stability** | Pages 2-4 | Show whether pipeline ranking changes under different metrics (M vs B vs G). |
| **Mechanism** | Page 6 | Link distance/weight/role to accuracy. Does the mechanism explain the performance? |
| **Failure** | Pages 5, 9 | Highlight subjects where ALL pipelines are below threshold. Why? |
| **Growth** | Page 10 | How fragile are current conclusions? Would 5 more subjects change the ranking? |
| **Reproducibility** | All pages | One-click export: current view → CSV table + PNG figure + caption text. |

---

## Data Loading Architecture (`data_loader.py`)

### Lazy Loading Strategy

```python
class DataStore:
    """Central data store with lazy loading and caching."""
    
    def __init__(self, data_dir: str, dataset_name: str):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self._summary_df = None
        self._detail_df = None
        self._roles_cache = {}
        self._derived = None
    
    @property
    def summary_df(self) -> pd.DataFrame:
        """Loaded on first access. ~1MB for BNCI, ~5MB for Stieger."""
        if self._summary_df is None:
            self._summary_df = self._load_summaries()
        return self._summary_df
    
    @property
    def detail_df(self) -> pd.DataFrame:
        """Loaded on first access. ~50MB for BNCI.
        Heavy columns (y_pred, y_true, session_roles) loaded as object refs,
        only materialized when accessed."""
        if self._detail_df is None:
            self._detail_df = self._load_details()
        return self._detail_df
    
    def get_roles(self, subject: int, pipeline_file: str) -> pd.DataFrame:
        """Loaded on demand per subject × pipeline. Cached after first load."""
        key = (subject, pipeline_file)
        if key not in self._roles_cache:
            self._roles_cache[key] = self._load_single_roles(subject, pipeline_file)
        return self._roles_cache[key]
    
    @property
    def derived(self) -> dict:
        """Pre-computed tables. Built once from summary_df."""
        if self._derived is None:
            self._derived = self._compute_derived()
        return self._derived
    
    def _compute_derived(self) -> dict:
        return {
            'subject_pipeline': self._build_subject_pipeline_df(),
            'config_agg': self._build_config_agg_df(),
            'matched_sets': self._build_matched_sets(),
            'completion': self._build_completion_matrix(),
            'qc_results': self._run_qc(),
        }
    
    def get_matched_subjects(self, pipelines: list) -> list:
        """Return subjects that have data for ALL specified pipelines."""
        key = frozenset(pipelines)
        return self.derived['matched_sets'].get(key, [])
```

### Caching with Streamlit

```python
@st.cache_resource
def get_data_store(data_dir: str, dataset_name: str) -> DataStore:
    return DataStore(data_dir, dataset_name)

# In app.py:
store = get_data_store(DATA_DIRS[dataset], dataset)

# Refresh button:
if st.sidebar.button("Refresh Data"):
    st.cache_resource.clear()
    st.rerun()
```

---

## Constants & Config (`utils.py`)

```python
# Pipeline mapping
PIPE_MAP = {
    'MAP': 'MAP', 'DWP': 'DWP',
    'MMP_merge_then_adapt': 'MMP_mta', 'MMP_moe': 'MMP_moe',
    'BDP': 'BDP_fb', 'BDP_bridge_to_far': 'BDP_bf'
}
PIPE_ORDER = ['MAP', 'DWP', 'MMP_mta', 'MMP_moe', 'BDP_fb', 'BDP_bf']

# Colors
PIPE_COLORS = {
    'MAP': '#1f77b4', 'DWP': '#ff7f0e',
    'MMP_mta': '#2ca02c', 'MMP_moe': '#d62728',
    'BDP_fb': '#9467bd', 'BDP_bf': '#8c564b'
}
ROLE_COLORS = {
    'bridge': '#2ecc71', 'far': '#e67e22', 'dropped': '#95a5a6',
    'target': '#e74c3c', 's_star': '#2ecc71', 'selected': '#3498db',
    'overlap_only': '#f39c12', 'not_used': '#bdc3c7'
}

# Metric definitions (for sidebar display)
METRIC_DEFS = {
    'M(s,p)': 'Mean accuracy over all configs (primary)',
    'B(s,p)': 'Best config accuracy (oracle, supplementary)',
    'G(s,p)': 'Mean DA gain over all configs',
    'H(s,p)': 'Fraction of configs where DA helps',
}

# Dataset paths
DATA_DIRS = {
    'bnci004': '/Users/yiming/Documents/WORK/Ubuntu result/new/bnci004/',
    'stieger2021': '/Users/yiming/Documents/WORK/Ubuntu result/new/stieger2021/',
}
```

---

## Implementation Order

### Phase 1: Foundation (Day 1)

**Goal:** Data loads correctly, QC passes, matched subjects computed.

- [ ] `utils.py`: Constants, PIPE_MAP, colors, format helpers
- [ ] `data_loader.py`:
  - [ ] `DataStore.__init__()` and directory scanning
  - [ ] `_load_summaries()`: scan for `{dataset}_*_*.pkl`, exclude detail/roles/ckpt, concatenate, add pipe_short and config_label columns
  - [ ] `_build_completion_matrix()`: Subject × Pipeline boolean matrix
  - [ ] `_build_matched_sets()`: for each pipeline combination, compute matched subjects
  - [ ] `_build_subject_pipeline_df()`: compute M(s,p), B(s,p), G(s,p), H(s,p)
  - [ ] `_run_qc()`: duplicate key check, n_valid_pairs check
  - [ ] Test with BNCI data: verify 1296 summary rows, 9 subjects, 6 pipelines, 24 configs
- [ ] `app.py`: Sidebar with dataset selector, page routing via `st.navigation()`
- [ ] `pages/1_overview.py`:
  - [ ] Experiment summary table
  - [ ] Completion heatmap (plotly imshow)
  - [ ] QC badge panel
  - [ ] Matched subject panel
  - [ ] Data freshness + refresh button
- [ ] `.streamlit/config.toml`: wide layout, theme
- [ ] Verify: `streamlit run app.py` shows overview with BNCI data

### Phase 2: Pipeline Benchmark (Day 2)

**Goal:** Core comparison page fully functional.

- [ ] `data_loader.py`:
  - [ ] `_load_details()`: load detail pkl files (lazy columns for y_pred/y_true)
  - [ ] `_build_config_agg_df()`: mean/median/sd per config × pipeline
- [ ] `pages/2_pipeline_benchmark.py`:
  - [ ] Sidebar: metric selector (M/B/G/H), subject mode toggle, pipeline checkboxes
  - [ ] Bar chart with error bars (plotly)
  - [ ] Per-subject table with winner highlighting (st.dataframe + styler)
  - [ ] Summary stats table (Mean/Median/SD/P5/P25/P75/Min/Max)
  - [ ] Matched subjects label
  - [ ] W/T/L matrix heatmap
  - [ ] Mean delta heatmap (diverging colormap)
  - [ ] Winning configs table (only visible when metric=B)
  - [ ] Fairness banner when `All available` mode active
- [ ] `utils.py`:
  - [ ] `make_bar_with_error()`
  - [ ] `make_wtl_matrix()`
  - [ ] `make_paired_heatmap()`
  - [ ] `highlight_winner()` styler
- [ ] Test: toggle M vs B, verify numbers match PDF report

### Phase 3: Stability, Config & Subject Explorer (Day 3)

**Goal:** Stability analysis, config deep-dive, and individual subject exploration.

- [ ] `pages/3_stability_sensitivity.py`:
  - [ ] Best vs second-best gap: table + histogram
  - [ ] Config selection premium: B(s,p) - M(s,p) box plot per pipeline
  - [ ] Ranking stability across metrics: rank heatmap (metrics × pipelines)
  - [ ] Config variance per pipeline: bar chart of mean within-subject SD
  - [ ] Config sensitivity scatter: M(s,p) vs sd_c per subject
  - [ ] Stable configs table: top 5 lowest cross-subject SD
- [ ] `pages/4_config_explorer.py`:
  - [ ] Config heatmap (24 × 6) with sorting options
  - [ ] Config dominance analysis: top-3 vs bottom-3 contribution
  - [ ] Feature contribution grouped bar
  - [ ] Click-to-expand: per-subject breakdown for selected cell
  - [ ] Config stability ranking table
- [ ] `pages/5_subject_explorer.py`:
  - [ ] Subject profile card
  - [ ] Full 24 × 6 accuracy table (sortable)
  - [ ] Pipeline summary bar chart for this subject
  - [ ] Best vs mean gap per pipeline
  - [ ] Subject heterogeneity indicator
- [ ] Test: verify stability metrics match report Section 8, config heatmap matches 3b-i

### Phase 4: DA, Mechanism & Target Session (Day 4)

**Goal:** DA deep-dive, session mechanism visualization, and target session analysis.

- [ ] `pages/6_da_analysis.py`:
  - [ ] Overall DA gain bar chart
  - [ ] DA method paired comparison vs none
  - [ ] Config × Pipeline gain heatmap
  - [ ] Violin distribution
  - [ ] Feature × DA interaction grouped bar
  - [ ] Harm rate table
- [ ] `data_loader.py`:
  - [ ] `_load_single_roles()`: load one roles CSV on demand
- [ ] `pages/7_mechanism_explorer.py`:
  - [ ] Session role diagram (scatter)
  - [ ] Session table per fold
  - [ ] All-folds comparison table
  - [ ] Fold stability heatmap (session × fold, color=role)
  - [ ] Distance-performance scatter
  - [ ] Bridge count distribution (BDP)
  - [ ] Weight distribution (DWP)
- [ ] `pages/8_target_session.py`:
  - [ ] Accuracy by target session heatmap
  - [ ] Accuracy by target session grouped bar with error bars
  - [ ] Session difficulty ranking table
  - [ ] Pipeline × session interaction line chart
  - [ ] Session-level ranking stability table
  - [ ] Early vs late session comparison
  - [ ] Trial count effect scatter (if applicable)
- [ ] `utils.py`:
  - [ ] `make_session_diagram()`
  - [ ] `make_da_heatmap()`
  - [ ] `make_target_session_heatmap()`
- [ ] Test: navigate all mechanism pages, verify target session matches report 4d

### Phase 5: Prediction Error, Efficiency, Polish (Day 5-6)

**Goal:** Remaining pages, multi-dataset, export, testing.

- [ ] `pages/9_prediction_error.py`:
  - [ ] Per-class accuracy bar chart
  - [ ] Confusion matrix heatmap
  - [ ] Error consistency across pipelines (Jaccard of error sets)
  - [ ] Hard sessions table
  - [ ] Hard subjects table
- [ ] `pages/10_efficiency_progress.py`:
  - [ ] Pipeline timing bar chart
  - [ ] Accuracy-time frontier scatter with Pareto
  - [ ] Per-subject timing table
  - [ ] Top 10 slowest configs
  - [ ] Progress tracker (line chart for growing datasets)
  - [ ] Completion forecast
- [ ] Multi-dataset support:
  - [ ] Stieger2021: handle partial completion, variable sessions (7 vs 11)
  - [ ] Test all 10 pages with both datasets
- [ ] Export:
  - [ ] Download buttons: table → CSV, figure → PNG
  - [ ] Current view caption text (auto-generated description of filters applied)
- [ ] Polish:
  - [ ] Consistent colors across all pages
  - [ ] Loading spinners for heavy computations
  - [ ] Error handling for missing files
  - [ ] Help tooltips on complex controls

---

## File Structure

```
Viso/Streamlit/
├── PLAN.md                       # This file
├── app.py                        # Entry point: sidebar, dataset selector, page routing
├── data_loader.py                # DataStore class: lazy loading, caching, derived tables
├── utils.py                      # Constants, colors, format helpers, shared plot functions
├── pages/
│   ├── 1_overview.py             # Overview & QC
│   ├── 2_pipeline_benchmark.py   # Pipeline comparison (core page)
│   ├── 3_stability_sensitivity.py # Best-vs-2nd gap, ranking stability, config variance
│   ├── 4_config_explorer.py      # Config heatmap, dominance, feature contribution
│   ├── 5_subject_explorer.py     # Single-subject deep-dive
│   ├── 6_da_analysis.py          # DA gain: heatmap, violin, harm rate
│   ├── 7_mechanism_explorer.py   # Session roles, distance, weights (BDP/MMP/DWP)
│   ├── 8_target_session.py       # Accuracy by target session, hard sessions, interactions
│   ├── 9_prediction_error.py     # Per-class, confusion, error consistency
│   └── 10_efficiency_progress.py # Timing, Pareto frontier, progress tracker
├── .streamlit/
│   └── config.toml               # Theme, layout, server settings
└── requirements.txt              # streamlit, plotly, pandas, numpy
```

---

## Dependencies

```
streamlit>=1.30
plotly>=5.18
pandas>=2.0
numpy>=1.24
```

```bash
pip install streamlit plotly
cd "/Users/yiming/Agent/PhD working/agent result/Viso/Streamlit"
streamlit run app.py
```

---

## Testing Checklist

### Data Integrity
- [ ] BNCI: 1296 summary rows (9 subj × 6 pipe × 24 cfg)
- [ ] BNCI: 6480 detail rows (1296 × 5 folds)
- [ ] Stieger partial: loads available, shows --- for missing
- [ ] Matched subjects computed correctly
- [ ] QC catches injected errors (duplicate key, mismatched y_pred length)

### Pipeline Benchmark
- [ ] M(s,p) numbers match report 3a.1 / 3b.1
- [ ] B(s,p) numbers match report 3a / 3b
- [ ] W/T/L matrix matches report 3d
- [ ] Switching metric recomputes everything
- [ ] Matched-only vs All-available changes subject count

### Stability & Sensitivity
- [ ] Best vs 2nd-best gap matches report 8a
- [ ] B(s,p) - M(s,p) distribution is correct per pipeline
- [ ] Ranking stability: rank shifts highlighted when metric changes
- [ ] Config variance: mean within-subject SD computed correctly

### Config Explorer
- [ ] Heatmap values match report 3b-i
- [ ] Sorting works (by accuracy, gain, stability)
- [ ] Click-to-expand shows correct per-subject values
- [ ] Config dominance: top-3 vs bottom-3 percentages sum correctly

### Subject Explorer
- [ ] All 24 × 6 values match raw pkl data
- [ ] M(s,p) and B(s,p) match subject_pipeline_df

### DA Analysis
- [ ] Overall gain matches report 6a
- [ ] Config gain heatmap matches report 6b
- [ ] Harm rate correctly computed

### Mechanism Explorer
- [ ] BDP roles match appendix A in report
- [ ] MMP roles match appendix B in report
- [ ] DWP weights match detail['weights']

### Target Session Analysis
- [ ] Accuracy by target matches report 4d values exactly
- [ ] Session difficulty ranking is correct (hardest first)
- [ ] Early vs late session split is dataset-appropriate
- [ ] Pipeline × session interaction lines cross-checked with raw data

### Prediction Error
- [ ] Per-class accuracy matches report 3b-ii / 10b
- [ ] Confusion matrix sums to correct total
- [ ] Error consistency Jaccard is symmetric

### Cross-Dataset
- [ ] Switching dataset reloads everything
- [ ] Stieger2021 partial data: no crashes, correct matched sets
- [ ] Variable session counts handled (7 vs 11 in Stieger)
- [ ] Export produces valid CSV/PNG
