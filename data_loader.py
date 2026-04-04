"""
Data loading module for the Streamlit EEG dashboard.

Provides a DataStore class that lazily loads experiment results from
pickle and CSV files, caches them, and computes derived analytics.
"""

import os
import glob
import pickle
import itertools
import logging

import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Pipeline name mappings
# ---------------------------------------------------------------------------

PIPE_MAP = {
    "MAP": "MAP",
    "DWP": "DWP",
    "MMP_merge_then_adapt": "MMP_mta",
    "MMP_moe": "MMP_moe",
    "BDP": "BDP_fb",
    "BDP_bridge_to_far": "BDP_bf",
}

PIPE_ORDER = ["MAP", "DWP", "MMP_mta", "MMP_moe", "BDP_fb", "BDP_bf"]

# Reverse mapping: short display name -> file/code name
PIPE_FILE_MAP = {v: k for k, v in PIPE_MAP.items()}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached factory
# ---------------------------------------------------------------------------

@st.cache_resource(ttl=300)
def get_data_store(data_dir: str, dataset_name: str) -> "DataStore":
    """Return a cached DataStore instance for the given directory and dataset."""
    return DataStore(data_dir, dataset_name)


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------

class DataStore:
    """Lazy-loading store for experiment summary, detail, and role data."""

    def __init__(self, data_dir: str, dataset_name: str):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self._summary_df: pd.DataFrame | None = None
        self._detail_df: pd.DataFrame | None = None
        self._roles_cache: dict[tuple, pd.DataFrame] = {}
        self._derived: dict | None = None

    # -- Lazy properties -----------------------------------------------------

    @property
    def summary_df(self) -> pd.DataFrame:
        if self._summary_df is None:
            self._summary_df = self._load_summaries()
        return self._summary_df

    @property
    def detail_df(self) -> pd.DataFrame:
        if self._detail_df is None:
            self._detail_df = self._load_details()
        return self._detail_df

    @property
    def derived(self) -> dict:
        if self._derived is None:
            self._derived = self._compute_derived()
        return self._derived

    # -- Loading methods -----------------------------------------------------

    def _load_summaries(self) -> pd.DataFrame:
        """Load all per-subject summary pickle files and concatenate."""
        pattern = os.path.join(self.data_dir, f"{self.dataset_name}_*_*.pkl")
        all_files = glob.glob(pattern)

        # Exclude detail, roles, and checkpoint files
        exclude_tokens = ("_detail", "_roles", ".ckpt")
        pkl_files = [
            f for f in all_files
            if not any(tok in os.path.basename(f) for tok in exclude_tokens)
        ]

        if not pkl_files:
            logger.warning("No summary pkl files found for pattern: %s", pattern)
            return pd.DataFrame()

        frames = []
        for fpath in pkl_files:
            try:
                df = pd.read_pickle(fpath)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", fpath, exc)
                continue

            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            # Parse subject and pipeline from filename
            # Expected: {dataset}_{subject}_{pipeline}.pkl
            basename = os.path.splitext(os.path.basename(fpath))[0]
            parts = basename.split("_", 1)  # split off dataset prefix
            if len(parts) < 2:
                continue
            remainder = parts[1]  # e.g. "3_MAP" or "3_BDP_bridge_to_far"

            # Find which pipeline name matches the end of the remainder
            subject_str, pipe_code = self._parse_subject_pipeline(remainder)
            if subject_str is None:
                logger.warning("Cannot parse subject/pipeline from: %s", basename)
                continue

            df = df.copy()
            df["subject"] = int(subject_str)
            df["dataset"] = self.dataset_name
            df["pipe_short"] = PIPE_MAP.get(pipe_code, pipe_code)

            # Build config label from feature / classifier / da columns
            if all(c in df.columns for c in ("feature", "classifier", "da")):
                df["config_label"] = (
                    df["feature"].astype(str) + "/"
                    + df["classifier"].astype(str) + "/"
                    + df["da"].astype(str)
                )
            else:
                df["config_label"] = "unknown"

            # Ensure cvMeanAcc is numeric
            if "cvMeanAcc" in df.columns:
                df["cvMeanAcc"] = pd.to_numeric(df["cvMeanAcc"], errors="coerce")

            frames.append(df)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        return result

    def _load_details(self) -> pd.DataFrame:
        """Load all detail pickle files and concatenate into a DataFrame."""
        pattern = os.path.join(self.data_dir, f"{self.dataset_name}_*_detail.pkl")
        detail_files = glob.glob(pattern)

        if not detail_files:
            logger.warning("No detail pkl files found for pattern: %s", pattern)
            return pd.DataFrame()

        all_records: list[dict] = []
        for fpath in detail_files:
            try:
                with open(fpath, "rb") as fh:
                    records = pickle.load(fh)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", fpath, exc)
                continue

            if isinstance(records, list):
                all_records.extend(records)
            else:
                logger.warning("Unexpected type in %s: %s", fpath, type(records))

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # Add pipe_short
        if "pipeline" in df.columns:
            df["pipe_short"] = df["pipeline"].map(PIPE_MAP).fillna(df["pipeline"])

        # Build config label
        if all(c in df.columns for c in ("feature", "classifier", "da")):
            df["config_label"] = (
                df["feature"].astype(str) + "/"
                + df["classifier"].astype(str) + "/"
                + df["da"].astype(str)
            )
        else:
            df["config_label"] = "unknown"

        return df

    def get_roles(self, subject, pipeline_short: str) -> pd.DataFrame:
        """Load session roles CSV for a given subject and pipeline.

        Results are cached in memory.  Returns an empty DataFrame if the
        file does not exist.
        """
        cache_key = (subject, pipeline_short)
        if cache_key in self._roles_cache:
            return self._roles_cache[cache_key]

        pipe_file = PIPE_FILE_MAP.get(pipeline_short, pipeline_short)
        fname = f"{self.dataset_name}_{subject}_{pipe_file}_roles.csv"
        fpath = os.path.join(self.data_dir, fname)

        if os.path.isfile(fpath):
            try:
                df = pd.read_csv(fpath)
            except Exception as exc:
                logger.warning("Failed to read roles CSV %s: %s", fpath, exc)
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        self._roles_cache[cache_key] = df
        return df

    # -- Derived analytics ---------------------------------------------------

    def _compute_derived(self) -> dict:
        """Compute all derived analytics from the summary data."""
        sdf = self.summary_df

        if sdf.empty:
            return {
                "completion": pd.DataFrame(),
                "subject_pipeline": pd.DataFrame(),
                "config_agg": pd.DataFrame(),
                "qc_results": pd.DataFrame(
                    columns=["check_name", "status", "message", "severity"]
                ),
                "matched_sets": {},
            }

        return {
            "completion": self._build_completion(sdf),
            "subject_pipeline": self._build_subject_pipeline(sdf),
            "config_agg": self._build_config_agg(sdf),
            "qc_results": self._run_qc(),
            "matched_sets": self._build_matched_sets(),
        }

    def _build_completion(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """Boolean matrix: subjects x PIPE_ORDER, True if data exists."""
        if sdf.empty:
            return pd.DataFrame()

        subjects = sorted(sdf["subject"].unique())
        pipes_present = sdf.groupby("subject")["pipe_short"].apply(set)

        rows = []
        for subj in subjects:
            present = pipes_present.get(subj, set())
            rows.append({p: (p in present) for p in PIPE_ORDER})

        return pd.DataFrame(rows, index=subjects)

    def _build_subject_pipeline(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """Per (subject, pipeline) aggregate metrics."""
        if sdf.empty or "cvMeanAcc" not in sdf.columns:
            return pd.DataFrame()

        rows = []
        for (subj, pipe), grp in sdf.groupby(["subject", "pipe_short"]):
            acc = grp["cvMeanAcc"].dropna()
            if acc.empty:
                continue

            baseline_vals = grp["baseline"].dropna() if "baseline" in grp.columns else pd.Series(dtype=float)
            gains = (grp["cvMeanAcc"] - grp["baseline"]) if "baseline" in grp.columns else pd.Series(dtype=float)

            best_idx = acc.idxmax()
            best_row = grp.loc[best_idx]
            best_baseline = best_row.get("baseline", np.nan)

            rows.append({
                "subject": subj,
                "pipe_short": pipe,
                "M_acc": acc.mean(),
                "B_acc": acc.max(),
                "median_acc": acc.median(),
                "sd_acc": acc.std(ddof=1) if len(acc) > 1 else 0.0,
                "G_gain": gains.mean() if not gains.empty else np.nan,
                "H_helps": (
                    (grp["cvMeanAcc"] > grp["baseline"]).mean()
                    if "baseline" in grp.columns else np.nan
                ),
                "n_cfg": len(grp),
                "best_feature": best_row.get("feature", ""),
                "best_classifier": best_row.get("classifier", ""),
                "best_da": best_row.get("da", ""),
                "best_da_gain": (
                    float(acc.max() - best_baseline)
                    if pd.notna(best_baseline) else np.nan
                ),
            })

        return pd.DataFrame(rows)

    def _build_config_agg(self, sdf: pd.DataFrame) -> pd.DataFrame:
        """Aggregate accuracy stats per configuration across subjects."""
        if sdf.empty or "cvMeanAcc" not in sdf.columns:
            return pd.DataFrame()

        group_cols = ["pipe_short", "feature", "classifier", "da", "config_label"]
        available_cols = [c for c in group_cols if c in sdf.columns]
        if not available_cols:
            return pd.DataFrame()

        rows = []
        for keys, grp in sdf.groupby(available_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            info = dict(zip(available_cols, keys))

            acc = grp["cvMeanAcc"].dropna()
            gains = (
                (grp["cvMeanAcc"] - grp["baseline"])
                if "baseline" in grp.columns else pd.Series(dtype=float)
            )

            info["mean_acc"] = acc.mean() if not acc.empty else np.nan
            info["median_acc"] = acc.median() if not acc.empty else np.nan
            info["sd_acc"] = acc.std(ddof=1) if len(acc) > 1 else 0.0
            info["mean_gain"] = gains.mean() if not gains.empty else np.nan
            info["helps_rate"] = (
                (grp["cvMeanAcc"] > grp["baseline"]).mean()
                if "baseline" in grp.columns else np.nan
            )
            info["n_subject"] = grp["subject"].nunique()
            rows.append(info)

        return pd.DataFrame(rows)

    # -- Matched subjects ----------------------------------------------------

    def get_matched_subjects(self, pipelines: list[str]) -> list:
        """Return subjects that have data for ALL specified pipelines."""
        sdf = self.summary_df
        if sdf.empty:
            return []

        subject_pipes = sdf.groupby("subject")["pipe_short"].apply(set)
        required = set(pipelines)
        return sorted([
            subj for subj, pipes in subject_pipes.items()
            if required.issubset(pipes)
        ])

    def _build_matched_sets(self) -> dict:
        """Pre-compute matched subject sets for common pipeline combinations."""
        sdf = self.summary_df
        if sdf.empty:
            return {}

        available_pipes = [p for p in PIPE_ORDER if p in sdf["pipe_short"].values]
        result: dict[frozenset, list] = {}

        # Full set of all available pipelines
        if available_pipes:
            key = frozenset(available_pipes)
            result[key] = self.get_matched_subjects(available_pipes)

        # All pairs
        for combo in itertools.combinations(available_pipes, 2):
            key = frozenset(combo)
            if key not in result:
                result[key] = self.get_matched_subjects(list(combo))

        # All triples (useful for comparison views)
        if len(available_pipes) >= 3:
            for combo in itertools.combinations(available_pipes, 3):
                key = frozenset(combo)
                if key not in result:
                    result[key] = self.get_matched_subjects(list(combo))

        # Each individual pipeline
        for p in available_pipes:
            key = frozenset([p])
            if key not in result:
                result[key] = self.get_matched_subjects([p])

        return result

    # -- Quality checks ------------------------------------------------------

    def _run_qc(self) -> pd.DataFrame:
        """Run basic quality checks on the summary data."""
        sdf = self.summary_df
        checks: list[dict] = []

        # Check 1: duplicate keys
        if not sdf.empty:
            key_cols = ["subject", "pipe_short", "config_label"]
            available_keys = [c for c in key_cols if c in sdf.columns]
            if available_keys:
                dup_mask = sdf.duplicated(subset=available_keys, keep=False)
                n_dups = dup_mask.sum()
                checks.append({
                    "check_name": "duplicate_keys",
                    "status": "FAIL" if n_dups > 0 else "PASS",
                    "message": (
                        f"{n_dups} duplicate rows on {available_keys}"
                        if n_dups > 0 else "No duplicate keys found"
                    ),
                    "severity": "warning" if n_dups > 0 else "info",
                })

        # Check 2: n_valid_pairs consistency within each pipeline
        if not sdf.empty and "n_valid_pairs" in sdf.columns:
            inconsistent = []
            for pipe, grp in sdf.groupby("pipe_short"):
                unique_counts = grp.groupby("subject")["n_valid_pairs"].nunique()
                bad_subjects = unique_counts[unique_counts > 1]
                if not bad_subjects.empty:
                    inconsistent.append(
                        f"{pipe}: subjects {list(bad_subjects.index)}"
                    )
            checks.append({
                "check_name": "n_valid_pairs_consistency",
                "status": "FAIL" if inconsistent else "PASS",
                "message": (
                    "Inconsistent n_valid_pairs: " + "; ".join(inconsistent)
                    if inconsistent
                    else "n_valid_pairs consistent within each (subject, pipeline)"
                ),
                "severity": "warning" if inconsistent else "info",
            })

        # Check 3: missing cvMeanAcc values
        if not sdf.empty and "cvMeanAcc" in sdf.columns:
            n_missing = sdf["cvMeanAcc"].isna().sum()
            checks.append({
                "check_name": "missing_cvMeanAcc",
                "status": "FAIL" if n_missing > 0 else "PASS",
                "message": (
                    f"{n_missing} rows with missing cvMeanAcc"
                    if n_missing > 0 else "All cvMeanAcc values present"
                ),
                "severity": "warning" if n_missing > 0 else "info",
            })

        # Check 4: completion coverage
        if not sdf.empty:
            subjects = sdf["subject"].unique()
            pipes = sdf["pipe_short"].unique()
            total_possible = len(subjects) * len(PIPE_ORDER)
            total_present = sdf.groupby(["subject", "pipe_short"]).ngroups
            pct = 100.0 * total_present / total_possible if total_possible else 0
            checks.append({
                "check_name": "completion_coverage",
                "status": "PASS" if pct >= 80 else "WARN",
                "message": (
                    f"{total_present}/{total_possible} subject-pipeline "
                    f"combinations present ({pct:.1f}%)"
                ),
                "severity": "info" if pct >= 80 else "warning",
            })

        if not checks:
            checks.append({
                "check_name": "no_data",
                "status": "WARN",
                "message": "No summary data loaded; QC skipped",
                "severity": "warning",
            })

        return pd.DataFrame(checks)

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _parse_subject_pipeline(remainder: str):
        """Parse '{subject}_{pipeline_code}' from the remainder after dataset prefix.

        Tries to match against known pipeline code names (longest first)
        so that multi-part names like 'BDP_bridge_to_far' are handled correctly.

        Returns (subject_str, pipeline_code) or (None, None).
        """
        # Sort pipeline codes longest first to greedily match
        sorted_codes = sorted(PIPE_MAP.keys(), key=len, reverse=True)
        for code in sorted_codes:
            suffix = f"_{code}"
            if remainder.endswith(suffix):
                subject_str = remainder[: -len(suffix)]
                if subject_str.isdigit():
                    return subject_str, code
        return None, None
