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

        # Newer shared-gate experiment exports do not store n_session in the
        # per-pipeline summary pickle.  The dashboard still needs it for
        # utilization denominators, so infer it from the fold count or the
        # shared CI-gate files when possible.
        if "n_session" not in result.columns:
            result["n_session"] = np.nan
        result["n_session"] = pd.to_numeric(result["n_session"], errors="coerce")
        missing_n_session = result["n_session"].isna()
        if missing_n_session.any() and "n_valid_pairs" in result.columns:
            inferred = (
                pd.to_numeric(result["n_valid_pairs"], errors="coerce")
                .groupby(result["subject"])
                .transform("max")
            )
            result.loc[missing_n_session, "n_session"] = inferred[missing_n_session]

        missing_n_session = result["n_session"].isna()
        if missing_n_session.any():
            gate_counts = self._shared_gate_session_counts()
            if gate_counts:
                inferred = result["subject"].map(gate_counts)
                result.loc[missing_n_session, "n_session"] = inferred[missing_n_session]
        if result["n_session"].notna().all():
            result["n_session"] = result["n_session"].astype(int)

        # Classify BDP degradation status from the 'score' column
        if "score" in result.columns:
            result["degrade_status"] = result["score"].apply(
                self._classify_degradation
            )
        else:
            result["degrade_status"] = "n/a"

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
                # Extract degradation fields from nested bestSetup
                for rec in records:
                    bs = rec.get("bestSetup")
                    if isinstance(bs, dict):
                        rec["degraded"] = bs.get("degraded", False)
                        rec["score_mode"] = bs.get("score_mode", "unknown")
                        rec["partition_mode"] = bs.get("partition_mode", "unknown")
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

        df = self._load_shared_gate_roles(subject, pipeline_short)
        if not df.empty:
            self._roles_cache[cache_key] = df
            return df

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
                "degradation": pd.DataFrame(),
            }

        return {
            "completion": self._build_completion(sdf),
            "subject_pipeline": self._build_subject_pipeline(sdf),
            "config_agg": self._build_config_agg(sdf),
            "qc_results": self._run_qc(),
            "matched_sets": self._build_matched_sets(),
            "degradation": self._build_degradation_summary(),
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

    # -- Degradation analytics -----------------------------------------------

    def _build_degradation_summary(self) -> pd.DataFrame:
        """Aggregate per-(subject, pipe_short, feature) degradation ratios.

        Only meaningful for BDP pipelines; returns an empty DataFrame
        if no detail data is available.
        """
        ddf = self.detail_df
        if ddf.empty or "degraded" not in ddf.columns:
            return pd.DataFrame()

        bdp_mask = ddf["pipe_short"].isin(["BDP_fb", "BDP_bf"])
        bdp = ddf.loc[bdp_mask].copy()
        if bdp.empty:
            return pd.DataFrame()

        group_cols = ["subject", "pipe_short", "feature"]
        available = [c for c in group_cols if c in bdp.columns]
        if not available:
            return pd.DataFrame()

        rows = []
        for keys, grp in bdp.groupby(available):
            if not isinstance(keys, tuple):
                keys = (keys,)
            info = dict(zip(available, keys))
            total = len(grp)
            n_degraded = grp["degraded"].sum()
            info["total_pairs"] = total
            info["degraded_pairs"] = int(n_degraded)
            info["degraded_ratio"] = n_degraded / total if total else 0.0

            # Accuracy split
            pure = grp.loc[~grp["degraded"], "acc_DA"] if "acc_DA" in grp.columns else pd.Series(dtype=float)
            deg = grp.loc[grp["degraded"], "acc_DA"] if "acc_DA" in grp.columns else pd.Series(dtype=float)
            info["acc_pure"] = pure.mean() if not pure.empty else np.nan
            info["acc_degraded"] = deg.mean() if not deg.empty else np.nan
            rows.append(info)

        return pd.DataFrame(rows)

    @staticmethod
    def _classify_degradation(score_val) -> str:
        """Classify a summary-row 'score' value into a degradation status."""
        s = str(score_val)
        if s == "bridge_proxy":
            return "pure"
        if s.startswith("map_"):
            return "full_degrade"
        if "mixed" in s:
            return "partial"
        return "n/a"

    def _shared_gate_session_counts(self) -> dict[int, int]:
        """Infer per-subject session counts from shared-gate CI files."""
        pattern = os.path.join(
            self.data_dir, f"{self.dataset_name}_*_shared_ci_gates.csv"
        )
        counts: dict[int, int] = {}
        for fpath in glob.glob(pattern):
            basename = os.path.basename(fpath)
            prefix = f"{self.dataset_name}_"
            suffix = "_shared_ci_gates.csv"
            if not (basename.startswith(prefix) and basename.endswith(suffix)):
                continue
            subject_str = basename[len(prefix):-len(suffix)]
            if not subject_str.isdigit():
                continue
            subj = int(subject_str)
            try:
                df = pd.read_csv(fpath, usecols=lambda c: c in {
                    "pair_id", "target_session_abs_idx",
                })
            except Exception as exc:
                logger.warning("Failed to inspect shared CI gates %s: %s", fpath, exc)
                continue
            if "target_session_abs_idx" in df.columns:
                counts[subj] = int(df["target_session_abs_idx"].nunique())
            elif "pair_id" in df.columns:
                counts[subj] = int(df["pair_id"].nunique())
        return counts

    def _load_shared_gate_roles(self, subject, pipeline_short: str) -> pd.DataFrame:
        """Build mechanism roles from shared-gate CI files when roles CSVs are absent.

        The shared-gate experiment stores one CI/partition table per subject
        instead of per-pipeline role CSVs.  This adapter exposes enough of the
        old roles schema for the Session Mechanisms page.
        """
        if pipeline_short not in {"BDP_fb", "BDP_bf", "MMP_mta", "MMP_moe"}:
            return pd.DataFrame()

        fpath = os.path.join(
            self.data_dir, f"{self.dataset_name}_{subject}_shared_ci_gates.csv"
        )
        if not os.path.isfile(fpath):
            return pd.DataFrame()

        try:
            gates = pd.read_csv(fpath)
        except Exception as exc:
            logger.warning("Failed to read shared CI gates %s: %s", fpath, exc)
            return pd.DataFrame()
        if gates.empty:
            return pd.DataFrame()

        required = {
            "subject", "pair_id", "target_session_abs_idx",
            "target_session_label", "train_local_idx",
            "train_session_abs_idx", "train_session_label", "feature",
            "dist_est", "dist_lwr", "dist_upr",
        }
        if not required.issubset(gates.columns):
            missing = sorted(required.difference(gates.columns))
            logger.warning("Shared CI gates missing role columns %s in %s", missing, fpath)
            return pd.DataFrame()

        gates = gates.copy()
        feature_order = {"CSP": 0, "logvar": 8, "TS": 16}
        gates["method_row"] = gates["feature"].map(feature_order)
        if gates["method_row"].isna().any():
            fallback_codes = pd.factorize(gates["feature"], sort=True)[0] * 8
            gates["method_row"] = gates["method_row"].fillna(pd.Series(fallback_codes, index=gates.index))
        gates["method_row"] = gates["method_row"].astype(int)

        pipeline_code = PIPE_FILE_MAP.get(pipeline_short, pipeline_short)
        common = pd.DataFrame({
            "subject": gates["subject"],
            "dataset": self.dataset_name,
            "pipeline": pipeline_code,
            "feature": gates["feature"],
            "classifier": "lda",
            "da": "none",
            "method_row": gates["method_row"],
            "pair_id": gates["pair_id"],
            "target_id": gates["target_session_abs_idx"],
            "target_session_label": gates["target_session_label"],
            "local_idx": gates["train_local_idx"],
            "session_abs_idx": gates["train_session_abs_idx"],
            "session_label": gates["train_session_label"],
            "dist_est": gates["dist_est"],
            "dist_lwr": gates["dist_lwr"],
            "dist_upr": gates["dist_upr"],
            "dist_type": gates.get("dist_type", "mmd"),
            "partition_mode": gates.get("partition_mode", np.nan),
            "degraded": gates.get("degraded", np.nan),
            "is_best": gates.get("is_best_nearest", False),
        })
        common["dist_to_session"] = common["target_id"]
        common["dist_est_method"] = "shared_gate"
        common["weight"] = np.nan

        if pipeline_short.startswith("BDP"):
            selection = common.copy()
            selection["stage"] = "selection"
            selection["role"] = np.where(
                gates.get("is_near_bridge", False), "bridge", "far"
            )
            selection["score_mode"] = "shared_gate_bridge_proxy"

            final = common.copy()
            final["stage"] = "final"
            final["role"] = np.where(
                gates.get("is_final_source", False), "train", "not_used"
            )
            final["score_mode"] = "shared_gate_final"
            roles = pd.concat([selection, final], ignore_index=True)
        else:
            main = common.copy()
            main["stage"] = "main"
            main["role"] = np.select(
                [
                    gates.get("is_best_nearest", False),
                    gates.get("is_near_bridge", False),
                ],
                ["nearest", "near"],
                default="far",
            )
            main["score_mode"] = "shared_gate_near_set"

            final = common.copy()
            final["stage"] = "final"
            final["role"] = np.where(
                gates.get("is_final_source", False), "train", "not_used"
            )
            final["score_mode"] = "shared_gate_final"
            roles = pd.concat([main, final], ignore_index=True)

        target_rows = self._shared_gate_target_rows(gates, pipeline_code, pipeline_short)
        if not target_rows.empty:
            roles = pd.concat([roles, target_rows], ignore_index=True)

        return roles

    def _shared_gate_target_rows(
        self, gates: pd.DataFrame, pipeline_code: str, pipeline_short: str
    ) -> pd.DataFrame:
        """Create target rows matching the legacy roles schema."""
        target_cols = [
            "subject", "pair_id", "target_session_abs_idx",
            "target_session_label", "feature", "method_row",
        ]
        targets = gates[target_cols].drop_duplicates().copy()
        if targets.empty:
            return pd.DataFrame()

        stages = ["selection", "final"] if pipeline_short.startswith("BDP") else ["main", "final"]
        rows = []
        for stage in stages:
            df = pd.DataFrame({
                "subject": targets["subject"],
                "dataset": self.dataset_name,
                "pipeline": pipeline_code,
                "feature": targets["feature"],
                "classifier": "lda",
                "da": "none",
                "method_row": targets["method_row"],
                "pair_id": targets["pair_id"],
                "target_id": targets["target_session_abs_idx"],
                "target_session_label": targets["target_session_label"],
                "local_idx": np.nan,
                "session_abs_idx": targets["target_session_abs_idx"],
                "session_label": targets["target_session_label"],
                "role": "target",
                "stage": stage,
                "is_best": False,
                "weight": np.nan,
                "dist_to_session": np.nan,
                "dist_est_method": "shared_gate",
                "dist_est": np.nan,
                "dist_lwr": np.nan,
                "dist_upr": np.nan,
                "dist_type": "mmd",
                "partition_mode": np.nan,
                "degraded": np.nan,
                "score_mode": "shared_gate",
            })
            rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

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
