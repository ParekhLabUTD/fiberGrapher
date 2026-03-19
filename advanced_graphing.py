"""
Advanced Graphing Module for Fiber Photometry Analysis
Pure post-hoc visualization layer - operates only on existing batch processing outputs.

All functions are pure (no Streamlit). The Streamlit Tab 5 UI calls these functions.
"""

import os
import re
from datetime import datetime as dt
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(s: str) -> str:
    """Sanitize string for use in filenames."""
    return re.sub(r'[^\w\-_. ]', '_', str(s))


def _timestamp_suffix() -> str:
    """Generate timestamp suffix for output files."""
    return dt.now().strftime("%Y%m%d_%H%M%S")


def _unique_path(path: str) -> str:
    """If *path* already exists, insert a timestamp before the extension."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{_timestamp_suffix()}{ext}"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_sessions(events_folder: str) -> List[str]:
    """
    Return sorted list of session prism CSV basenames inside
    ``<events_folder>/prism_tables/sessions/``.
    """
    prism_dir = os.path.join(events_folder, "prism_tables", "sessions")
    if not os.path.isdir(prism_dir):
        return []
    return sorted(f for f in os.listdir(prism_dir) if f.endswith("_prism.csv"))


def discover_sessions_recursive(
    folder: str,
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Recursively scan *folder* for ``*_prism.csv`` files and their
    matching ``*_peri_event_traces_LONG.csv`` files.

    Returns dict keyed by prism CSV basename::

        {basename: {"prism_path": str, "long_csv_path": str | None}}
    """
    prism_files: Dict[str, Dict[str, Optional[str]]] = {}
    long_files: List[str] = []

    for root, _dirs, files in os.walk(folder):
        for f in files:
            full_path = os.path.join(root, f)
            if f.endswith("_prism.csv"):
                prism_files[f] = {"prism_path": full_path, "long_csv_path": None}
            elif f.endswith("_peri_event_traces_LONG.csv"):
                long_files.append(full_path)

    # Match LONG CSVs to prism files by session key
    # Prism names: {mouse}_{session}_prism.csv  -> key = {mouse}_{session}
    # LONG  names: {session}_peri_event_traces_LONG.csv
    # The prism key CONTAINS the LONG key, so check long_key in prism_key
    for basename, info in prism_files.items():
        prism_key = basename.replace("_prism.csv", "")
        for long_path in long_files:
            long_key = os.path.basename(long_path).replace(
                "_peri_event_traces_LONG.csv", ""
            )
            if long_key and long_key in prism_key:
                info["long_csv_path"] = long_path
                break

    return prism_files


# ---------------------------------------------------------------------------
# Event-folder-aware discovery (for redesigned Tab 5)
# ---------------------------------------------------------------------------

def discover_event_folders(base_path: str) -> List[str]:
    """
    Return sorted list of batch-processing output folder names inside
    ``<base_path>/plots/``.
    """
    plots_dir = os.path.join(base_path, "plots")
    if not os.path.isdir(plots_dir):
        return []
    return sorted(
        f for f in os.listdir(plots_dir)
        if os.path.isdir(os.path.join(plots_dir, f))
    )


def discover_prism_sessions(event_folder: str) -> Dict[str, str]:
    """
    List session-level prism CSVs in ``<event_folder>/prism_tables/sessions/``.

    Returns ``{display_label: absolute_path}`` sorted by label.
    """
    prism_dir = os.path.join(event_folder, "prism_tables", "sessions")
    if not os.path.isdir(prism_dir):
        return {}
    result: Dict[str, str] = {}
    for f in sorted(os.listdir(prism_dir)):
        if f.endswith("_prism.csv"):
            result[f] = os.path.join(prism_dir, f)
    return result


def discover_long_trace_sessions(event_folder: str) -> Dict[str, str]:
    """
    List trial-level LONG CSVs in ``<event_folder>/mice/<mouseID>/``.

    Returns ``{display_label: absolute_path}`` sorted by label.
    """
    mice_dir = os.path.join(event_folder, "mice")
    if not os.path.isdir(mice_dir):
        return {}
    result: Dict[str, str] = {}
    for root, _dirs, files in os.walk(mice_dir):
        for f in sorted(files):
            if f.endswith("_peri_event_traces_LONG.csv"):
                result[f] = os.path.join(root, f)
    return dict(sorted(result.items()))


def run_advanced_graphing_from_discovered(
    discovered_sessions: Dict[str, Dict[str, Optional[str]]],
    selected_basenames: List[str],
    output_folder: str,
    signal_type: str,
    baseline_window: Tuple[float, float],
    response_window: Tuple[float, float],
    plot_types: List[str],
) -> Dict:
    """
    Run advanced graphing using pre-discovered session paths.

    *discovered_sessions* is the dict returned by
    :func:`discover_sessions_recursive`.
    """
    os.makedirs(output_folder, exist_ok=True)

    all_results: List[Dict] = []
    errors: List[Dict] = []

    for basename in selected_basenames:
        try:
            info = discovered_sessions[basename]
            prism_path = info["prism_path"]
            long_csv = info["long_csv_path"]

            session_key = _safe_name(basename.replace("_prism.csv", ""))
            session_output = os.path.join(output_folder, session_key)

            result = process_session(
                session_prism_path=prism_path,
                session_long_csv_path=long_csv,
                baseline_window=baseline_window,
                response_window=response_window,
                signal_type=signal_type,
                plot_types=plot_types,
                output_folder=session_output,
            )
            all_results.append(result)
        except Exception as e:
            errors.append({"session": basename, "error": str(e)})

    # Aggregated export
    aggregated_csv_path = None
    if all_results:
        agg_path = os.path.join(output_folder, "aggregated_session_stats.csv")
        aggregated_csv_path = generate_aggregated_export(all_results, agg_path)

    return {
        "output_folder": output_folder,
        "aggregated_csv": aggregated_csv_path,
        "n_sessions_processed": len(all_results),
        "n_errors": len(errors),
        "errors": errors,
        "signal_type": signal_type,
        "baseline_window": baseline_window,
        "response_window": response_window,
        "plot_types": plot_types,
        "session_results": all_results,
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_session_prism_data(prism_path: str) -> pd.DataFrame:
    """
    Load session-level mean trace from prism CSV.

    Expected columns: time_s, mean, [sem]
    """
    if not os.path.exists(prism_path):
        raise FileNotFoundError(f"Prism file not found: {prism_path}")

    df = pd.read_csv(prism_path)
    required = ["time_s", "mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Prism CSV missing columns {missing}. Got {df.columns.tolist()}"
        )
    return df


def load_trial_traces(long_csv_path: str) -> pd.DataFrame:
    """
    Load trial-level peri-event traces from LONG CSV.

    Expected columns: trial_index, time_idx, time_s, signal_z,
                      session, mouse, group
    """
    if not os.path.exists(long_csv_path):
        raise FileNotFoundError(f"Trial traces file not found: {long_csv_path}")

    df = pd.read_csv(long_csv_path)
    required = ["trial_index", "time_idx", "time_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Trial CSV missing columns {missing}. Got {df.columns.tolist()}"
        )
    return df


def find_long_csv_for_session(
    events_folder: str, session_prism_basename: str
) -> Optional[str]:
    """
    Locate the ``*_peri_event_traces_LONG.csv`` that corresponds to a prism
    session file.  Search inside ``<events_folder>/mice/``.

    The prism filename has the form ``{mouse}_{session}_prism.csv``.
    The LONG CSV lives under ``mice/{mouse}/{session}_peri_event_traces_LONG.csv``.
    """
    # Prism name: {mouse}_{session}_prism.csv  -> prism_key = {mouse}_{session}
    # LONG  name: {session}_peri_event_traces_LONG.csv
    # The prism key CONTAINS the session basename, so we extract the LONG
    # key and check if it is contained in the prism key.
    prism_key = session_prism_basename.replace("_prism.csv", "")
    mice_folder = os.path.join(events_folder, "mice")
    if not os.path.isdir(mice_folder):
        return None

    for root, _dirs, files in os.walk(mice_folder):
        for f in files:
            if f.endswith("_peri_event_traces_LONG.csv"):
                long_key = f.replace("_peri_event_traces_LONG.csv", "")
                if long_key and long_key in prism_key:
                    return os.path.join(root, f)
    return None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def extract_session_metadata(
    session_prism_basename: str,
    long_csv_path: Optional[str] = None,
) -> Dict:
    """
    Extract mouse, group, session identifiers.

    Primary source: prism filename ``{mouse}_{session}_prism.csv``
    Secondary source: columns in the LONG CSV (mouse, group, session).
    """
    name = session_prism_basename.replace("_prism.csv", "")
    parts = name.split("_", 1)
    mouse_id = parts[0] if len(parts) >= 2 else "unknown"
    session_name = parts[1] if len(parts) >= 2 else name
    group_name = "unknown"

    # Try to get group from LONG CSV (much more reliable)
    if long_csv_path and os.path.exists(long_csv_path):
        try:
            df = pd.read_csv(long_csv_path, nrows=1)
            if "group" in df.columns:
                group_name = str(df["group"].iloc[0])
            if "mouse" in df.columns:
                mouse_id = str(df["mouse"].iloc[0])
            if "session" in df.columns:
                session_name = str(df["session"].iloc[0])
        except Exception:
            pass

    return {
        "mouse": mouse_id,
        "session": session_name,
        "group": group_name,
    }


# ---------------------------------------------------------------------------
# Per-trial computations
# ---------------------------------------------------------------------------

def _resolve_signal_column(df: pd.DataFrame, signal_type: str) -> str:
    """Return the column name in *df* that holds the requested signal type."""
    if signal_type == "z-score":
        if "signal_z" in df.columns:
            return "signal_z"
        raise ValueError("Column 'signal_z' not found in trial CSV")

    # raw ΔF/F — check common alternatives
    for col in ("dff", "signal_raw", "raw_dff", "signal_z"):
        if col in df.columns:
            return col
    raise ValueError(
        f"No suitable raw signal column found. Columns: {df.columns.tolist()}"
    )


def compute_trial_window_means(
    trial_df: pd.DataFrame,
    window_start: float,
    window_end: float,
    signal_column: str,
) -> np.ndarray:
    """
    For each trial, compute the mean of *signal_column* within
    [window_start, window_end].

    Returns 1-D array of length n_trials.
    """
    mask = (trial_df["time_s"] >= window_start) & (trial_df["time_s"] <= window_end)
    sub = trial_df.loc[mask]
    if sub.empty:
        raise ValueError(
            f"No data points in window [{window_start}, {window_end}]"
        )
    return sub.groupby("trial_index")[signal_column].mean().values


def compute_trial_window_aucs(
    trial_df: pd.DataFrame,
    window_start: float,
    window_end: float,
    signal_column: str,
) -> np.ndarray:
    """
    For each trial, compute trapezoidal AUC of *signal_column* within
    [window_start, window_end].

    Returns 1-D array of length n_trials.
    """
    mask = (trial_df["time_s"] >= window_start) & (trial_df["time_s"] <= window_end)
    sub = trial_df.loc[mask]
    if sub.empty:
        raise ValueError(
            f"No data points in window [{window_start}, {window_end}]"
        )

    aucs = []
    for _trial, grp in sub.groupby("trial_index"):
        grp_sorted = grp.sort_values("time_s")
        auc = float(np.trapz(grp_sorted[signal_column].values,
                              grp_sorted["time_s"].values))
        aucs.append(auc)
    return np.array(aucs)


# ---------------------------------------------------------------------------
# Session-level statistics (wraps per-trial computations)
# ---------------------------------------------------------------------------

def compute_window_stats(
    trial_df: pd.DataFrame,
    window_start: float,
    window_end: float,
    signal_column: str,
) -> Dict:
    """
    Compute mean, stddev, n for signal mean in a window using trial-level data.
    """
    trial_means = compute_trial_window_means(
        trial_df, window_start, window_end, signal_column
    )
    return {
        "mean": float(np.mean(trial_means)),
        "stddev": float(np.std(trial_means, ddof=0)),
        "n": len(trial_means),
    }


def compute_auc_stats(
    trial_df: pd.DataFrame,
    window_start: float,
    window_end: float,
    signal_column: str,
) -> Dict:
    """
    Compute AUC mean, stddev, n for a window using trial-level data.
    """
    trial_aucs = compute_trial_window_aucs(
        trial_df, window_start, window_end, signal_column
    )
    return {
        "auc_mean": float(np.mean(trial_aucs)),
        "auc_stddev": float(np.std(trial_aucs, ddof=0)),
        "n": len(trial_aucs),
    }


# ---------------------------------------------------------------------------
# Fallback: session-mean-only stats (when LONG CSV unavailable)
# ---------------------------------------------------------------------------

def compute_window_stats_from_mean(
    time_s: np.ndarray,
    mean_trace: np.ndarray,
    window_start: float,
    window_end: float,
) -> Dict:
    """Fallback stats computed only from the session mean trace (no trials)."""
    mask = (time_s >= window_start) & (time_s <= window_end)
    if not mask.any():
        raise ValueError(
            f"No data points in window [{window_start}, {window_end}]"
        )
    vals = mean_trace[mask]
    return {
        "mean": float(np.mean(vals)),
        "stddev": float(np.std(vals, ddof=0)),
        "n": 1,
    }


def compute_auc_stats_from_mean(
    time_s: np.ndarray,
    mean_trace: np.ndarray,
    window_start: float,
    window_end: float,
) -> Dict:
    """Fallback AUC computed only from the session mean trace (no trials)."""
    mask = (time_s >= window_start) & (time_s <= window_end)
    if not mask.any():
        raise ValueError(
            f"No data points in window [{window_start}, {window_end}]"
        )
    t = time_s[mask]
    v = mean_trace[mask]
    auc = float(np.trapz(v, t))
    return {
        "auc_mean": auc,
        "auc_stddev": 0.0,
        "n": 1,
    }


# ---------------------------------------------------------------------------
# Plotting (pure functions — no Streamlit)
# ---------------------------------------------------------------------------

def plot_signal_mean_bars(
    session_id: str,
    baseline_stats: Dict,
    response_stats: Dict,
    output_path: str,
    metadata: Dict,
    signal_type: str,
    baseline_window: Tuple[float, float],
    response_window: Tuple[float, float],
) -> str:
    """
    Create bar plot comparing baseline vs response signal means.

    Returns path to saved figure.
    """
    output_path = _unique_path(output_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = [0, 1]
    means = [baseline_stats["mean"], response_stats["mean"]]
    stds = [baseline_stats["stddev"], response_stats["stddev"]]
    labels = ["Baseline", "Response"]

    ax.bar(
        x_pos, means, yerr=stds, capsize=10, alpha=0.7,
        color=["steelblue", "coral"], edgecolor="black", linewidth=1.5,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(f"Mean Signal ({signal_type})", fontsize=12)
    ax.set_title(
        f"Signal Mean: {metadata['mouse']} | {metadata['session']}\n"
        f"Group: {metadata['group']} | "
        f"Baseline [{baseline_window[0]}, {baseline_window[1]}]s | "
        f"Response [{response_window[0]}, {response_window[1]}]s\n"
        f"n = {baseline_stats['n']} trials",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_auc_bars(
    session_id: str,
    baseline_auc: Dict,
    response_auc: Dict,
    output_path: str,
    metadata: Dict,
    signal_type: str,
    baseline_window: Tuple[float, float],
    response_window: Tuple[float, float],
) -> str:
    """
    Create bar plot comparing baseline vs response AUC.

    Returns path to saved figure.
    """
    output_path = _unique_path(output_path)

    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = [0, 1]
    means = [baseline_auc["auc_mean"], response_auc["auc_mean"]]
    stds = [baseline_auc["auc_stddev"], response_auc["auc_stddev"]]
    labels = ["Baseline", "Response"]

    ax.bar(
        x_pos, means, yerr=stds, capsize=10, alpha=0.7,
        color=["steelblue", "coral"], edgecolor="black", linewidth=1.5,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(f"AUC ({signal_type})", fontsize=12)
    ax.set_title(
        f"AUC: {metadata['mouse']} | {metadata['session']}\n"
        f"Group: {metadata['group']} | "
        f"Baseline [{baseline_window[0]}, {baseline_window[1]}]s | "
        f"Response [{response_window[0]}, {response_window[1]}]s\n"
        f"n = {baseline_auc['n']} trials",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_heatmap(
    trial_df: pd.DataFrame,
    output_path: str,
    metadata: Dict,
    signal_type: str,
    signal_column: str = "signal_z",
) -> str:
    """
    Create trial-level heatmap with MATLAB-style contrast scaling.

    Rows = trials (ordered by trial_index).
    Columns = timepoints (time_idx).
    Values = signal_column.
    Symmetric scaling: clim = percentile(abs(signal), 98).

    Returns path to saved figure.
    """
    output_path = _unique_path(output_path)

    # Pivot to create matrix: trials x timepoints
    matrix = trial_df.pivot(
        index="trial_index", columns="time_idx", values=signal_column
    )

    # Get time vector for x-axis
    time_mapping = (
        trial_df[["time_idx", "time_s"]]
        .drop_duplicates()
        .sort_values("time_idx")
    )
    time_vector = time_mapping["time_s"].values

    # Compute symmetric color limits (98th percentile of absolute values)
    flat = matrix.values.ravel()
    flat_clean = flat[~np.isnan(flat)]
    if len(flat_clean) == 0:
        clim = 1.0
    else:
        clim = float(np.percentile(np.abs(flat_clean), 98))
    if clim == 0:
        clim = 1.0

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-clim,
        vmax=clim,
        interpolation="nearest",
    )

    # Vertical line at t=0
    zero_idx = int(np.argmin(np.abs(time_vector)))
    ax.axvline(zero_idx, color="black", linestyle="--", linewidth=2, alpha=0.7)

    # Axis labels
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Trial Index", fontsize=12)

    # X-axis ticks
    n_ticks = min(10, len(time_vector))
    tick_indices = np.linspace(0, len(time_vector) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f"{time_vector[i]:.1f}" for i in tick_indices])

    # Y-axis ticks
    trial_indices = matrix.index.values
    n_y = min(10, len(trial_indices))
    y_tick_indices = np.linspace(0, len(trial_indices) - 1, n_y, dtype=int)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels([str(trial_indices[i]) for i in y_tick_indices])

    # Colorbar
    fig.colorbar(im, ax=ax, label=f"Signal ({signal_type})")

    # Title
    ax.set_title(
        f"Heatmap: {metadata['mouse']} | {metadata['session']}\n"
        f"Group: {metadata['group']} | "
        f"n_trials={len(trial_indices)} | clim=±{clim:.2f}",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

def save_session_stats_csv(
    session_id: str,
    baseline_signal: Dict,
    response_signal: Dict,
    baseline_auc: Optional[Dict],
    response_auc: Optional[Dict],
    output_path: str,
    metadata: Dict,
    signal_type: str,
    baseline_window: Tuple[float, float],
    response_window: Tuple[float, float],
) -> str:
    """
    Save per-session statistics to Prism-ready CSV.

    Returns path to saved CSV.
    """
    output_path = _unique_path(output_path)

    rows = [
        {
            "window": "baseline",
            "signal_mean": baseline_signal["mean"],
            "signal_stddev": baseline_signal["stddev"],
            "signal_n": baseline_signal["n"],
        },
        {
            "window": "response",
            "signal_mean": response_signal["mean"],
            "signal_stddev": response_signal["stddev"],
            "signal_n": response_signal["n"],
        },
    ]

    if baseline_auc is not None and response_auc is not None:
        rows[0]["auc_mean"] = baseline_auc["auc_mean"]
        rows[0]["auc_stddev"] = baseline_auc["auc_stddev"]
        rows[0]["auc_n"] = baseline_auc["n"]
        rows[1]["auc_mean"] = response_auc["auc_mean"]
        rows[1]["auc_stddev"] = response_auc["auc_stddev"]
        rows[1]["auc_n"] = response_auc["n"]

    for r in rows:
        r["mouse"] = metadata["mouse"]
        r["session"] = metadata["session"]
        r["group"] = metadata["group"]
        r["signal_type"] = signal_type
        r["baseline_window"] = f"[{baseline_window[0]}, {baseline_window[1]}]"
        r["response_window"] = f"[{response_window[0]}, {response_window[1]}]"

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path


def generate_aggregated_export(
    all_session_stats: List[Dict], output_path: str
) -> str:
    """
    Generate aggregated CSV with one row per session plus summary statistics.

    For Prism comparison plots (not graphed in Tab 5).

    Returns path to saved CSV.
    """
    output_path = _unique_path(output_path)

    rows = []
    for stats in all_session_stats:
        md = stats["metadata"]
        row = {
            "mouse": md["mouse"],
            "session": md["session"],
            "group": md["group"],
            "baseline_signal_mean": stats["baseline_signal"]["mean"],
            "baseline_signal_stddev": stats["baseline_signal"]["stddev"],
            "baseline_signal_n": stats["baseline_signal"]["n"],
            "response_signal_mean": stats["response_signal"]["mean"],
            "response_signal_stddev": stats["response_signal"]["stddev"],
            "response_signal_n": stats["response_signal"]["n"],
            "baseline_auc_mean": stats["baseline_auc"]["auc_mean"],
            "baseline_auc_stddev": stats["baseline_auc"]["auc_stddev"],
            "baseline_auc_n": stats["baseline_auc"]["n"],
            "response_auc_mean": stats["response_auc"]["auc_mean"],
            "response_auc_stddev": stats["response_auc"]["auc_stddev"],
            "response_auc_n": stats["response_auc"]["n"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Summary rows: mean and stddev across sessions
    numeric_cols = [c for c in df.columns if c not in ("mouse", "session", "group")]

    summary_mean = {"mouse": "MEAN_ACROSS_SESSIONS", "session": "", "group": ""}
    summary_std = {"mouse": "STDDEV_ACROSS_SESSIONS", "session": "", "group": ""}
    for col in numeric_cols:
        summary_mean[col] = df[col].mean()
        summary_std[col] = df[col].std()

    df = pd.concat(
        [df, pd.DataFrame([summary_mean, summary_std])], ignore_index=True
    )
    df.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# Session processor
# ---------------------------------------------------------------------------

def process_session(
    session_prism_path: str,
    session_long_csv_path: Optional[str],
    baseline_window: Tuple[float, float],
    response_window: Tuple[float, float],
    signal_type: str,
    plot_types: List[str],
    output_folder: str,
) -> Dict:
    """
    Process a single session and generate requested plots.

    Args:
        session_prism_path: Path to session prism CSV
        session_long_csv_path: Path to trial traces LONG CSV (needed for
            trial-level stats and heatmaps)
        baseline_window: (start, end) in seconds
        response_window: (start, end) in seconds
        signal_type: 'z-score' or 'raw ΔF/F'
        plot_types: List of plot types ('signal_mean', 'auc', 'heatmap')
        output_folder: Where to save outputs

    Returns:
        Dictionary with paths to generated files and computed statistics
    """
    os.makedirs(output_folder, exist_ok=True)

    # --- Load data ---
    prism_df = load_session_prism_data(session_prism_path)
    time_s = prism_df["time_s"].values
    mean_trace = prism_df["mean"].values

    session_basename = os.path.basename(session_prism_path)
    metadata = extract_session_metadata(session_basename, session_long_csv_path)

    # --- Load trial-level data if available ---
    trial_df: Optional[pd.DataFrame] = None
    signal_col: Optional[str] = None
    has_trials = False

    if session_long_csv_path and os.path.exists(session_long_csv_path):
        trial_df = load_trial_traces(session_long_csv_path)
        signal_col = _resolve_signal_column(trial_df, signal_type)
        has_trials = True

        n_trials = int(trial_df["trial_index"].nunique())
        if n_trials <= 3:
            # This is a warning condition — still proceed
            pass
    else:
        n_trials = 1  # Fallback

    # --- Validate windows against data ---
    if time_s.min() > baseline_window[0] or time_s.max() < baseline_window[1]:
        raise ValueError(
            f"Baseline window [{baseline_window[0]}, {baseline_window[1]}]s "
            f"exceeds data range [{time_s.min():.2f}, {time_s.max():.2f}]s "
            f"for session {session_basename}"
        )
    if time_s.min() > response_window[0] or time_s.max() < response_window[1]:
        raise ValueError(
            f"Response window [{response_window[0]}, {response_window[1]}]s "
            f"exceeds data range [{time_s.min():.2f}, {time_s.max():.2f}]s "
            f"for session {session_basename}"
        )

    # --- Compute statistics ---
    if has_trials and trial_df is not None and signal_col is not None:
        baseline_signal = compute_window_stats(
            trial_df, baseline_window[0], baseline_window[1], signal_col
        )
        response_signal = compute_window_stats(
            trial_df, response_window[0], response_window[1], signal_col
        )
        baseline_auc = compute_auc_stats(
            trial_df, baseline_window[0], baseline_window[1], signal_col
        )
        response_auc = compute_auc_stats(
            trial_df, response_window[0], response_window[1], signal_col
        )
    else:
        # Fallback: use session mean trace only
        baseline_signal = compute_window_stats_from_mean(
            time_s, mean_trace, baseline_window[0], baseline_window[1]
        )
        response_signal = compute_window_stats_from_mean(
            time_s, mean_trace, response_window[0], response_window[1]
        )
        baseline_auc = compute_auc_stats_from_mean(
            time_s, mean_trace, baseline_window[0], baseline_window[1]
        )
        response_auc = compute_auc_stats_from_mean(
            time_s, mean_trace, response_window[0], response_window[1]
        )

    results = {
        "metadata": metadata,
        "baseline_signal": baseline_signal,
        "response_signal": response_signal,
        "baseline_auc": baseline_auc,
        "response_auc": response_auc,
        "n_trials": n_trials,
        "output_files": [],
        "warnings": [],
    }

    if n_trials <= 3 and n_trials > 0:
        results["warnings"].append(
            f"Session has only {n_trials} trials — statistics may be unreliable"
        )

    # --- Build filenames ---
    safe_session = _safe_name(metadata["session"])
    safe_mouse = _safe_name(metadata["mouse"])
    safe_group = _safe_name(metadata["group"])
    sig_tag = "zscore" if signal_type == "z-score" else "rawdFF"
    bl_tag = f"bl{baseline_window[0]}to{baseline_window[1]}"
    rsp_tag = f"rsp{response_window[0]}to{response_window[1]}"
    base_name = f"{safe_mouse}_{safe_group}_{safe_session}_{sig_tag}_{bl_tag}_{rsp_tag}"

    # --- Generate requested plots ---
    if "signal_mean" in plot_types:
        png = os.path.join(output_folder, f"{base_name}_signal_mean_bars.png")
        plot_signal_mean_bars(
            session_basename, baseline_signal, response_signal,
            png, metadata, signal_type, baseline_window, response_window,
        )
        results["output_files"].append(png)

    if "auc" in plot_types:
        png = os.path.join(output_folder, f"{base_name}_auc_bars.png")
        plot_auc_bars(
            session_basename, baseline_auc, response_auc,
            png, metadata, signal_type, baseline_window, response_window,
        )
        results["output_files"].append(png)

    if "heatmap" in plot_types:
        if trial_df is None or signal_col is None:
            raise FileNotFoundError(
                f"Heatmap requires trial traces CSV, but not found for "
                f"session {session_basename}"
            )
        png = os.path.join(output_folder, f"{base_name}_heatmap.png")
        plot_heatmap(trial_df, png, metadata, signal_type, signal_col)
        results["output_files"].append(png)

    # --- Save session stats CSV ---
    csv_path = os.path.join(output_folder, f"{base_name}_stats.csv")
    save_session_stats_csv(
        session_basename, baseline_signal, response_signal,
        baseline_auc, response_auc,
        csv_path, metadata, signal_type, baseline_window, response_window,
    )
    results["output_files"].append(csv_path)

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_advanced_graphing(
    events_folder: str,
    selected_sessions: List[str],
    signal_type: str,
    baseline_window: Tuple[float, float],
    response_window: Tuple[float, float],
    plot_types: List[str],
) -> Dict:
    """
    Main entry point for advanced graphing.

    Args:
        events_folder: Path to batch processing output folder
            (e.g. ``<base>/plots/<event_timestamp>/``)
        selected_sessions: List of session prism CSV basenames
        signal_type: 'z-score' or 'raw ΔF/F'
        baseline_window: (start, end) in seconds
        response_window: (start, end) in seconds
        plot_types: List of plot types to generate

    Returns:
        Summary dictionary with paths and statistics
    """
    prism_sessions_folder = os.path.join(
        events_folder, "prism_tables", "sessions"
    )
    if not os.path.exists(prism_sessions_folder):
        raise FileNotFoundError(
            f"Prism sessions folder not found: {prism_sessions_folder}"
        )

    # --- Create output folder ---
    output_root = os.path.join(events_folder, "advanced_graphs")
    os.makedirs(output_root, exist_ok=True)

    all_results: List[Dict] = []
    errors: List[Dict] = []

    for session_basename in selected_sessions:
        try:
            session_prism_path = os.path.join(
                prism_sessions_folder, session_basename
            )
            if not os.path.exists(session_prism_path):
                raise FileNotFoundError(
                    f"Session prism file not found: {session_prism_path}"
                )

            # Locate matching LONG CSV
            long_csv = find_long_csv_for_session(events_folder, session_basename)

            # Build session output subfolder
            session_key = _safe_name(
                session_basename.replace("_prism.csv", "")
            )
            session_output = os.path.join(output_root, session_key)

            result = process_session(
                session_prism_path=session_prism_path,
                session_long_csv_path=long_csv,
                baseline_window=baseline_window,
                response_window=response_window,
                signal_type=signal_type,
                plot_types=plot_types,
                output_folder=session_output,
            )
            all_results.append(result)

        except Exception as e:
            errors.append({"session": session_basename, "error": str(e)})

    # --- Aggregated export ---
    aggregated_csv_path = None
    if all_results:
        agg_path = os.path.join(output_root, "aggregated_session_stats.csv")
        aggregated_csv_path = generate_aggregated_export(all_results, agg_path)

    summary = {
        "output_folder": output_root,
        "aggregated_csv": aggregated_csv_path,
        "n_sessions_processed": len(all_results),
        "n_errors": len(errors),
        "errors": errors,
        "signal_type": signal_type,
        "baseline_window": baseline_window,
        "response_window": response_window,
        "plot_types": plot_types,
        "session_results": all_results,
    }

    return summary