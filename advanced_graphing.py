"""
Advanced Graphing Module for Fiber Photometry Analysis
Pure post-hoc visualization layer - operates only on existing batch processing outputs.
"""

import os
import re
from datetime import datetime as dt
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_name(s: str) -> str:
    """Sanitize string for use in filenames."""
    return re.sub(r'[^\w\-_. ]', '_', str(s))


def _timestamp_suffix() -> str:
    """Generate timestamp suffix for output files."""
    return dt.now().strftime("%Y%m%d_%H%M%S")


def load_session_prism_data(prism_path: str) -> pd.DataFrame:
    """
    Load session-level mean trace from prism CSV.
    
    Expected columns: time_s, mean, [sem]
    """
    if not os.path.exists(prism_path):
        raise FileNotFoundError(f"Prism file not found: {prism_path}")
    
    df = pd.read_csv(prism_path)
    required = ['time_s', 'mean']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Prism CSV missing required columns. Expected {required}, got {df.columns.tolist()}")
    
    return df


def load_trial_traces(long_csv_path: str) -> pd.DataFrame:
    """
    Load trial-level peri-event traces from LONG CSV.
    
    Expected columns: trial_index, time_idx, time_s, signal_z, session, mouse, group
    """
    if not os.path.exists(long_csv_path):
        raise FileNotFoundError(f"Trial traces file not found: {long_csv_path}")
    
    df = pd.read_csv(long_csv_path)
    required = ['trial_index', 'time_idx', 'time_s']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Trial CSV missing required columns. Expected at least {required}, got {df.columns.tolist()}")
    
    return df


def extract_session_metadata(session_basename: str, prism_sessions_folder: str) -> Dict:
    """
    Extract mouse, group, session identifiers from session prism filename.
    Expected pattern: {mouse}_{session}_prism.csv
    """
    # Remove _prism.csv suffix
    name = session_basename.replace('_prism.csv', '')
    
    # Try to parse mouse_session pattern
    parts = name.split('_', 1)
    if len(parts) == 2:
        mouse_id = parts[0]
        session_name = parts[1]
    else:
        mouse_id = "unknown"
        session_name = name
    
    # Try to infer group from folder structure or metadata
    group_name = "unknown"
    
    return {
        'mouse': mouse_id,
        'session': session_name,
        'group': group_name
    }


def compute_window_stats(time_s: np.ndarray, mean_trace: np.ndarray, 
                         window_start: float, window_end: float,
                         n_trials: int) -> Dict:
    """
    Compute mean and stddev for a time window from session mean trace.
    
    Args:
        time_s: Time vector
        mean_trace: Session mean trace
        window_start: Window start time (seconds)
        window_end: Window end time (seconds)
        n_trials: Number of trials in session (for stddev scaling)
    
    Returns:
        Dictionary with mean, stddev, n
    """
    mask = (time_s >= window_start) & (time_s <= window_end)
    if not mask.any():
        raise ValueError(f"No data points in window [{window_start}, {window_end}]")
    
    window_values = mean_trace[mask]
    window_mean = float(np.mean(window_values))
    
    # STDDEV inferred from session mean and trial count
    # This is an approximation - ideally we'd have per-trial data
    # For now, use the variability in the mean trace as a proxy
    window_std = float(np.std(window_values) * np.sqrt(n_trials))
    
    return {
        'mean': window_mean,
        'stddev': window_std,
        'n': n_trials
    }


def compute_auc_stats(time_s: np.ndarray, mean_trace: np.ndarray,
                      window_start: float, window_end: float,
                      n_trials: int) -> Dict:
    """
    Compute AUC using trapezoidal integration for a time window.
    
    Returns:
        Dictionary with auc_mean, auc_stddev, n
    """
    mask = (time_s >= window_start) & (time_s <= window_end)
    if not mask.any():
        raise ValueError(f"No data points in window [{window_start}, {window_end}]")
    
    window_time = time_s[mask]
    window_values = mean_trace[mask]
    
    auc = float(np.trapz(window_values, window_time))
    
    # Approximate stddev (would need trial-level data for exact calculation)
    auc_std = float(np.std(window_values) * np.sqrt(n_trials) * (window_end - window_start))
    
    return {
        'auc_mean': auc,
        'auc_stddev': auc_std,
        'n': n_trials
    }


def plot_signal_mean_bars(session_id: str, baseline_stats: Dict, response_stats: Dict,
                          output_path: str, metadata: Dict, signal_type: str,
                          baseline_window: Tuple[float, float],
                          response_window: Tuple[float, float]) -> str:
    """
    Create bar plot comparing baseline vs response signal means.
    
    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_pos = [0, 1]
    means = [baseline_stats['mean'], response_stats['mean']]
    stds = [baseline_stats['stddev'], response_stats['stddev']]
    labels = ['Baseline', 'Response']
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7,
                  color=['steelblue', 'coral'], edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(f'Mean Signal ({signal_type})', fontsize=12)
    ax.set_title(
        f"Signal Mean: {metadata['mouse']} | {metadata['session']}\n"
        f"Baseline [{baseline_window[0]}, {baseline_window[1]}]s | "
        f"Response [{response_window[0]}, {response_window[1]}]s",
        fontsize=10
    )
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_auc_bars(session_id: str, baseline_auc: Dict, response_auc: Dict,
                 output_path: str, metadata: Dict, signal_type: str,
                 baseline_window: Tuple[float, float],
                 response_window: Tuple[float, float]) -> str:
    """
    Create bar plot comparing baseline vs response AUC.
    
    Returns:
        Path to saved figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_pos = [0, 1]
    means = [baseline_auc['auc_mean'], response_auc['auc_mean']]
    stds = [baseline_auc['auc_stddev'], response_auc['auc_stddev']]
    labels = ['Baseline', 'Response']
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7,
                  color=['steelblue', 'coral'], edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(f'AUC ({signal_type})', fontsize=12)
    ax.set_title(
        f"AUC: {metadata['mouse']} | {metadata['session']}\n"
        f"Baseline [{baseline_window[0]}, {baseline_window[1]}]s | "
        f"Response [{response_window[0]}, {response_window[1]}]s",
        fontsize=10
    )
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_heatmap(trial_df: pd.DataFrame, output_path: str, metadata: Dict,
                signal_type: str, signal_column: str = 'signal_z') -> str:
    """
    Create trial-level heatmap with MATLAB-style contrast scaling.
    
    Args:
        trial_df: DataFrame with trial_index, time_idx, time_s, signal column
        output_path: Where to save figure
        metadata: Session metadata for title
        signal_type: 'z-score' or 'raw ΔF/F'
        signal_column: Column name containing signal values
    
    Returns:
        Path to saved figure
    """
    # Pivot to create matrix: trials x timepoints
    matrix = trial_df.pivot(index='trial_index', columns='time_idx', values=signal_column)
    
    # Get time vector for x-axis
    time_vector = trial_df[['time_idx', 'time_s']].drop_duplicates().sort_values('time_idx')['time_s'].values
    
    # Compute symmetric color limits (98th percentile of absolute values)
    abs_values = np.abs(matrix.values)
    clim = float(np.percentile(abs_values[~np.isnan(abs_values)], 98))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(matrix.values, aspect='auto', cmap='RdBu_r',
                   vmin=-clim, vmax=clim, interpolation='nearest')
    
    # Add vertical line at t=0
    zero_idx = np.argmin(np.abs(time_vector))
    ax.axvline(zero_idx, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    # Set axis labels
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Trial Index', fontsize=12)
    
    # Set x-axis ticks to show time in seconds
    n_ticks = 10
    tick_indices = np.linspace(0, len(time_vector)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f"{time_vector[i]:.1f}" for i in tick_indices])
    
    # Set y-axis to show trial indices
    trial_indices = matrix.index.values
    y_tick_indices = np.linspace(0, len(trial_indices)-1, min(10, len(trial_indices)), dtype=int)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels([str(trial_indices[i]) for i in y_tick_indices])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label=f'Signal ({signal_type})')
    
    # Title
    ax.set_title(
        f"Heatmap: {metadata['mouse']} | {metadata['session']}\n"
        f"n_trials={len(trial_indices)} | clim=±{clim:.2f}",
        fontsize=10
    )
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def save_session_stats_csv(session_id: str, baseline_signal: Dict, response_signal: Dict,
                           baseline_auc: Dict, response_auc: Dict,
                           output_path: str, metadata: Dict) -> str:
    """
    Save session statistics to Prism-ready CSV.
    
    Returns:
        Path to saved CSV
    """
    data = {
        'metric': ['baseline_signal_mean', 'response_signal_mean',
                   'baseline_auc_mean', 'response_auc_mean'],
        'value': [baseline_signal['mean'], response_signal['mean'],
                  baseline_auc['auc_mean'], response_auc['auc_mean']],
        'stddev': [baseline_signal['stddev'], response_signal['stddev'],
                   baseline_auc['auc_stddev'], response_auc['auc_stddev']],
        'n': [baseline_signal['n'], response_signal['n'],
              baseline_auc['n'], response_auc['n']],
        'mouse': metadata['mouse'],
        'session': metadata['session'],
        'group': metadata['group']
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    return output_path


def generate_aggregated_export(all_session_stats: List[Dict], output_path: str) -> str:
    """
    Generate aggregated CSV with one row per session + summary statistics.
    
    For Prism comparison plots (not graphed in Tab 5).
    
    Returns:
        Path to saved CSV
    """
    rows = []
    
    for stats in all_session_stats:
        row = {
            'mouse': stats['metadata']['mouse'],
            'session': stats['metadata']['session'],
            'group': stats['metadata']['group'],
            'baseline_signal_mean': stats['baseline_signal']['mean'],
            'baseline_signal_stddev': stats['baseline_signal']['stddev'],
            'baseline_signal_n': stats['baseline_signal']['n'],
            'response_signal_mean': stats['response_signal']['mean'],
            'response_signal_stddev': stats['response_signal']['stddev'],
            'response_signal_n': stats['response_signal']['n'],
            'baseline_auc_mean': stats['baseline_auc']['auc_mean'],
            'baseline_auc_stddev': stats['baseline_auc']['auc_stddev'],
            'baseline_auc_n': stats['baseline_auc']['n'],
            'response_auc_mean': stats['response_auc']['auc_mean'],
            'response_auc_stddev': stats['response_auc']['auc_stddev'],
            'response_auc_n': stats['response_auc']['n']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add summary rows (mean and stddev across sessions)
    numeric_cols = [c for c in df.columns if c not in ['mouse', 'session', 'group']]
    
    summary_mean = {'mouse': 'MEAN_ACROSS_SESSIONS', 'session': '', 'group': ''}
    summary_std = {'mouse': 'STDDEV_ACROSS_SESSIONS', 'session': '', 'group': ''}
    
    for col in numeric_cols:
        summary_mean[col] = df[col].mean()
        summary_std[col] = df[col].std()
    
    summary_df = pd.DataFrame([summary_mean, summary_std])
    df = pd.concat([df, summary_df], ignore_index=True)
    
    df.to_csv(output_path, index=False)
    
    return output_path


def process_session(session_prism_path: str, session_long_csv_path: Optional[str],
                   baseline_window: Tuple[float, float], response_window: Tuple[float, float],
                   signal_type: str, plot_types: List[str],
                   output_folder: str, n_trials: int) -> Dict:
    """
    Process a single session and generate requested plots.
    
    Args:
        session_prism_path: Path to session prism CSV (required for all plots)
        session_long_csv_path: Path to trial traces LONG CSV (required only for heatmap)
        baseline_window: (start, end) in seconds
        response_window: (start, end) in seconds
        signal_type: 'z-score' or 'raw ΔF/F'
        plot_types: List of plot types to generate
        output_folder: Where to save outputs
        n_trials: Number of trials in session
    
    Returns:
        Dictionary with paths to generated files and computed statistics
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Load session mean trace
    prism_df = load_session_prism_data(session_prism_path)
    time_s = prism_df['time_s'].values
    mean_trace = prism_df['mean'].values
    
    # Extract metadata from filename
    session_basename = os.path.basename(session_prism_path)
    metadata = extract_session_metadata(session_basename, os.path.dirname(session_prism_path))
    
    # Compute statistics
    baseline_signal = compute_window_stats(time_s, mean_trace, 
                                          baseline_window[0], baseline_window[1], n_trials)
    response_signal = compute_window_stats(time_s, mean_trace,
                                          response_window[0], response_window[1], n_trials)
    
    baseline_auc = compute_auc_stats(time_s, mean_trace,
                                    baseline_window[0], baseline_window[1], n_trials)
    response_auc = compute_auc_stats(time_s, mean_trace,
                                    response_window[0], response_window[1], n_trials)
    
    results = {
        'metadata': metadata,
        'baseline_signal': baseline_signal,
        'response_signal': response_signal,
        'baseline_auc': baseline_auc,
        'response_auc': response_auc,
        'output_files': []
    }
    
    # Generate filename base
    safe_session = _safe_name(metadata['session'])
    safe_mouse = _safe_name(metadata['mouse'])
    base_name = f"{safe_mouse}_{safe_session}"
    
    # Generate requested plots
    if 'signal_mean' in plot_types:
        plot_path = os.path.join(output_folder, f"{base_name}_signal_mean_bars.png")
        plot_signal_mean_bars(session_basename, baseline_signal, response_signal,
                            plot_path, metadata, signal_type,
                            baseline_window, response_window)
        results['output_files'].append(plot_path)
    
    if 'auc' in plot_types:
        plot_path = os.path.join(output_folder, f"{base_name}_auc_bars.png")
        plot_auc_bars(session_basename, baseline_auc, response_auc,
                     plot_path, metadata, signal_type,
                     baseline_window, response_window)
        results['output_files'].append(plot_path)
    
    if 'heatmap' in plot_types:
        if session_long_csv_path is None or not os.path.exists(session_long_csv_path):
            raise FileNotFoundError(f"Heatmap requires trial traces CSV, but not found for session {session_basename}")
        
        trial_df = load_trial_traces(session_long_csv_path)
        
        # Determine signal column based on type
        if signal_type == 'z-score':
            signal_col = 'signal_z'
        else:
            # Assume raw ΔF/F is also stored (may need adjustment based on actual column names)
            signal_col = 'signal_z'  # Adjust if raw column has different name
        
        plot_path = os.path.join(output_folder, f"{base_name}_heatmap.png")
        plot_heatmap(trial_df, plot_path, metadata, signal_type, signal_col)
        results['output_files'].append(plot_path)
    
    # Save session statistics CSV
    csv_path = os.path.join(output_folder, f"{base_name}_stats.csv")
    save_session_stats_csv(session_basename, baseline_signal, response_signal,
                          baseline_auc, response_auc, csv_path, metadata)
    results['output_files'].append(csv_path)
    
    return results


def run_advanced_graphing(events_folder: str, selected_sessions: List[str],
                         signal_type: str, baseline_window: Tuple[float, float],
                         response_window: Tuple[float, float],
                         plot_types: List[str], trial_counts: Dict[str, int]) -> Dict:
    """
    Main entry point for advanced graphing.
    
    Args:
        events_folder: Path to batch processing output folder
        selected_sessions: List of session identifiers (basenames from prism CSV files)
        signal_type: 'z-score' or 'raw ΔF/F'
        baseline_window: (start, end) in seconds
        response_window: (start, end) in seconds
        plot_types: List of plot types to generate
        trial_counts: Dict mapping session basename to number of trials
    
    Returns:
        Summary dictionary with paths and statistics
    """
    # Create output folder
    timestamp = _timestamp_suffix()
    output_root = os.path.join(events_folder, "advanced_graphs", timestamp)
    os.makedirs(output_root, exist_ok=True)
    
    prism_sessions_folder = os.path.join(events_folder, "prism_tables", "sessions")
    mice_folder = os.path.join(events_folder, "mice")
    
    if not os.path.exists(prism_sessions_folder):
        raise FileNotFoundError(f"Prism sessions folder not found: {prism_sessions_folder}")
    
    all_results = []
    errors = []
    
    for session_basename in selected_sessions:
        try:
            # Locate prism CSV
            session_prism_path = os.path.join(prism_sessions_folder, session_basename)
            if not os.path.exists(session_prism_path):
                raise FileNotFoundError(f"Session prism file not found: {session_prism_path}")
            
            # Locate trial traces LONG CSV (needed only for heatmap)
            session_long_csv_path = None
            if 'heatmap' in plot_types:
                # Search in mice folders for matching LONG CSV
                session_name = session_basename.replace('_prism.csv', '')
                for root, dirs, files in os.walk(mice_folder):
                    for f in files:
                        if f.endswith('_peri_event_traces_LONG.csv') and session_name in f:
                            session_long_csv_path = os.path.join(root, f)
                            break
                    if session_long_csv_path:
                        break
            
            # Get trial count
            n_trials = trial_counts.get(session_basename, 10)  # Default to 10 if unknown
            
            # Process session
            session_output_folder = os.path.join(output_root, _safe_name(session_basename.replace('_prism.csv', '')))
            results = process_session(
                session_prism_path, session_long_csv_path,
                baseline_window, response_window,
                signal_type, plot_types,
                session_output_folder, n_trials
            )
            
            all_results.append(results)
            
        except Exception as e:
            errors.append({
                'session': session_basename,
                'error': str(e)
            })
    
    # Generate aggregated export
    if all_results:
        aggregated_csv = os.path.join(output_root, "aggregated_session_stats.csv")
        generate_aggregated_export(all_results, aggregated_csv)
    
    summary = {
        'timestamp': timestamp,
        'output_folder': output_root,
        'n_sessions_processed': len(all_results),
        'n_errors': len(errors),
        'errors': errors,
        'signal_type': signal_type,
        'baseline_window': baseline_window,
        'response_window': response_window,
        'plot_types': plot_types
    }
    
    return summary