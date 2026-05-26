import os
import json
import math
import itertools
import hashlib
from collections import defaultdict
from datetime import datetime as dt

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from splice_utils import load_splices, get_splice_mask, check_snippet_overlaps_splice, offset_splices
from math_utils import (
    downsample_block_average, match_lengths, compute_dff,
    zscore_baseline_with_fallback, compute_sem,
    apply_pct_alignment as _apply_pct_alignment,
    apply_code0_cutoff, compute_pct_offset,
    compute_peri_time_vector, compute_trial_metrics,
)

# helper safe name
def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

# simple stream finder (case-insensitive substring)
def find_stream_by_substr(block, substr):
    for k in getattr(block, "streams", {}).keys():
        if substr.lower() in k.lower():
            return k
    return None

# simple loader wrapper - expects 'tdt' available in the environment
def load_tdt_block(path):
    try:
        import tdt
    except Exception as e:
        raise RuntimeError("tdt library not available. Install tdt or provide a custom loader.") from e
    return tdt.read_block(path)

# Prism export helpers
def save_prism_xy(time_array, mean_array, sem_array, out_path):
    """
    Save a three-column XY table (time, mean, sem) suitable for Prism XY.
    No rounding (full float precision).
    """
    df = pd.DataFrame({
        "time_s": time_array,
        "mean": mean_array,
        "sem": sem_array
    })
    df.to_csv(out_path, index=False)

def save_prism_multigroup(time_array, group_mean_dict, group_sem_dict, out_path):
    """
    Save a multi-group table with columns:
    time_s, <group1>_mean, <group1>_sem, <group2>_mean, <group2>_sem, ...
    """
    cols = {"time_s": time_array}
    for g in group_mean_dict:
        cols[f"{g}_mean"] = group_mean_dict[g]
        cols[f"{g}_sem"] = group_sem_dict[g]
    df = pd.DataFrame(cols)
    df.to_csv(out_path, index=False)

def run_batch_processing(
    base_path,
    selected_event,
    pre_t, post_t,
    baseline_lower, baseline_upper,
    downsample_factor,
    metric_start, metric_end,
    metadata_list,
    groups_dict,
    find_stream_by_substr=find_stream_by_substr,
    load_tdt_block=load_tdt_block,
    verbose=True,
    code12_index=1,
    pct_onset_index=1,
    pct_onset_map=None,
    code0_cutoff=False,
    signal_start_cutoff=0.0,
    global_zscore=False
):
    timestamp_str = dt.now().strftime("%Y%m%d_%H%M%S")
    event_folder = os.path.join(base_path, "plots", f"{_safe_name(selected_event)}_{timestamp_str}")
    mice_root = os.path.join(event_folder, "mice")
    groups_root = os.path.join(event_folder, "groups")
    prism_root = os.path.join(event_folder, "prism_tables")
    prism_sessions = os.path.join(prism_root, "sessions")
    prism_mice = os.path.join(prism_root, "mice")
    prism_groups = os.path.join(prism_root, "groups")
    prism_comparison = os.path.join(prism_root, "comparison")

    for p in [mice_root, groups_root, prism_sessions, prism_mice, prism_groups, prism_comparison]:
        os.makedirs(p, exist_ok=True)

    def norm(p): return os.path.normpath(p)
    # This is CRITICAL for dual-mouse sessions
    meta_lookup = {}
    for m in (metadata_list or []):
        key = (norm(m['path']), m.get('mouseID'))
        meta_lookup[key] = m
    
    if verbose:
        print(f"\n=== METADATA LOOKUP DEBUG ===")
        print(f"Total metadata entries: {len(metadata_list)}")
        print(f"Unique (path, mouseID) combinations: {len(meta_lookup)}")
        for key, meta in list(meta_lookup.items())[:5]:  # Show first 5
            print(f"  {key[1]} @ {os.path.basename(key[0])} -> signalChannelSet={meta.get('signalChannelSet')}")
        print("=" * 50 + "\n")

    all_trial_rows = []
    all_trace_dfs = []  

    summary = {
        "run_timestamp": timestamp_str,
        "n_groups": 0,
        "n_mice": 0,
        "n_sessions": 0,
        "n_trials_total": 0,
        "valid_trials": 0,
        "invalid_trials": 0,
        "output_files": {},
        "errors": []
    }

    processed_sessions = set()
    mice_seen = set()
    groups_seen = set()

    per_mouse_pooled_trials = defaultdict(list)   # mouse -> list of trial arrays (all sessions)
    per_mouse_session_counts = defaultdict(int)
    per_mouse_summary_metrics = defaultdict(list) # global per-mouse trial rows

    # NEW: per-group per-mouse pools so group means only use sessions assigned to that group
    per_mouse_pooled_trials_by_group = defaultdict(lambda: defaultdict(list))  # group -> mouse -> list of trial arrays
    per_mouse_summary_metrics_by_group = defaultdict(lambda: defaultdict(list))

    color_cycle = itertools.cycle([
        "red", "green", "purple", "orange", "brown", "pink", "gray"
    ])

    summary["n_groups"] = len(groups_dict or {})
    for group_name, mice in (groups_dict or {}).items():
        groups_seen.add(group_name)
        for mouse_id, session_paths in (mice or {}).items():
            mice_seen.add(mouse_id)
            for s_path in session_paths:
                summary["n_sessions"] += 1
                if verbose:
                    print(f"\n--- Processing block ---", flush=True)
                    print(f"Group: {group_name} | Mouse: {mouse_id}", flush=True)
                    print(f"Session path: {s_path}", flush=True)

                norm_path = norm(s_path)
                
                # *** FIX: Use mouse-specific metadata lookup ***
                meta_key = (norm_path, mouse_id)
                meta = meta_lookup.get(meta_key)
                
                if meta is None:
                    err = f"metadata_missing for mouse {mouse_id} in {s_path}"
                    print(f"ERROR: {err}")
                    print(f"  Available keys for this path:")
                    for k in meta_lookup.keys():
                        if k[0] == norm_path:
                            print(f"    - Mouse: {k[1]}")
                    summary["errors"].append({"session": s_path, "mouse": mouse_id, "error": err})
                    continue

                # *** FIX: Get the CORRECT signalChannelSet for THIS mouse ***
                scs = int(meta.get("signalChannelSet", 1))
                if verbose:
                    print(f"Mouse {mouse_id} -> signalChannelSet = {scs}")
                
                if scs == 1:
                    sig_sub, ctrl_sub = "_465A", "_415A"
                else:
                    sig_sub, ctrl_sub = "_465C", "_415C"
                
                if verbose:
                    print(f"Using channels: Signal={sig_sub}, Control={ctrl_sub}")
                try:
                    block = load_tdt_block(s_path)
                except Exception as e:
                    err = f"failed_to_load_block: {e}"
                    print("ERROR:", err, flush=True)
                    summary["errors"].append({"session": s_path, "error": err})
                    continue

                try:
                    avail_streams = list(block.streams.keys())
                except Exception:
                    avail_streams = []
                if verbose:
                    print(f"Loaded block: {s_path}", flush=True)
                    print("Available stream keys:", avail_streams, flush=True)

                sig_key = find_stream_by_substr(block, sig_sub)
                ctrl_key = find_stream_by_substr(block, ctrl_sub)
                if sig_key is None or ctrl_key is None:
                    err = f"streams_missing expected {sig_sub} / {ctrl_sub}"
                    print("WARNING:", err, "available:", avail_streams, flush=True)
                    summary["errors"].append({"session": s_path, "error": err, "available_streams": avail_streams})
                    continue

                try:
                    sig = np.asarray(block.streams[sig_key].data).flatten()
                    ctrl = np.asarray(block.streams[ctrl_key].data).flatten()
                    fs_orig = float(block.streams[sig_key].fs)
                except Exception as e:
                    err = f"failed_extract_streams: {e}"
                    print("ERROR:", err, flush=True)
                    summary["errors"].append({"session": s_path, "error": err})
                    continue

                if verbose:
                    print(f"Signal key: {sig_key}, Control key: {ctrl_key}", flush=True)
                    print(f"Signal len={len(sig)}, Control len={len(ctrl)}, Fs_orig={fs_orig}", flush=True)

                event_times = [e['timestamp_s'] for e in meta.get("events", []) if e.get("event") == selected_event]
                offset = 0.0  # default; overwritten if PCT alignment succeeds

                # Determine PCT onset index for this session (per-session map overrides global)
                _session_pct_idx = pct_onset_index
                if pct_onset_map:
                    _pct_key = (s_path, mouse_id)
                    if _pct_key in pct_onset_map:
                        _session_pct_idx = pct_onset_map[_pct_key]
                    elif norm_path in pct_onset_map:
                        _session_pct_idx = pct_onset_map[norm_path]
                if verbose:
                    print(f"PCT onset index for this session: {_session_pct_idx}", flush=True)

                try:
                    if meta.get('event_interpretor', 1) == 2:
                        scs_val = meta.get('signalChannelSet', 1)
                        pct_name = 'PtC1' if scs_val == 1 else 'PtC2'
                        c_name = 'C1__' if scs_val == 1 else 'C2__'
                        
                        onset_list = None
                        c12_idx = 1  # default to 1 (second code 12) for PtC
                        used_epoc = None
                        
                        try:
                            # Prefer C1__/C2__ which uses the first code 12
                            if c_name in block.epocs:
                                onset_list = block.epocs[c_name].onset
                                c12_idx = 0
                                used_epoc = c_name
                            elif pct_name in block.epocs:
                                onset_list = block.epocs[pct_name].onset
                                c12_idx = 1
                                used_epoc = pct_name
                        except Exception:
                            pass

                        code12_times = [e["timestamp_s"] for e in meta.get("events", []) if e.get("code") == 12]
                        required_pct = _session_pct_idx + 1
                        required_code12s = c12_idx + 1
                        
                        if onset_list is not None and len(onset_list) >= required_pct and len(code12_times) >= required_code12s:
                            pct_onset = float(onset_list[_session_pct_idx])
                            offset = compute_pct_offset(pct_onset, code12_times, code12_index=c12_idx)
                            sig, ctrl = _apply_pct_alignment(sig, ctrl, fs_orig, offset)
                            event_times = [t - offset for t in event_times]
                            if verbose:
                                print(f"Alignment applied using {used_epoc} onset index {_session_pct_idx}, code12 index {c12_idx}. offset={offset:.3f}s trimmed samples -> {len(sig)} remain", flush=True)
                        else:
                            if verbose:
                                print(f"Alignment conditions not met (need {required_pct} onset(s) and {required_code12s} code12(s)); skipping alignment", flush=True)
                except Exception as e:
                    print("WARNING: alignment failed:", e, flush=True)

                # --- Code 0 cutoff: truncate signal/control after code 0 timestamp ---
                if code0_cutoff:
                    _code0_times = [e["timestamp_s"] for e in meta.get("events", []) if e.get("code") == 0]
                    if _code0_times:
                        _code0_t = _code0_times[0]
                        sig, ctrl = apply_code0_cutoff(sig, ctrl, fs_orig, _code0_t)
                        event_times = [t for t in event_times if t <= _code0_t]
                        if verbose:
                            print(f"Code 0 cutoff at {_code0_t:.2f}s -> {len(sig)} samples remain", flush=True)

                # --- Signal start cutoff: match Tab 2 by dropping bad early signal before z-scoring ---
                signal_start_cutoff = float(signal_start_cutoff or 0.0)
                if signal_start_cutoff > 0:
                    start_sample = int(round(signal_start_cutoff * fs_orig))
                    if start_sample > 0 and start_sample < len(sig):
                        sig = sig[start_sample:]
                        ctrl = ctrl[start_sample:]
                        event_times = [t - signal_start_cutoff for t in event_times]
                        event_times = [t for t in event_times if t >= 0]
                        if verbose:
                            print(f"Signal start cutoff at {signal_start_cutoff:.2f}s -> {len(sig)} samples remain; events shifted by -{signal_start_cutoff:.2f}s", flush=True)

                if len(event_times) == 0:
                    warn = f"no '{selected_event}' events in session {s_path}"
                    print("WARNING:", warn, flush=True)
                    summary["errors"].append({"session": s_path, "error": warn})
                    continue

                # --- Load and adjust splices for this block (per channel) ---
                block_splices_raw = load_splices(s_path, channel=scs)
                block_splices = offset_splices(block_splices_raw, offset) if block_splices_raw else []
                if verbose and block_splices:
                    print(f"Loaded {len(block_splices)} splice region(s) for this block", flush=True)
                    for _ss, _se in block_splices:
                        print(f"  Splice: {_ss:.2f}s - {_se:.2f}s (aligned coords)", flush=True)

                ds = int(downsample_factor)
                if ds > 1:
                    # Block-averaging downsample (Lerner et al. 2015)
                    sig = downsample_block_average(sig, ds)
                    ctrl = downsample_block_average(ctrl, ds)
                    fs = fs_orig / ds
                else:
                    fs = fs_orig

                if verbose:
                    print(f"After downsampling: sig_len={len(sig)}, fs={fs}", flush=True)

                # --- Safe truncation before regression ---
                _orig_sig_len, _orig_ctrl_len = len(sig), len(ctrl)
                sig, ctrl = match_lengths(sig, ctrl)
                min_len = len(sig)
                if _orig_sig_len != _orig_ctrl_len:
                    if verbose:
                        print(f"Signal/control length mismatch: sig={_orig_sig_len}, ctrl={_orig_ctrl_len}. Truncating to {min_len}.", flush=True)

                try:
                    # Exclude spliced regions from polyfit regression
                    if block_splices:
                        splice_mask = get_splice_mask(min_len, fs, block_splices)
                    else:
                        splice_mask = None
                    dff, _fitted, _coeffs = compute_dff(sig, ctrl,
                                                        splice_mask=splice_mask,
                                                        epsilon=1e-12)
                except Exception as e:
                    err = f"regression_dff_failed: {e}"
                    print("ERROR:", err, flush=True)
                    summary["errors"].append({"session": s_path, "error": err})
                    continue

                if global_zscore:
                    dff_mean = float(np.mean(dff))
                    dff_std = float(np.std(dff))
                    if dff_std > 0:
                        dff_z_session = (dff - dff_mean) / dff_std
                    else:
                        dff_z_session = dff - dff_mean

                if verbose:
                    print(f"Computed dF/F len={len(dff)} mean={np.mean(dff):.3f} std={np.std(dff):.3f}", flush=True)
                    if global_zscore:
                        print(f"Using full-session z-score for peri-event traces", flush=True)

                peri_times, n_samples = compute_peri_time_vector(pre_t, post_t, fs)
                if n_samples <= 0:
                    err = f"invalid n_samples computed: {(pre_t+post_t)}*{fs} -> {n_samples}"
                    print("ERROR:", err, flush=True)
                    summary["errors"].append({"session": s_path, "error": err})
                    continue

                if verbose:
                    print(f"Peri window samples: {n_samples}, peri_times[0:3]={peri_times[:3]}", flush=True)

                mouse_folder = os.path.join(mice_root, _safe_name(mouse_id))
                os.makedirs(mouse_folder, exist_ok=True)
                session_basename = _safe_name(os.path.basename(s_path))
                session_plot_path = os.path.join(mouse_folder, f"{session_basename}_session_avg.png")
                session_metrics_path = os.path.join(mouse_folder, f"{session_basename}_metrics.csv")
                session_prism_csv = os.path.join(prism_sessions, f"{_safe_name(mouse_id)}_{session_basename}_prism.csv")

                peri_traces = []
                session_metric_rows = []
                trial_index = 0

                for ts in event_times:
                    start_idx = int(round((ts - pre_t) * fs))
                    end_idx = start_idx + n_samples
                    trial_index += 1
                    if start_idx < 0 or end_idx > len(dff):
                        if verbose:
                            print(f"  Skip trial {trial_index}: start={start_idx} end={end_idx} out of bounds (len={len(dff)})", flush=True)
                        summary["invalid_trials"] += 1
                        summary["n_trials_total"] += 1
                        continue

                    # Skip trials whose peri-event window overlaps a spliced region
                    if block_splices and check_snippet_overlaps_splice(ts, pre_t, post_t, block_splices):
                        if verbose:
                            print(f"  Skip trial {trial_index}: peri-event window overlaps splice region", flush=True)
                        summary["invalid_trials"] += 1
                        summary["n_trials_total"] += 1
                        continue

                    bstart_rel = int(round((baseline_lower + pre_t) * fs))
                    bend_rel = int(round((baseline_upper + pre_t) * fs))

                    if global_zscore:
                        snippet = dff_z_session[start_idx:end_idx].astype(float)
                        bstart_rel = max(0, bstart_rel)
                        bend_rel = min(len(snippet), bend_rel)
                        if bend_rel > bstart_rel:
                            baseline_vals = snippet[bstart_rel:bend_rel]
                            mu = float(np.mean(baseline_vals))
                            snippet_z = snippet - mu
                            zscored_flag = True
                        else:
                            fallback_end = int(round(pre_t * fs))
                            if fallback_end > 0:
                                fallback_vals = snippet[:fallback_end]
                                mu = float(np.mean(fallback_vals))
                                snippet_z = snippet - mu
                                zscored_flag = True
                            else:
                                mu = float(np.mean(dff_z_session))
                                snippet_z = snippet - mu
                                zscored_flag = True
                        
                        peri_traces.append(snippet_z)
                    else:
                        snippet = dff[start_idx:end_idx].astype(float)
                        fallback_end = int(round(pre_t * fs))
                        global_mean = float(np.mean(dff))
                        snippet_z, zscored_flag = zscore_baseline_with_fallback(
                            snippet, bstart_rel, bend_rel,
                            fallback_end_idx=fallback_end,
                            global_mean=global_mean
                        )

                        metrics = compute_trial_metrics(
                            snippet_z, fs, metric_start, metric_end, pre_t,
                            detect_inhibitory=True
                        )
                        peak = metrics["peak"]
                        auc = metrics["auc"]
                        latency = metrics["latency"]

                        peri_traces.append(snippet_z)
                        session_metric_rows.append({
                            "group": group_name,
                            "mouse": mouse_id,
                            "session": os.path.basename(s_path),
                            "trial_index": trial_index,
                            "peak": peak,
                            "auc": auc,
                            "latency": latency,
                            "n_timepoints": len(snippet_z),
                            "zscored": bool(zscored_flag),
                            "fs": fs,
                            "signalChannelSet": scs
                        })

                        all_trial_rows.append(session_metric_rows[-1])
                    summary["n_trials_total"] += 1
                    summary["valid_trials"] += 1

                if len(peri_traces) == 0:
                    print(f"WARNING: no valid peri-event snippets for session {s_path}", flush=True)
                    summary["errors"].append({"session": s_path, "error": "no_valid_trials"})
                    continue

                peri_arr = np.vstack(peri_traces)
                session_mean = np.mean(peri_arr, axis=0)
                session_sem = compute_sem(peri_arr)

                if global_zscore:
                    baseline_mask = (peri_times >= baseline_lower) & (peri_times <= baseline_upper)
                    response_mask = (peri_times >= metric_start) & (peri_times <= metric_end)

                    baseline_segment = session_mean[baseline_mask]
                    response_segment = session_mean[response_mask]
                    response_times = peri_times[response_mask]

                    baseline_mean = float(np.mean(baseline_segment)) if len(baseline_segment) > 0 else np.nan
                    response_mean = float(np.mean(response_segment)) if len(response_segment) > 0 else np.nan
                    delta = response_mean - baseline_mean if np.isfinite(response_mean) and np.isfinite(baseline_mean) else np.nan

                    if len(response_segment) > 0:
                        peak_idx = int(np.argmax(np.abs(response_segment)))
                        peak = float(response_segment[peak_idx])
                        latency = float(response_times[peak_idx])
                        auc = float(np.sum((response_segment[:-1] + response_segment[1:]) * 0.5 * np.diff(response_times))) if len(response_segment) > 1 else 0.0
                    else:
                        peak, latency, auc = np.nan, np.nan, np.nan

                    session_metric_rows = [{
                        "group": group_name,
                        "mouse": mouse_id,
                        "session": os.path.basename(s_path),
                        "metric_source": "session_mean_trace",
                        "baseline_mean": baseline_mean,
                        "response_mean": response_mean,
                        "delta": delta,
                        "peak": peak,
                        "auc": auc,
                        "latency": latency,
                        "n_trials": int(peri_arr.shape[0]),
                        "n_timepoints": int(peri_arr.shape[1]),
                        "session_z_baseline_corrected": True,
                        "fs": fs,
                        "signalChannelSet": scs
                    }]
                    all_trial_rows.append(session_metric_rows[0])
                session_traces_long_csv = os.path.join(
                    mouse_folder, f"{session_basename}_peri_event_traces_LONG.csv"
                )

                n_trials, n_tp = peri_arr.shape
                global_time_s = []

                for ts in event_times[:n_trials]:
                    peri_global = ts + peri_times
                    global_time_s.extend(peri_global)


                df_long = pd.DataFrame({
                    # trial index (1-based, consistent with metrics)
                    "trial_index": np.repeat(np.arange(1, n_trials + 1), n_tp),
                    "global_time_s": global_time_s,

                    # stable integer index for reshaping/pivoting later
                    "time_idx": np.tile(np.arange(n_tp), n_trials),

                    # real peri-event time vector (no rounding)
                    "time_s": np.tile(peri_times, n_trials),

                    # signal value (already baseline-normalized / z-scored)
                    "signal_z": peri_arr.reshape(-1)
                })

                # Attach mouse/session identifiers (CRITICAL for dual-mouse)
                df_long["mouse"] = mouse_id
                df_long["group"] = group_name
                df_long["session"] = os.path.basename(s_path)
                df_long["fs"] = fs
                df_long["signalChannelSet"] = scs

                df_long.to_csv(session_traces_long_csv, index=False)

                summary["output_files"].setdefault(
                    "peri_event_long_csvs", []
                ).append(session_traces_long_csv)

                # collect for optional master export
                all_trace_dfs.append(df_long)

                if verbose:
                    print(f"Saved peri-event LONG traces -> {session_traces_long_csv}", flush=True)


                fig, ax = plt.subplots(figsize=(6,4))
                ax.plot(peri_times, session_mean)
                ax.fill_between(peri_times, session_mean - session_sem, session_mean + session_sem, alpha=0.3)
                ax.axvline(0, color="red", linestyle="--")
                ax.set_title(f"{group_name} | {mouse_id} | {os.path.basename(s_path)}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("ΔF/F (z-scored baseline)")
                fig.tight_layout()
                fig.savefig(session_plot_path)
                plt.close(fig)
                if verbose:
                    print(f"Saved session plot -> {session_plot_path}", flush=True)

                # Save Prism session table
                try:
                    save_prism_xy(peri_times, session_mean, session_sem, session_prism_csv)
                    if verbose:
                        print(f"Saved Prism session CSV -> {session_prism_csv}", flush=True)
                    summary["output_files"].setdefault("prism_session_csvs", []).append(session_prism_csv)
                except Exception as e:
                    print(f"WARNING: failed to save Prism session CSV: {e}", flush=True)

                session_df = pd.DataFrame(session_metric_rows)
                session_df.to_csv(session_metrics_path, index=False)
                if verbose:
                    print(f"Saved session metrics -> {session_metrics_path}", flush=True)

                # append to global per-mouse pool
                for tr in peri_traces:
                    per_mouse_pooled_trials[mouse_id].append(tr)

                # append to per-group per-mouse pool (so group means use only sessions assigned to this group)
                for tr in peri_traces:
                    per_mouse_pooled_trials_by_group[group_name][mouse_id].append(tr)

                per_mouse_session_counts[mouse_id] += 1
                per_mouse_summary_metrics[mouse_id].extend(session_metric_rows)
                per_mouse_summary_metrics_by_group[group_name][mouse_id].extend(session_metric_rows)

                processed_sessions.add(s_path)

    # -------------------
    # per-mouse aggregation (global combined across all sessions)
    mice_count = len(per_mouse_pooled_trials)
    summary["n_mice"] = len(mice_seen)
    if verbose:
        print(f"\n--- Per-mouse pooling for {mice_count} mice ---", flush=True)

    # DEBUG dump before building group means
    def arr_hash(a):
        try:
            b = np.round(a, 6).tobytes()
        except Exception:
            a = np.asarray(a)
            b = np.round(a, 6).tobytes()
        return hashlib.md5(b).hexdigest()

    print("\n--- DEBUG DUMP START ---", flush=True)
    print("groups_dict (raw):", repr(groups_dict), flush=True)
    print("mice_seen:", repr(mice_seen), flush=True)

    print("\nper_mouse_pooled_trials keys and info:", flush=True)
    for m, trials in per_mouse_pooled_trials.items():
        try:
            trials_arr = np.vstack(trials)
            mean_trace = np.mean(trials_arr, axis=0)
            print(f"  MOUSE: {m} | n_trials={len(trials)} | mean_shape={mean_trace.shape} | hash={arr_hash(mean_trace)} | first5={np.around(mean_trace[:5],4).tolist()}", flush=True)
        except Exception as e:
            print(f"  MOUSE: {m} | ERROR building mean: {e}", flush=True)

    print("\nper_group_mouse_means_by_group (pre-build) summary:", flush=True)
    for gname, mice_map in per_mouse_pooled_trials_by_group.items():
        counts = {m: len(trs) for m, trs in mice_map.items()}
        print(f"  GROUP: {gname} -> {counts}", flush=True)

    try:
        for root, dirs, files in os.walk(event_folder):
            if root == event_folder:
                print("\nTop-level event_folder contents:", "dirs:", dirs, "files:", files, flush=True)
                break
    except Exception as e:
        print("Could not list event_folder:", e, flush=True)

    print("--- DEBUG DUMP END ---\n", flush=True)

    # -------------------
    # Build per-group mouse means from per_mouse_pooled_trials_by_group
    per_group_mouse_means = defaultdict(list)
    per_group_mouse_metrics_rows = defaultdict(list)

    for gname in (groups_dict or {}).keys():
        mice_map = per_mouse_pooled_trials_by_group.get(gname, {})
        for mouse_id, trials in mice_map.items():
            if not trials:
                continue
            trials_arr = np.vstack(trials)
            mouse_mean = np.mean(trials_arr, axis=0)
            mouse_sem = compute_sem(trials_arr)

            per_group_mouse_means[gname].append({
                "mouse": mouse_id,
                "mean": mouse_mean,
                "sem": mouse_sem
            })

            # compute per-mouse summary metrics for this group
            mrows = per_mouse_summary_metrics_by_group.get(gname, {}).get(mouse_id, [])
            if len(mrows) > 0:
                mdf = pd.DataFrame(mrows)
                mean_peak = float(mdf["peak"].mean(skipna=True))
                mean_auc = float(mdf["auc"].mean(skipna=True))
                mean_latency = float(mdf["latency"].mean(skipna=True))
                n_trials = int(len(mdf))
            else:
                mean_peak = mean_auc = mean_latency = np.nan
                n_trials = 0

            per_group_mouse_metrics_rows[gname].append({
                "group": gname,
                "mouse": mouse_id,
                "mean_peak": mean_peak,
                "mean_auc": mean_auc,
                "mean_latency": mean_latency,
                "n_trials": n_trials
            })

    # -------------------
    # Ensure single comparison folder exists
    comparison_root = os.path.join(event_folder, "comparison")
    os.makedirs(comparison_root, exist_ok=True)
    summary["output_files"]["comparison_root"] = comparison_root

    # -------------------
    # group-level aggregates and plots
    if verbose:
        print("\n--- Group-level aggregation ---", flush=True)

    group_color_cycle = itertools.cycle([
        "red", "green", "purple", "orange", "brown", "pink", "gray"
    ])

    group_means_for_comparison = {}
    # For Prism multi-group table construction
    group_mean_dict = {}
    group_sem_dict = {}

    for gname, mouse_infos in per_group_mouse_means.items():
        gfolder = os.path.join(groups_root, _safe_name(gname))
        os.makedirs(gfolder, exist_ok=True)
        group_mean_plot = os.path.join(gfolder, f"{_safe_name(gname)}_mean_trace.png")
        group_all_mice_plot = os.path.join(gfolder, f"{_safe_name(gname)}_all_mice_traces.png")
        group_summary_csv = os.path.join(gfolder, f"{_safe_name(gname)}_summary_metrics.csv")
        group_prism_csv = os.path.join(prism_groups, f"{_safe_name(gname)}_group_mean_prism.csv")

        if len(mouse_infos) == 0:
            print(f"WARNING: no mouse means for group {gname}", flush=True)
            summary["errors"].append({"group": gname, "error": "no_mouse_means"})
            continue

        arr = np.vstack([mi["mean"] for mi in mouse_infos])
        g_mean = np.mean(arr, axis=0)
        g_sem = compute_sem(arr)
        group_means_for_comparison[gname] = (g_mean, g_sem)
        group_mean_dict[gname] = g_mean
        group_sem_dict[gname] = g_sem

        group_color = next(group_color_cycle)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(peri_times, g_mean, label=f"{gname} mean", color=group_color)
        ax.fill_between(peri_times, g_mean - g_sem, g_mean + g_sem, alpha=0.25, color=group_color)

        for mi in mouse_infos:
            ax.plot(peri_times, mi["mean"], alpha=0.6, linestyle='-', label=mi["mouse"], linewidth=1.0)

        ax.axvline(0, color="black", linestyle="--")
        ax.set_title(f"Group avg: {gname}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔF/F (z-scored baseline)")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        fig.savefig(group_mean_plot)
        plt.close(fig)
        if verbose:
            print(f"Saved group mean plot -> {group_mean_plot}", flush=True)

        # Save Prism group table (time, mean, sem)
        try:
            save_prism_xy(peri_times, g_mean, g_sem, group_prism_csv)
            if verbose:
                print(f"Saved Prism group CSV -> {group_prism_csv}", flush=True)
            summary["output_files"].setdefault("prism_group_csvs", []).append(group_prism_csv)
        except Exception as e:
            print(f"WARNING: failed to save Prism group CSV: {e}", flush=True)

        fig, ax = plt.subplots(figsize=(8,5))
        for mi in mouse_infos:
            ax.plot(peri_times, mi["mean"], label=mi["mouse"], alpha=0.8)
            ax.fill_between(peri_times, mi["mean"] - mi["sem"], mi["mean"] + mi["sem"], alpha=0.15)

        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔF/F (z-scored baseline)")
        ax.set_title(f"All mice traces: {gname}")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        fig.savefig(group_all_mice_plot)
        plt.close(fig)
        if verbose:
            print(f"Saved group all-mice overlay -> {group_all_mice_plot}", flush=True)

        gm_df = pd.DataFrame(per_group_mouse_metrics_rows.get(gname, []))
        gm_df.to_csv(group_summary_csv, index=False)
        if verbose:
            print(f"Saved group summary metrics -> {group_summary_csv}", flush=True)

        summary["output_files"].setdefault("group_plots", []).append(group_mean_plot)
        summary["output_files"].setdefault("group_overlays", []).append(group_all_mice_plot)
        summary["output_files"].setdefault("group_csvs", []).append(group_summary_csv)

    # -------------------
    # final group comparison across groups: overlay all group means with SEM shading
    combined_fig_path = os.path.join(comparison_root, f"{_safe_name(selected_event)}_group_comparison.png")
    fig, ax = plt.subplots(figsize=(10,6))

    group_color_cycle = itertools.cycle([
        "red", "green", "purple", "orange", "brown", "pink", "gray"
    ])

    for gname, (g_mean, g_sem) in group_means_for_comparison.items():
        color = next(group_color_cycle)
        ax.plot(peri_times, g_mean, label=gname, color=color)
        ax.fill_between(peri_times, g_mean - g_sem, g_mean + g_sem, alpha=0.2, color=color)

    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (z-scored baseline)")
    ax.set_title(f"Group comparison — Event: {selected_event}")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(combined_fig_path)
    plt.close(fig)
    summary["output_files"]["group_comparison_plot"] = combined_fig_path
    if verbose:
        print(f"Saved group comparison plot -> {combined_fig_path}", flush=True)

    # Save Prism multi-group comparison CSV
    try:
        comparison_prism_csv = os.path.join(prism_comparison, f"{_safe_name(selected_event)}_group_comparison_prism.csv")
        save_prism_multigroup(peri_times, group_mean_dict, group_sem_dict, comparison_prism_csv)
        summary["output_files"]["prism_comparison_csv"] = comparison_prism_csv
        if verbose:
            print(f"Saved Prism comparison CSV -> {comparison_prism_csv}", flush=True)
    except Exception as e:
        print(f"WARNING: failed to save Prism comparison CSV: {e}", flush=True)

    # -------------------
    if len(all_trial_rows) > 0:
        all_trials_df = pd.DataFrame(all_trial_rows)
        if global_zscore:
            all_trials_df.sort_values(by=["group", "mouse", "session"], inplace=True)
            all_trials_csv = os.path.join(event_folder, "all_groups_session_mean_metrics.csv")
        else:
            all_trials_csv = os.path.join(event_folder, "all_groups_trial_metrics.csv")
        all_trials_df.to_csv(all_trials_csv, index=False)
        summary["output_files"]["all_groups_trial_metrics_csv"] = all_trials_csv
        if verbose:
            print(f"Saved master trial CSV -> {all_trials_csv}", flush=True)
    else:
        if verbose:
            print("No trial rows collected; master trial CSV not created.", flush=True)

    if len(all_trace_dfs) > 0:
        all_traces_df = pd.concat(all_trace_dfs, ignore_index=True)
        all_traces_csv = os.path.join(
            event_folder, "all_groups_peri_event_traces_LONG.csv"
        )
        all_traces_df.to_csv(all_traces_csv, index=False)
        summary["output_files"]["all_groups_peri_event_traces_LONG_csv"] = all_traces_csv
        if verbose:
            print(f"Saved MASTER peri-event LONG traces -> {all_traces_csv}", flush=True)


    # Also save per-mouse Prism tables (global combined)
    for mouse_id, trials in per_mouse_pooled_trials.items():
        try:
            if not trials:
                continue
            trials_arr = np.vstack(trials)
            mouse_mean = np.mean(trials_arr, axis=0)
            mouse_sem = compute_sem(trials_arr)
            mouse_prism_csv = os.path.join(prism_mice, f"{_safe_name(mouse_id)}_combined_prism.csv")
            save_prism_xy(peri_times, mouse_mean, mouse_sem, mouse_prism_csv)
            summary["output_files"].setdefault("prism_mouse_csvs", []).append(mouse_prism_csv)
            if verbose:
                print(f"Saved Prism mouse CSV -> {mouse_prism_csv}", flush=True)
        except Exception as e:
            print(f"WARNING: failed to save Prism mouse CSV for {mouse_id}: {e}", flush=True)

    summary["n_mice"] = len(mice_seen)
    summary["n_sessions"] = summary["n_sessions"]
    summary["n_trials_total"] = summary["n_trials_total"]
    summary["valid_trials"] = summary["valid_trials"]
    summary["invalid_trials"] = summary["invalid_trials"]

    summary["output_files"].update({
        "event_folder": event_folder,
        "mice_root": mice_root,
        "groups_root": groups_root,
        "prism_root": prism_root,
        "prism_sessions": prism_sessions,
        "prism_mice": prism_mice,
        "prism_groups": prism_groups,
        "prism_comparison": prism_comparison
    })

    summary_path = os.path.join(event_folder, "summary_log.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["output_files"]["summary_log"] = summary_path
    if verbose:
        print(f"Saved summary_log.json -> {summary_path}", flush=True)

    return summary
