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
    verbose=True
):
    timestamp_str = dt.now().strftime("%Y%m%d_%H%M%S")
    event_folder = os.path.join(base_path, "plots", f"{_safe_name(selected_event)}_{timestamp_str}")
    mice_root = os.path.join(event_folder, "mice")
    groups_root = os.path.join(event_folder, "groups")
    os.makedirs(mice_root, exist_ok=True)
    os.makedirs(groups_root, exist_ok=True)

    def norm(p): return os.path.normpath(p)
    
    # Create a lookup that maps (path, mouseID) -> metadata entry
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

    per_mouse_pooled_trials = defaultdict(list)
    per_mouse_session_counts = defaultdict(int)
    per_mouse_summary_metrics = defaultdict(list)

    per_mouse_pooled_trials_by_group = defaultdict(lambda: defaultdict(list))
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
                    print(f"\n{'='*60}")
                    print(f"Processing: Group={group_name} | Mouse={mouse_id}")
                    print(f"Session: {s_path}")
                    print('='*60)

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
                    print(f"ERROR: {err}")
                    summary["errors"].append({"session": s_path, "error": err})
                    continue

                try:
                    avail_streams = list(block.streams.keys())
                except Exception:
                    avail_streams = []
                
                if verbose:
                    print(f"Available streams: {avail_streams}")

                sig_key = find_stream_by_substr(block, sig_sub)
                ctrl_key = find_stream_by_substr(block, ctrl_sub)
                
                if sig_key is None or ctrl_key is None:
                    err = f"streams_missing: expected {sig_sub}/{ctrl_sub}"
                    print(f"ERROR: {err}")
                    print(f"  Available: {avail_streams}")
                    summary["errors"].append({
                        "session": s_path, 
                        "mouse": mouse_id,
                        "error": err, 
                        "available_streams": avail_streams
                    })
                    continue

                try:
                    sig = np.asarray(block.streams[sig_key].data).flatten()
                    ctrl = np.asarray(block.streams[ctrl_key].data).flatten()
                    fs_orig = float(block.streams[sig_key].fs)
                except Exception as e:
                    err = f"failed_extract_streams: {e}"
                    print(f"ERROR: {err}")
                    summary["errors"].append({"session": s_path, "error": err})
                    continue

                if verbose:
                    print(f"Extracted: {sig_key} (len={len(sig)}), {ctrl_key} (len={len(ctrl)}), Fs={fs_orig}")

                # Get events for THIS mouse from the metadata
                event_times = [e['timestamp_s'] for e in meta.get("events", []) if e.get("event") == selected_event]
                
                # Alignment logic
                try:
                    if meta.get('interpretor', 1) == 2:
                        pct_name = 'PtC1' if scs == 1 else 'PtC2'
                        onset_list = None
                        try:
                            onset_list = block.epocs[pct_name].onset
                        except Exception:
                            pass
                        code12_times = [e["timestamp_s"] for e in meta.get("events", []) if e.get("code") == 12]
                        if onset_list is not None and len(onset_list) > 1 and len(code12_times) >= 2:
                            pct_onset = float(onset_list[1])
                            offset = pct_onset - code12_times[1]
                            time_full = np.arange(len(sig)) / fs_orig
                            time_full -= offset
                            valid = time_full >= 0
                            sig = sig[valid]
                            ctrl = ctrl[valid]
                            event_times = [t - offset for t in event_times]
                            if verbose:
                                print(f"Alignment: offset={offset:.3f}s, samples after trim={len(sig)}")
                        else:
                            if verbose:
                                print("Alignment conditions not met; skipping")
                except Exception as e:
                    print(f"WARNING: alignment failed: {e}")

                if len(event_times) == 0:
                    warn = f"no '{selected_event}' events for mouse {mouse_id}"
                    print(f"WARNING: {warn}")
                    summary["errors"].append({"session": s_path, "mouse": mouse_id, "error": warn})
                    continue

                # Downsampling
                ds = int(downsample_factor)
                if ds > 1:
                    sig = sig[::ds]
                    ctrl = ctrl[::ds]
                    fs = fs_orig / ds
                else:
                    fs = fs_orig

                # Length matching
                min_len = min(len(sig), len(ctrl))
                if len(sig) != len(ctrl):
                    if verbose:
                        print(f"Length mismatch: sig={len(sig)}, ctrl={len(ctrl)}. Truncating to {min_len}")
                    sig = sig[:min_len]
                    ctrl = ctrl[:min_len]

                # dF/F calculation
                try:
                    p = np.polyfit(ctrl, sig, 1)
                    fitted = p[0] * ctrl + p[1]
                    dff = 100.0 * (sig - fitted) / (fitted + 1e-12)
                except Exception as e:
                    err = f"regression_dff_failed: {e}"
                    print(f"ERROR: {err}")
                    summary["errors"].append({"session": s_path, "error": err})
                    continue

                if verbose:
                    print(f"dF/F: len={len(dff)}, mean={np.mean(dff):.3f}, std={np.std(dff):.3f}")

                # Peri-event window setup
                n_samples = int(round((pre_t + post_t) * fs))
                if n_samples <= 0:
                    err = f"invalid n_samples: {n_samples}"
                    print(f"ERROR: {err}")
                    summary["errors"].append({"session": s_path, "error": err})
                    continue
                peri_times = (np.arange(n_samples) / fs) - pre_t

                # Create output folders
                mouse_folder = os.path.join(mice_root, _safe_name(mouse_id))
                os.makedirs(mouse_folder, exist_ok=True)
                session_basename = _safe_name(os.path.basename(s_path))
                session_plot_path = os.path.join(mouse_folder, f"{session_basename}_session_avg.png")
                session_metrics_path = os.path.join(mouse_folder, f"{session_basename}_metrics.csv")

                peri_traces = []
                session_metric_rows = []
                trial_index = 0

                # Extract trials
                for ts in event_times:
                    start_idx = int(round((ts - pre_t) * fs))
                    end_idx = start_idx + n_samples
                    trial_index += 1
                    
                    if start_idx < 0 or end_idx > len(dff):
                        if verbose:
                            print(f"  Skip trial {trial_index}: out of bounds")
                        summary["invalid_trials"] += 1
                        summary["n_trials_total"] += 1
                        continue

                    snippet = dff[start_idx:end_idx].astype(float)

                    # Baseline z-scoring
                    bstart_rel = int(round((baseline_lower + pre_t) * fs))
                    bend_rel = int(round((baseline_upper + pre_t) * fs))
                    bstart_rel = max(0, bstart_rel)
                    bend_rel = min(len(snippet), bend_rel)
                    
                    if bend_rel > bstart_rel:
                        baseline_vals = snippet[bstart_rel:bend_rel]
                        mu = float(np.mean(baseline_vals))
                        sigma = float(np.std(baseline_vals))
                        if sigma > 0:
                            snippet_z = (snippet - mu) / sigma
                            zscored_flag = True
                        else:
                            snippet_z = snippet - mu
                            zscored_flag = False
                    else:
                        fallback_end = int(round(pre_t * fs))
                        if fallback_end > 0:
                            fallback_vals = snippet[:fallback_end]
                            mu = float(np.mean(fallback_vals))
                            sigma = float(np.std(fallback_vals))
                            if sigma > 0:
                                snippet_z = (snippet - mu) / sigma
                                zscored_flag = True
                            else:
                                snippet_z = snippet - mu
                                zscored_flag = False
                        else:
                            mu = float(np.mean(dff))
                            snippet_z = snippet - mu
                            zscored_flag = False

                    # Compute metrics
                    mstart_idx = int(round((pre_t + metric_start) * fs))
                    mend_idx = int(round((pre_t + metric_end) * fs))
                    mstart_idx = max(0, mstart_idx)
                    mend_idx = min(len(snippet_z), mend_idx)

                    if mend_idx > mstart_idx:
                        post_segment = snippet_z[mstart_idx:mend_idx]
                        peak = float(np.max(post_segment))
                        auc = float(np.trapz(post_segment, dx=1.0/fs))
                        latency_idx = int(np.argmax(post_segment))
                        latency = float(latency_idx / fs + metric_start)
                    else:
                        peak, auc, latency = np.nan, np.nan, np.nan

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
                        "signalChannelSet": scs  # Track which channel was used
                    })

                    all_trial_rows.append(session_metric_rows[-1])
                    summary["n_trials_total"] += 1
                    summary["valid_trials"] += 1

                if len(peri_traces) == 0:
                    print(f"WARNING: no valid trials for {mouse_id} in {s_path}")
                    summary["errors"].append({"session": s_path, "mouse": mouse_id, "error": "no_valid_trials"})
                    continue

                # Plot session average
                peri_arr = np.vstack(peri_traces)
                session_mean = np.mean(peri_arr, axis=0)
                session_sem = peri_arr.std(axis=0) / math.sqrt(peri_arr.shape[0])

                fig, ax = plt.subplots(figsize=(6,4))
                ax.plot(peri_times, session_mean)
                ax.fill_between(peri_times, session_mean - session_sem, session_mean + session_sem, alpha=0.3)
                ax.axvline(0, color="red", linestyle="--")
                ax.set_title(f"{group_name} | {mouse_id} | {os.path.basename(s_path)}\nChannels: {sig_sub}/{ctrl_sub}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("ΔF/F (z-scored baseline)")
                fig.tight_layout()
                fig.savefig(session_plot_path)
                plt.close(fig)
                if verbose:
                    print(f"Saved: {session_plot_path}")

                # Save metrics
                session_df = pd.DataFrame(session_metric_rows)
                session_df.to_csv(session_metrics_path, index=False)
                if verbose:
                    print(f"Saved: {session_metrics_path}")

                # Pool trials
                for tr in peri_traces:
                    per_mouse_pooled_trials[mouse_id].append(tr)
                    per_mouse_pooled_trials_by_group[group_name][mouse_id].append(tr)

                per_mouse_session_counts[mouse_id] += 1
                per_mouse_summary_metrics[mouse_id].extend(session_metric_rows)
                per_mouse_summary_metrics_by_group[group_name][mouse_id].extend(session_metric_rows)

                processed_sessions.add(s_path)

    # Build group-level averages
    per_group_mouse_means = defaultdict(list)
    per_group_mouse_metrics_rows = defaultdict(list)

    for gname in (groups_dict or {}).keys():
        mice_map = per_mouse_pooled_trials_by_group.get(gname, {})
        for mouse_id, trials in mice_map.items():
            if not trials:
                continue
            trials_arr = np.vstack(trials)
            mouse_mean = np.mean(trials_arr, axis=0)
            mouse_sem = trials_arr.std(axis=0) / math.sqrt(trials_arr.shape[0])

            per_group_mouse_means[gname].append({
                "mouse": mouse_id,
                "mean": mouse_mean,
                "sem": mouse_sem
            })

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

    # Create comparison folder
    comparison_root = os.path.join(event_folder, "comparison")
    os.makedirs(comparison_root, exist_ok=True)
    summary["output_files"]["comparison_root"] = comparison_root

    # Group plots
    group_means_for_comparison = {}
    group_color_cycle = itertools.cycle([
        "red", "green", "purple", "orange", "brown", "pink", "gray"
    ])

    for gname, mouse_infos in per_group_mouse_means.items():
        gfolder = os.path.join(groups_root, _safe_name(gname))
        os.makedirs(gfolder, exist_ok=True)
        
        if len(mouse_infos) == 0:
            print(f"WARNING: no mouse means for group {gname}")
            continue

        arr = np.vstack([mi["mean"] for mi in mouse_infos])
        g_mean = np.mean(arr, axis=0)
        g_sem = arr.std(axis=0) / math.sqrt(arr.shape[0])
        group_means_for_comparison[gname] = (g_mean, g_sem)

        group_color = next(group_color_cycle)

        # Group mean plot
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
        group_mean_plot = os.path.join(gfolder, f"{_safe_name(gname)}_mean_trace.png")
        fig.savefig(group_mean_plot)
        plt.close(fig)
        if verbose:
            print(f"Saved: {group_mean_plot}")

        # All mice overlay
        fig, ax = plt.subplots(figsize=(8,5))
        for mi in mouse_infos:
            ax.plot(peri_times, mi["mean"], label=mi["mouse"], alpha=0.8)
        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔF/F (z-scored baseline)")
        ax.set_title(f"All mice: {gname}")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        group_all_mice_plot = os.path.join(gfolder, f"{_safe_name(gname)}_all_mice_traces.png")
        fig.savefig(group_all_mice_plot)
        plt.close(fig)

        # Save metrics
        gm_df = pd.DataFrame(per_group_mouse_metrics_rows.get(gname, []))
        group_summary_csv = os.path.join(gfolder, f"{_safe_name(gname)}_summary_metrics.csv")
        gm_df.to_csv(group_summary_csv, index=False)

    # Group comparison plot
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
        print(f"Saved: {combined_fig_path}")

    # Save all trial metrics
    if len(all_trial_rows) > 0:
        all_trials_df = pd.DataFrame(all_trial_rows)
        all_trials_csv = os.path.join(event_folder, "all_groups_trial_metrics.csv")
        all_trials_df.to_csv(all_trials_csv, index=False)
        summary["output_files"]["all_groups_trial_metrics_csv"] = all_trials_csv

    summary["n_mice"] = len(mice_seen)
    summary["output_files"].update({
        "event_folder": event_folder,
        "mice_root": mice_root,
        "groups_root": groups_root,
        "comparison_root": comparison_root
    })

    summary_path = os.path.join(event_folder, "summary_log.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["output_files"]["summary_log"] = summary_path
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"Total mice: {len(mice_seen)}")
        print(f"Total sessions: {summary['n_sessions']}")
        print(f"Valid trials: {summary['valid_trials']}")
        print(f"Output: {event_folder}")
        print('='*60)

    return summary
