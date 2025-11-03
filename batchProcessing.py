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
    meta_lookup = { norm(m['path']): m for m in (metadata_list or []) }

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
                meta = meta_lookup.get(norm_path)
                if meta is None:
                    err = f"metadata_missing for {s_path}"
                    print("WARNING:", err, flush=True)
                    summary["errors"].append({"session": s_path, "error": err})
                    continue

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

                scs = int(meta.get("signalChannelSet", 1))
                if scs == 1:
                    sig_sub, ctrl_sub = "_465A", "_415A"
                else:
                    sig_sub, ctrl_sub = "_465C", "_415C"

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
                try:
                    if meta.get('interpretor', 1) == 2:
                        pct_name = 'PtC1' if meta.get('signalChannelSet',1) == 1 else 'PtC2'
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
                                print(f"Alignment applied. offset={offset:.3f}s trimmed samples -> {len(sig)} remain", flush=True)
                        else:
                            if verbose:
                                print("Alignment conditions not met (onset/code12); skipping alignment", flush=True)
                except Exception as e:
                    print("WARNING: alignment failed:", e, flush=True)

                if len(event_times) == 0:
                    warn = f"no '{selected_event}' events in session {s_path}"
                    print("WARNING:", warn, flush=True)
                    summary["errors"].append({"session": s_path, "error": warn})
                    continue

                ds = int(downsample_factor)
                if ds > 1:
                    sig = sig[::ds]
                    ctrl = ctrl[::ds]
                    fs = fs_orig / ds
                else:
                    fs = fs_orig

                if verbose:
                    print(f"After downsampling: sig_len={len(sig)}, fs={fs}", flush=True)
                min_len = min(len(sig), len(ctrl))
                if len(sig) != len(ctrl):
                    if verbose:
                        print(f"Signal/control length mismatch: sig={len(sig)}, ctrl={len(ctrl)}. Truncating to {min_len}.", flush=True)
                    sig = sig[:min_len]
                    ctrl = ctrl[:min_len]

                try:
                    p = np.polyfit(ctrl, sig, 1)
                    fitted = p[0] * ctrl + p[1]
                    dff = 100.0 * (sig - fitted) / (fitted + 1e-12)
                except Exception as e:
                    err = f"regression_dff_failed: {e}"
                    print("ERROR:", err, flush=True)
                    summary["errors"].append({"session": s_path, "error": err})
                    continue

                if verbose:
                    print(f"Computed dF/F len={len(dff)} mean={np.mean(dff):.3f} std={np.std(dff):.3f}", flush=True)

                n_samples = int(round((pre_t + post_t) * fs))
                if n_samples <= 0:
                    err = f"invalid n_samples computed: {(pre_t+post_t)}*{fs} -> {n_samples}"
                    print("ERROR:", err, flush=True)
                    summary["errors"].append({"session": s_path, "error": err})
                    continue
                peri_times = (np.arange(n_samples) / fs) - pre_t

                if verbose:
                    print(f"Peri window samples: {n_samples}, peri_times[0:3]={peri_times[:3]}", flush=True)

                mouse_folder = os.path.join(mice_root, _safe_name(mouse_id))
                os.makedirs(mouse_folder, exist_ok=True)
                session_basename = _safe_name(os.path.basename(s_path))
                session_plot_path = os.path.join(mouse_folder, f"{session_basename}_session_avg.png")
                session_metrics_path = os.path.join(mouse_folder, f"{session_basename}_metrics.csv")

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

                    snippet = dff[start_idx:end_idx].astype(float)

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
                        "fs": fs
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
                session_sem = peri_arr.std(axis=0) / math.sqrt(peri_arr.shape[0])

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
    # show how many session-level trials exist per group/mouse
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
            mouse_sem = trials_arr.std(axis=0) / math.sqrt(trials_arr.shape[0])

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

    for gname, mouse_infos in per_group_mouse_means.items():
        gfolder = os.path.join(groups_root, _safe_name(gname))
        os.makedirs(gfolder, exist_ok=True)
        group_mean_plot = os.path.join(gfolder, f"{_safe_name(gname)}_mean_trace.png")
        group_all_mice_plot = os.path.join(gfolder, f"{_safe_name(gname)}_all_mice_traces.png")
        group_summary_csv = os.path.join(gfolder, f"{_safe_name(gname)}_summary_metrics.csv")

        if len(mouse_infos) == 0:
            print(f"WARNING: no mouse means for group {gname}", flush=True)
            summary["errors"].append({"group": gname, "error": "no_mouse_means"})
            continue

        arr = np.vstack([mi["mean"] for mi in mouse_infos])
        g_mean = np.mean(arr, axis=0)
        g_sem = arr.std(axis=0) / math.sqrt(arr.shape[0])
        group_means_for_comparison[gname] = (g_mean, g_sem)

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

        fig, ax = plt.subplots(figsize=(8,5))
        for mi in mouse_infos:
            ax.plot(peri_times, mi["mean"], label=mi["mouse"], alpha=0.8)
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

    # -------------------
    if len(all_trial_rows) > 0:
        all_trials_df = pd.DataFrame(all_trial_rows)
        all_trials_csv = os.path.join(event_folder, "all_groups_trial_metrics.csv")
        all_trials_df.to_csv(all_trials_csv, index=False)
        summary["output_files"]["all_groups_trial_metrics_csv"] = all_trials_csv
        if verbose:
            print(f"Saved master trial CSV -> {all_trials_csv}", flush=True)
    else:
        if verbose:
            print("No trial rows collected; master trial CSV not created.", flush=True)

    summary["n_mice"] = len(mice_seen)
    summary["n_sessions"] = summary["n_sessions"]
    summary["n_trials_total"] = summary["n_trials_total"]
    summary["valid_trials"] = summary["valid_trials"]
    summary["invalid_trials"] = summary["invalid_trials"]

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
        print(f"Saved summary_log.json -> {summary_path}", flush=True)

    return summary
