import os
import json
import math
import itertools
import traceback
from collections import defaultdict
from datetime import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# helper safe name if you already have one, otherwise use this
def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))

def run_batch_processing(
    base_path,                    # top-level path string (same as 'path' in your UI)
    selected_event,               # event name string
    pre_t, post_t,                # peri-event times (secs)
    baseline_lower, baseline_upper,
    downsample_factor,
    metric_start, metric_end,
    metadata_list,                # st.session_state["metadata"]
    groups_dict,                  # st.session_state["groups"]
    find_stream_by_substr,        # function(block, substr) -> stream_key or None
    load_tdt_block,               # function(path) -> block object
    verbose=True
):
    """
    Processes sessions into the folder layout you specified.
    Returns a dict summary that is saved to summary_log.json by the caller.
    """

    timestamp_str = dt.now().strftime("%Y%m%d_%H%M%S")
    event_folder = os.path.join(base_path, "plots", f"{_safe_name(selected_event)}_{timestamp_str}")
    mice_root = os.path.join(event_folder, "mice")
    groups_root = os.path.join(event_folder, "groups")
    os.makedirs(mice_root, exist_ok=True)
    os.makedirs(groups_root, exist_ok=True)

    # lookup metadata by normalized path
    def norm(p): return os.path.normpath(p)
    meta_lookup = { norm(m['path']): m for m in (metadata_list or []) }

    # outputs and accumulators
    all_trial_rows = []   # rows for all_groups_trial_metrics.csv
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

    # keep per-mouse pooled (z-scored per-session) trials for later mouse-level pooling
    per_mouse_pooled_trials = defaultdict(list)   # mouse_id -> list of 1D arrays (trials)
    per_mouse_session_counts = defaultdict(int)
    per_mouse_summary_metrics = defaultdict(list) # mouse -> list of dicts (trial-level metrics) for mouse_summary later

    # flags for plotting
    color_cycle = itertools.cycle(None)  # let matplotlib choose colors

    # iterate groups -> mice -> sessions
    summary["n_groups"] = len(groups_dict)
    for group_name, mice in groups_dict.items():
        groups_seen.add(group_name)
        for mouse_id, session_paths in mice.items():
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

                # log available streams
                try:
                    avail_streams = list(block.streams.keys())
                except Exception:
                    avail_streams = []
                if verbose:
                    print(f"Loaded block: {s_path}", flush=True)
                    print("Available stream keys:", avail_streams, flush=True)

                # pick signal/control substrings based on meta
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

                # load raw arrays
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

                # handle special alignment for interpretor==2
                event_times = [e['timestamp_s'] for e in meta.get("events", []) if e.get("event") == selected_event]
                try:
                    if meta.get('interpretor', 1) == 2:
                        pct_name = 'PtC1' if meta.get('signalChannelSet',1) == 1 else 'PtC2'
                        onset_list = block.epocs.get(pct_name, {}).get('onset', None) if hasattr(block, "epocs") else None
                        # fallback: block.epocs[pct_name].onset as in your original code
                        try:
                            onset_list = block.epocs[pct_name].onset
                        except Exception:
                            onset_list = onset_list
                        code12_times = [e["timestamp_s"] for e in meta.get("events", []) if e.get("code") == 12]
                        if onset_list is not None and len(onset_list) > 1 and len(code12_times) >= 2:
                            pct_onset = float(onset_list[1])
                            offset = pct_onset - code12_times[1]
                            # apply offset to time base by trimming the start of signals
                            time_full = np.arange(len(sig)) / fs_orig
                            time_full -= offset
                            valid = time_full >= 0
                            sig = sig[valid]
                            ctrl = ctrl[valid]
                            # shift event_times so they align with trimmed signals
                            event_times = [t - offset for t in event_times]
                            if verbose:
                                print(f"Alignment applied. offset={offset:.3f}s trimmed samples -> {len(sig)} remain", flush=True)
                        else:
                            if verbose:
                                print("Alignment conditions not met (onset/code12); skipping alignment", flush=True)
                except Exception as e:
                    print("WARNING: alignment failed:", e, flush=True)
                    # continue using unaligned data

                if len(event_times) == 0:
                    warn = f"no '{selected_event}' events in session {s_path}"
                    print("WARNING:", warn, flush=True)
                    summary["errors"].append({"session": s_path, "error": warn})
                    continue

                # Downsample
                ds = int(downsample_factor)
                if ds > 1:
                    sig = sig[::ds]
                    ctrl = ctrl[::ds]
                    fs = fs_orig / ds
                else:
                    fs = fs_orig

                if verbose:
                    print(f"After downsampling: sig_len={len(sig)}, fs={fs}", flush=True)

                # compute dF/F via linear regression ctrl -> sig
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

                # compute n_samples and peri times so indexing is consistent
                n_samples = int(round((pre_t + post_t) * fs))
                if n_samples <= 0:
                    err = f"invalid n_samples computed: {(pre_t+post_t)}*{fs} -> {n_samples}"
                    print("ERROR:", err, flush=True)
                    summary["errors"].append({"session": s_path, "error": err})
                    continue
                peri_times = (np.arange(n_samples) / fs) - pre_t

                if verbose:
                    print(f"Peri window samples: {n_samples}, peri_times[0:3]={peri_times[:3]}", flush=True)

                # storage per session/mouse
                mouse_folder = os.path.join(mice_root, _safe_name(mouse_id))
                os.makedirs(mouse_folder, exist_ok=True)
                session_basename = _safe_name(os.path.basename(s_path))
                session_plot_path = os.path.join(mouse_folder, f"{session_basename}_session_avg.png")
                session_metrics_path = os.path.join(mouse_folder, f"{session_basename}_metrics.csv")

                # per-session trial traces and metrics
                peri_traces = []
                session_metric_rows = []
                trial_index = 0

                # iterate event times and extract snippets
                for ts in event_times:
                    # compute global start index for the snippet in dff
                    start_idx = int(round((ts - pre_t) * fs))
                    end_idx = start_idx + n_samples
                    trial_index += 1
                    # out of bounds -> skip
                    if start_idx < 0 or end_idx > len(dff):
                        if verbose:
                            print(f"  Skip trial {trial_index}: start={start_idx} end={end_idx} out of bounds (len={len(dff)})", flush=True)
                        summary["invalid_trials"] += 1
                        summary["n_trials_total"] += 1
                        continue

                    snippet = dff[start_idx:end_idx].astype(float)  # length should be n_samples

                    # baseline relative to snippet
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
                        # fallback: use pre-event portion of snippet if available
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
                            # last resort: subtract global mean of dff
                            mu = float(np.mean(dff))
                            snippet_z = snippet - mu
                            zscored_flag = False

                    # compute metrics within metric window relative indices on snippet
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

                    # store trial
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

                    # global master list
                    all_trial_rows.append(session_metric_rows[-1])
                    summary["n_trials_total"] += 1
                    summary["valid_trials"] += 1

                # finish session processing
                if len(peri_traces) == 0:
                    print(f"WARNING: no valid peri-event snippets for session {s_path}", flush=True)
                    summary["errors"].append({"session": s_path, "error": "no_valid_trials"})
                    continue

                # compute session mean and SEM
                peri_arr = np.vstack(peri_traces)  # trials x samples
                session_mean = np.mean(peri_arr, axis=0)
                session_sem = peri_arr.std(axis=0) / math.sqrt(peri_arr.shape[0])

                # save session plot
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

                # write session trial-by-trial CSV
                session_df = pd.DataFrame(session_metric_rows)
                session_df.to_csv(session_metrics_path, index=False)
                if verbose:
                    print(f"Saved session metrics -> {session_metrics_path}", flush=True)

                # append session-level traces to per-mouse pooled list (z-scored trials used)
                for tr in peri_traces:
                    per_mouse_pooled_trials[mouse_id].append(tr)
                per_mouse_session_counts[mouse_id] += 1
                per_mouse_summary_metrics[mouse_id].extend(session_metric_rows)

                processed_sessions.add(s_path)

    # -------------------
    # per-mouse aggregation (z-scored trials already pooled per mouse)
    mice_count = len(per_mouse_pooled_trials)
    summary["n_mice"] = len(mice_seen)
    if verbose:
        print(f"\n--- Per-mouse pooling for {mice_count} mice ---", flush=True)

    per_group_mouse_means = defaultdict(list)   # group -> list of mouse mean traces
    per_group_mouse_metrics_rows = defaultdict(list)  # group-> rows for group_summary_metrics.csv
    for mouse_id, trial_list in per_mouse_pooled_trials.items():
        if len(trial_list) == 0:
            continue
        # stack trials (all already z-scored per-session)
        trials_arr = np.vstack(trial_list)  # trials x samples
        mouse_mean = np.mean(trials_arr, axis=0)
        mouse_sem = trials_arr.std(axis=0) / math.sqrt(trials_arr.shape[0])

        # save mouse-level plots and metrics under mice/<mouse>/
        mouse_folder = os.path.join(mice_root, _safe_name(mouse_id))
        os.makedirs(mouse_folder, exist_ok=True)
        mouse_combined_plot = os.path.join(mouse_folder, f"{_safe_name(mouse_id)}_combined_trace.png")
        mouse_combined_metrics_csv = os.path.join(mouse_folder, f"{_safe_name(mouse_id)}_combined_metrics.csv")
        mouse_summary_metrics_csv = os.path.join(mouse_folder, f"{_safe_name(mouse_id)}_summary_metrics.csv")

        # timepoint CSV: time, mean, sem
        timepoint_df = pd.DataFrame({
            "time_s": peri_times,
            "mean": mouse_mean,
            "sem": mouse_sem
        })
        timepoint_df.to_csv(mouse_combined_metrics_csv, index=False)

        # summary scalar metrics (mean across all trials for this mouse)
        mouse_metric_rows = per_mouse_summary_metrics.get(mouse_id, [])
        if len(mouse_metric_rows) > 0:
            mdf = pd.DataFrame(mouse_metric_rows)
            # compute per-mouse mean peak/auc/latency (ignoring NaNs)
            mean_peak = float(mdf["peak"].mean(skipna=True))
            mean_auc = float(mdf["auc"].mean(skipna=True))
            mean_latency = float(mdf["latency"].mean(skipna=True))
            n_trials = int(len(mdf))
        else:
            mean_peak = mean_auc = mean_latency = np.nan
            n_trials = 0

        pd.DataFrame([{
            "mouse": mouse_id,
            "mean_peak": mean_peak,
            "mean_auc": mean_auc,
            "mean_latency": mean_latency,
            "n_trials": n_trials
        }]).to_csv(mouse_summary_metrics_csv, index=False)

        # save mouse combined trace plot
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(peri_times, mouse_mean)
        ax.fill_between(peri_times, mouse_mean - mouse_sem, mouse_mean + mouse_sem, alpha=0.3)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title(f"Mouse combined: {mouse_id}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔF/F (z-scored baseline)")
        fig.tight_layout()
        fig.savefig(mouse_combined_plot)
        plt.close(fig)
        if verbose:
            print(f"Saved mouse combined plot -> {mouse_combined_plot}", flush=True)

        # find which group(s) this mouse belongs to so we can add to per-group aggregate
        # groups_dict structure: group -> {mouse: [sessionpaths]}
        for gname, mice in groups_dict.items():
            if mouse_id in mice:
                per_group_mouse_means[gname].append(mouse_mean)
                # create per-mouse row for group summary metrics
                per_group_mouse_metrics_rows[gname].append({
                    "group": gname,
                    "mouse": mouse_id,
                    "mean_peak": mean_peak,
                    "mean_auc": mean_auc,
                    "mean_latency": mean_latency,
                    "n_trials": n_trials
                })

    # -------------------
    # group-level aggregates and plots
    if verbose:
        print("\n--- Group-level aggregation ---", flush=True)

    for gname, mouse_means in per_group_mouse_means.items():
        gfolder = os.path.join(groups_root, _safe_name(gname))
        os.makedirs(gfolder, exist_ok=True)
        group_mean_plot = os.path.join(gfolder, f"{_safe_name(gname)}_mean_trace.png")
        group_all_mice_plot = os.path.join(gfolder, f"{_safe_name(gname)}_all_mice_traces.png")
        group_summary_csv = os.path.join(gfolder, f"{_safe_name(gname)}_summary_metrics.csv")

        if len(mouse_means) == 0:
            print(f"WARNING: no mouse means for group {gname}", flush=True)
            summary["errors"].append({"group": gname, "error": "no_mouse_means"})
            continue

        arr = np.vstack(mouse_means)  # mice x samples
        g_mean = np.mean(arr, axis=0)
        g_sem = arr.std(axis=0) / math.sqrt(arr.shape[0])

        # save group mean trace
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(peri_times, g_mean, label=gname)
        ax.fill_between(peri_times, g_mean - g_sem, g_mean + g_sem, alpha=0.3)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title(f"Group avg: {gname}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔF/F (z-scored baseline)")
        fig.tight_layout()
        fig.savefig(group_mean_plot)
        plt.close(fig)
        if verbose:
            print(f"Saved group mean plot -> {group_mean_plot}", flush=True)

        # save overlay of each mouse mean
        fig, ax = plt.subplots(figsize=(8,5))
        for mm in mouse_means:
            ax.plot(peri_times, mm, alpha=0.6)
        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("ΔF/F (z-scored baseline)")
        ax.set_title(f"All mice traces: {gname}")
        fig.tight_layout()
        fig.savefig(group_all_mice_plot)
        plt.close(fig)
        if verbose:
            print(f"Saved group all-mice overlay -> {group_all_mice_plot}", flush=True)

        # group summary CSV listing each mouse's mean metrics
        gm_df = pd.DataFrame(per_group_mouse_metrics_rows.get(gname, []))
        gm_df.to_csv(group_summary_csv, index=False)
        if verbose:
            print(f"Saved group summary metrics -> {group_summary_csv}", flush=True)

    # -------------------
    # final group comparison across groups: overlay all group means
    combined_fig_path = os.path.join(event_folder, f"{_safe_name(selected_event)}_group_comparison.png")
    fig, ax = plt.subplots(figsize=(10,6))
    for gname, mouse_means in per_group_mouse_means.items():
        if len(mouse_means) == 0:
            continue
        arr = np.vstack(mouse_means)
        g_mean = np.mean(arr, axis=0)
        ax.plot(peri_times, g_mean, label=gname)
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ΔF/F (z-scored baseline)")
    ax.set_title(f"Group comparison — Event: {selected_event}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(combined_fig_path)
    plt.close(fig)
    summary["output_files"]["group_comparison_plot"] = combined_fig_path
    if verbose:
        print(f"Saved group comparison plot -> {combined_fig_path}", flush=True)

    # -------------------
    # write all_groups_trial_metrics.csv (master flattened trial table)
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

    # collect numeric summaries
    summary["n_mice"] = len(mice_seen)
    summary["n_sessions"] = summary["n_sessions"]
    summary["n_trials_total"] = summary["n_trials_total"]
    summary["valid_trials"] = summary["valid_trials"]
    summary["invalid_trials"] = summary["invalid_trials"]

    # include file paths for convenience
    summary["output_files"].update({
        "event_folder": event_folder,
        "mice_root": mice_root,
        "groups_root": groups_root
    })

    # write summary_log.json
    summary_path = os.path.join(event_folder, "summary_log.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["output_files"]["summary_log"] = summary_path
    if verbose:
        print(f"Saved summary_log.json -> {summary_path}", flush=True)

    return summary
