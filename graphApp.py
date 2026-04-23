import streamlit as st
import numpy as np
import tempfile
import os
import io
import plotly.graph_objs as go
import tdt
import hashlib
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import json
import matplotlib

from OHRBETsEventExtractor import parse_ohrbets_serial_log
from nxtEventExtrator import readfile, get_events, get_event_codes
from sessionInfoExtractor import *
from splice_utils import load_splices, save_splices, get_splice_mask, check_snippet_overlaps_splice, offset_splices
from batchProcessing import run_batch_processing
from advanced_graphing import (
    run_advanced_graphing, discover_sessions,
    discover_sessions_recursive, run_advanced_graphing_from_discovered,
    discover_event_folders, discover_prism_sessions,
    discover_long_trace_sessions,
)

matplotlib.use("Agg")

color_cycle = itertools.cycle([
    "red", "green", "purple", "orange", "brown", "pink", "gray"
])

import re
from datetime import datetime as dt
from collections import defaultdict

def _safe_name(s):
    return re.sub(r'[^\w\-_. ]', '_', str(s))

def find_stream_by_substr(block, substr):
    # case-insensitive substring search in stream keys
    for k in block.streams.keys():
        if substr.lower() in k.lower():
            return k
    return None

# Set up Streamlit
st.set_page_config(layout="wide")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Extractor",
    "Graph Viewer", 
    "Peri Event Plots",
    "Average Sessions",
    "Advanced Graphing"
])

if "code_map" not in st.session_state:
    st.session_state.code_map = code_map = {}

nxt_codes = []

if "tdt_settings" not in st.session_state:
    st.session_state.tdt_settings = {
        "data_folder": "",
        "block_folder": "",
        "signal_stream": None,
        "control_stream": None,
        "signal2_stream": None,
        "control2_stream": None,
        "align_stream": None,
        "extract_pct": False,
        "pct_channel": None
    }

@st.cache_resource
def load_tdt_block(path):
    return tdt.read_block(path)

def get_file_hash(file):
    file.seek(0)
    data = file.read()
    file.seek(0)
    return hashlib.md5(data if isinstance(data, (bytes, bytearray)) else data.encode()).hexdigest()

st.markdown("""
    <style>
    .credit-footer {
        position: fixed;
        bottom: 8px;
        right: 15px;
        font-size: 0.75rem;
        color: gray;
        opacity: 0.6;
        z-index: 100;
    }
    </style>
    <div class="credit-footer">
        Developed for Parekh Lab © 2025
            Aryan Bangad
    </div>
""", unsafe_allow_html=True)


with tab3:
    st.title("🧪 Peri-Event Plot Viewer")

    if st.button("🔄 Refresh"):
        st.rerun()

    if "extracted_data" not in st.session_state or "events" not in st.session_state["extracted_data"]:
        st.warning("❗ Please extract data and events first from the Data Extractor tab.")
    else:
        # Settings
        event_name = st.selectbox(
            "Select event code for alignment:",
            sorted(set(e["event"] for e in st.session_state["extracted_data"]["events"] if "event" in e))
        )
        signal_set = st.radio("Select signal/control pair:", ["1", "2"], horizontal=True)
        PRE_TIME = st.number_input("Seconds before event", value=5.0)
        POST_TIME = st.number_input("Seconds after event", value=10.0)
        baseline_mode = st.checkbox("Pre-event baseline?", value=False)
        if baseline_mode:
            lowerBound = st.number_input("Enter the lower bound of baseline period from T0 (s)", value=-4.0)
            upperBound = st.number_input("Enter the upper bound of baseline period from T0 (s)", value=-1.0)
        else:
            lowerBound, upperBound = 0, 0
        downsample_factor = st.number_input("Downsample factor", min_value=1, value=1, step=1)

        # Load data
        signal = st.session_state.extracted_data.get(f"signal{signal_set}")
        control = st.session_state.extracted_data.get(f"control{signal_set}")
        fs = st.session_state.extracted_data.get("fs")

        if signal is None or control is None or fs is None:
            st.error("Signal or control data missing.")
        else:
            # --- Optional PCT alignment ---
            apply_pct_alignment = st.checkbox("Align signal to PCT onset (code 12)", value=False, key="peri_pct_align")
            if apply_pct_alignment:
                peri_pct_choice = st.selectbox(
                    "Which PCT epoch onset to use for alignment?",
                    ["1st PCT onset", "2nd PCT onset"],
                    index=1,
                    key="peri_pct_choice"
                )
                peri_pct_idx = 0 if peri_pct_choice == "1st PCT onset" else 1
                try:
                    pct = st.session_state.extracted_data.get("pct", [])
                    events = st.session_state.extracted_data.get("events", [])
                    required_pct = peri_pct_idx + 1
                    if len(pct) < required_pct:
                        st.warning(f"Less than {required_pct} PCT onset(s) found — skipping alignment.")
                    else:
                        pct_onset = float(pct[peri_pct_idx])
                        code12_times = [e["timestamp_s"] for e in events if e.get("code") == 12]
                        if len(code12_times) < 2:
                            st.warning("Less than 2 code-12 events found — skipping alignment.")
                        else:
                            offset = pct_onset - code12_times[1]
                            time_full = np.arange(len(signal)) / fs
                            time_full -= offset
                            valid = time_full >= 0
                            signal = signal[valid]
                            control = control[valid]
                            st.info(f"Applied PCT alignment using {peri_pct_choice}, offset = {offset:.3f} s")
                except Exception as e:
                    st.warning(f"Alignment failed: {e}")
            # Downsample
            if downsample_factor > 1:
                signal = signal[::downsample_factor]
                control = control[::downsample_factor]
                fs = fs / downsample_factor

            min_len = min(len(signal), len(control))
            if len(signal) != len(control):
                signal = signal[:min_len]
                control = control[:min_len]

            # --- Load splices for regression masking ---
            _t3_block = st.session_state.tdt_settings.get("block_folder", "")
            _t3_channel = int(signal_set)
            _t3_raw_splices = []
            if _t3_block and os.path.isdir(_t3_block):
                _t3_raw_splices = load_splices(_t3_block, channel=_t3_channel)
            # Adjust splice times for any PCT offset applied above
            try:
                _t3_offset = offset  # set if PCT alignment ran
            except NameError:
                _t3_offset = 0.0
            _t3_splices = offset_splices(_t3_raw_splices, _t3_offset) if _t3_raw_splices else []

            # ΔF/F calculation (exclude spliced regions from polyfit)
            if _t3_splices:
                _t3_mask = get_splice_mask(min_len, fs, _t3_splices)
                fit = np.polyfit(control[_t3_mask], signal[_t3_mask], 1)
            else:
                fit = np.polyfit(control, signal, 1)
            fitted = fit[0] * control + fit[1]
            dFF = 100 * (signal - fitted) / fitted

            TRANGE = [int(-PRE_TIME * fs), int(POST_TIME * fs)]
            snippets = []

            for event in st.session_state.extracted_data["events"]:
                if event.get("event") != event_name:
                    continue
                on = event.get("timestamp_s")
                if on is None or on * fs < -TRANGE[0] or (on * fs + TRANGE[1]) >= len(dFF):
                    continue

                # Skip events whose peri-event window overlaps a splice
                if _t3_splices and check_snippet_overlaps_splice(on, PRE_TIME, POST_TIME, _t3_splices):
                    continue

                idx = int(on * fs)

                # peri-event snippet
                snippet = dFF[idx + TRANGE[0]: idx + TRANGE[1]]
                if len(snippet) != (TRANGE[1] - TRANGE[0]):
                    continue

                if baseline_mode:
                    # ----------------------------
                    # Per-event baseline z-scoring
                    # ----------------------------
                    baseline_start = int((on + lowerBound) * fs)
                    baseline_end   = int((on + upperBound) * fs)

                    # Clip to global dFF bounds
                    baseline_start = max(0, baseline_start)
                    baseline_end   = min(len(dFF), baseline_end)

                    if baseline_end > baseline_start:
                        baseline_vals = dFF[baseline_start:baseline_end]
                        baseline_mean = np.mean(baseline_vals)
                        baseline_std  = np.std(baseline_vals)

                        if baseline_std > 0:
                            snippet = (snippet - baseline_mean) / baseline_std
                        else:
                            snippet = snippet - baseline_mean
                else:
                    # ----------------------------
                    # Global z-scoring (whole trace, excluding spliced regions)
                    # ----------------------------
                    if _t3_splices:
                        _t3_clean_dFF = dFF[_t3_mask]
                    else:
                        _t3_clean_dFF = dFF
                    trace_mean = np.mean(_t3_clean_dFF)
                    trace_std = np.std(_t3_clean_dFF)
                    snippet = (snippet - trace_mean) / (trace_std if trace_std > 0 else 1.0)

                snippets.append(snippet)

            if not snippets:
                st.warning("No valid event-aligned snippets found.")
            else:
                snippets = np.array(snippets)
                mean_snip = np.mean(snippets, axis=0)
                std_snip = np.std(snippets, axis=0)

                # peri-event time vector in seconds
                peri_time = np.arange(TRANGE[0], TRANGE[1]) / fs

                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)

                # Plot individual traces
                for snip in snippets:
                    ax.plot(peri_time, snip, linewidth=0.5, color='gray', alpha=0.5)

                # Mean ± STD
                ax.plot(peri_time, mean_snip, color='green', linewidth=2, label='Mean ΔF/F')
                ax.fill_between(peri_time, mean_snip - std_snip, mean_snip + std_snip,
                                color='green', alpha=0.2, label='±STD')

                # Event onset marker
                ax.axvline(0, color='slategray', linestyle='--', linewidth=2, label=f'{event_name} Onset')

                # Labels
                if baseline_mode:
                    title_str = f"Peri-Event ΔF/F (Baseline Z-score: {lowerBound}s to {upperBound}s)"
                    ylabel_str = "ΔF/F (Baseline Z-score)"
                else:
                    title_str = "Peri-Event ΔF/F (Global Z-score)"
                    ylabel_str = "ΔF/F (Global Z-score)"

                ax.set_title(title_str)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(ylabel_str)
                ax.legend()
                ax.grid(True)

                st.pyplot(fig, width='stretch')

with tab1:
    st.title("📂 TDT + Events Data Extractor")
    base_path = st.text_input("Base folder path", value=st.session_state.tdt_settings["data_folder"] or ".")
    if os.path.isdir(base_path):
        folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and not f.startswith(".")])
        folder = st.selectbox("Available TDT Block Folders", folders, index=0 if not st.session_state.tdt_settings["block_folder"] else folders.index(os.path.basename(st.session_state.tdt_settings["block_folder"])) if os.path.basename(st.session_state.tdt_settings["block_folder"]) in folders else 0)

        full_path = os.path.join(base_path, folder)
        st.session_state.tdt_settings["data_folder"] = base_path
        st.session_state.tdt_settings["block_folder"] = full_path

        try:
            data = load_tdt_block(full_path)
            streams = list(data.streams.keys())
            pcts = list(data.epocs.keys())

            st.markdown("---")
            st.subheader("🔌 Select Signal/Control Streams")
            col1, col2 = st.columns(2)

            with col1:
                st.selectbox("Signal", [None] + streams, index=streams.index(st.session_state.tdt_settings["signal_stream"]) + 1 if st.session_state.tdt_settings["signal_stream"] in streams else 0, key="signal_stream")
                st.selectbox("Signal 2 (optional)", [None] + streams, index=streams.index(st.session_state.tdt_settings["signal2_stream"]) + 1 if st.session_state.tdt_settings["signal2_stream"] in streams else 0, key="signal2_stream")
            with col2:
                st.selectbox("Control", [None] + streams, index=streams.index(st.session_state.tdt_settings["control_stream"]) + 1 if st.session_state.tdt_settings["control_stream"] in streams else 0, key="control_stream")
                st.selectbox("Control 2 (optional)", [None] + streams, index=streams.index(st.session_state.tdt_settings["control2_stream"]) + 1 if st.session_state.tdt_settings["control2_stream"] in streams else 0, key="control2_stream")

            pctBool = st.checkbox("Also extract OHRBETS PCT data for events extraction?", key="extract_pct")
            if pctBool:
                st.selectbox("PtC channels",[None] + pcts, index=pcts.index(st.session_state.tdt_settings["pct_channel"]) + 1 if st.session_state.tdt_settings["pct_channel"] in pcts else 0, key="pct_channel")
            else:
                st.session_state.pct_channel = None

            if st.button("📥 Extract Data"):
                tdt_data = {
                "signal1": data.streams[st.session_state.signal_stream].data if st.session_state.signal_stream else None,
                "control1": data.streams[st.session_state.control_stream].data if st.session_state.control_stream else None,
                "signal2": data.streams[st.session_state.signal2_stream].data if st.session_state.signal2_stream else None,
                "control2": data.streams[st.session_state.control2_stream].data if st.session_state.control2_stream else None,
                "pct": (data.epocs[st.session_state.pct_channel].onset if st.session_state.pct_channel else None ) if pctBool else None,
                "fs": data.streams[st.session_state.signal_stream].fs if st.session_state.signal_stream else None,
                }

                st.session_state["extracted_data"] = tdt_data
                st.success("✅ TDT Data extracted and stored.")
        except Exception as e:
            st.error(f"Error reading TDT folder: {e}")  
    else:
        st.warning("⚠️ Please enter a valid folder path.")


    st.markdown("---")
    st.subheader("📎 Upload Events")

    log_fileCSV = st.file_uploader("Upload OHRBETS event log (`.csv`)", type=["csv"])
    log_fileNXT = st.file_uploader("Upload OpenScope / Synapse event log (`.nxt`)", type=["nxt"])

    if log_fileNXT is not None:
        new_file_hash = get_file_hash(log_fileNXT)

        if st.session_state.get("last_uploaded_nxt_hash") != new_file_hash:
        # New file uploaded — reset state
            st.session_state["last_uploaded_nxt_hash"] = new_file_hash

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nxt", mode='wb') as tmp:
                    tmp.write(log_fileNXT.read())
                    tmp_path = tmp.name

                events_dataNXT = readfile(tmp_path)
                nxt_codes = get_event_codes(events_dataNXT)

            # Update session state
                st.session_state["nxt_data"] = events_dataNXT
                st.session_state["nxt_codes"] = nxt_codes
                st.session_state["code_map"] = {code: "" for code in nxt_codes}

                st.info("New `.nxt` file detected — Code map and event codes reset.")
            finally:
                os.remove(tmp_path)

# Extract Events Button
    if st.button("📤 Extract Events"):
        all_events = []

        if log_fileCSV:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='wb') as tmp:
                    tmp.write(log_fileCSV.read())
                    tmp_path = tmp.name
                parsed = parse_ohrbets_serial_log(tmp_path)
                all_events.extend(parsed)
            finally:
                os.remove(tmp_path)

        if "nxt_data" in st.session_state and "code_map" in st.session_state:
            parsed = get_events(st.session_state["nxt_data"], st.session_state["code_map"])
            all_events.extend(parsed)

        if "extracted_data" not in st.session_state:
            st.session_state["extracted_data"] = {}

        st.session_state["extracted_data"]["events"] = all_events
        st.success(f"✅ Loaded {len(all_events)} events.")

    if log_fileNXT:
        st.markdown("---")
        st.subheader("📝 Code Map Editor (for `.nxt` event decoding)")

        with st.form("edit_code_map_form"):
            updated_map = {}
            for code in sorted(st.session_state["nxt_codes"]):
                default_label = st.session_state.code_map.get(code, "")
                label = st.text_input(f"Label for Code {code}", value=default_label, key=f"code_{code}")
                updated_map[code] = label

            submitted = st.form_submit_button("✅ Save Code Map")
            if submitted:
                st.session_state.code_map = updated_map
                st.success("Code map updated.")
    if st.button("lick filtering"):
        st.session_state["extracted_data"]["events"] = lickBoutFilter(st.session_state["extracted_data"]["events"])
        st.success("Lick bouts are implemented.")        

# =====================
# TAB 2: Graph Viewer
# =====================
with tab2:
    st.title("Fiber Photometry Grapher (Interactive)")

    downsample_factor = st.number_input("Downsampling factor", min_value=1, value=100)
    cutoff_time_from_pct = st.checkbox("Use PCT event (code 12) to align all events?", value=False)
    if cutoff_time_from_pct:
        graph_pct_choice = st.selectbox(
            "Which PCT epoch onset to use for alignment?",
            ["1st PCT onset", "2nd PCT onset"],
            index=1,
            key="graph_pct_choice"
        )
        graph_pct_idx = 0 if graph_pct_choice == "1st PCT onset" else 1
        cutoff_time = 0
    else:
        graph_pct_idx = 1  # default, unused when PCT alignment is off
        cutoff_time = st.number_input("Start time (seconds)", min_value=0.0, value=2.0)

    plot_choice = st.radio("Plot Type", ["ΔF/F", "Z-scored ΔF/F"])

    enabled_events = {}

    if "extracted_data" in st.session_state and "events" in st.session_state.extracted_data:
        all_codes = sorted(set(event["code"] for event in st.session_state.extracted_data["events"] if "code" in event))
        all_names = sorted(set(event["event_name"] for event in st.session_state.extracted_data["events"] if "event_name" in event))
        st.markdown("**Select which event codes to overlay:**")
        for code in all_codes:
            label = st.session_state.code_map.get(code, f"Code {code}")
            enabled = st.checkbox(f"Show: {label}", value=True, key=f"show_event_{code}")
            enabled_events[code] = enabled
        
        code_colors = {}
        for code in sorted(enabled_events.keys()):
            if enabled_events[code]:
                code_colors[code] = next(color_cycle)

    show_second_channel = False
    if "signal2" in st.session_state.get("extracted_data", {}) and st.session_state.extracted_data["signal2"] is not None:
        show_second_channel = st.checkbox("Also graph Signal 2 and Control 2", value=True)

    # =====================
    # Signal Splicing UI
    # =====================
    st.markdown("---")
    st.subheader("✂️ Signal Splicing")
    st.caption(
        "Define time regions to cut from the signal. Spliced regions are "
        "excluded from regression and analysis. They appear as red shading "
        "on the graph. Times are in the graph's displayed coordinate."
    )

    _splice_block_path = st.session_state.tdt_settings.get("block_folder", "")

    if _splice_block_path and os.path.isdir(_splice_block_path):
        # --- Channel selector ---
        _has_ch2 = show_second_channel
        _avail_channels = ["Channel 1", "Channel 2"] if _has_ch2 else ["Channel 1"]
        _splice_ch_label = st.radio("Edit splices for:", _avail_channels, horizontal=True, key="splice_ch_radio")
        _splice_ch = 1 if _splice_ch_label == "Channel 1" else 2
        _sk1 = "splices_ch1"
        _sk2 = "splices_ch2"

        # Auto-load splices per channel when block folder changes
        if st.session_state.get("_splice_block") != _splice_block_path:
            st.session_state[_sk1] = load_splices(_splice_block_path, channel=1)
            st.session_state[_sk2] = load_splices(_splice_block_path, channel=2)
            st.session_state["_splice_block"] = _splice_block_path
        # Ensure keys exist
        if _sk1 not in st.session_state:
            st.session_state[_sk1] = load_splices(_splice_block_path, channel=1)
        if _sk2 not in st.session_state:
            st.session_state[_sk2] = load_splices(_splice_block_path, channel=2)

        _active_sk = _sk1 if _splice_ch == 1 else _sk2

        # Compute display offset early so splice UI can reference it
        _splice_offset = cutoff_time
        if "extracted_data" in st.session_state:
            try:
                _d = st.session_state.extracted_data
                _splice_offset = _d["pct"][1] - [
                    e["timestamp_s"] for e in _d.get("events", []) if e.get("code") == 12
                ][1]
            except Exception:
                _splice_offset = cutoff_time

        col_ss, col_se = st.columns(2)
        with col_ss:
            splice_start_input = st.number_input(
                "Splice start (s, graph time)", min_value=0.0, value=0.0,
                step=0.1, format="%.2f", key="splice_start_input"
            )
        with col_se:
            splice_end_input = st.number_input(
                "Splice end (s, graph time)", min_value=0.0, value=1.0,
                step=0.1, format="%.2f", key="splice_end_input"
            )

        col_add, col_save, col_clear = st.columns(3)
        with col_add:
            if st.button("➕ Add Splice"):
                if splice_end_input > splice_start_input:
                    raw_start = splice_start_input + _splice_offset
                    raw_end = splice_end_input + _splice_offset
                    st.session_state[_active_sk].append((raw_start, raw_end))
                    st.session_state[_active_sk].sort(key=lambda x: x[0])
                    save_splices(_splice_block_path, st.session_state[_active_sk], channel=_splice_ch)
                    st.success(f"Added splice to {_splice_ch_label}: {splice_start_input:.2f}s → {splice_end_input:.2f}s")
                    st.rerun()
                else:
                    st.error("Splice end must be > splice start.")
        with col_save:
            if st.button("💾 Save Splices"):
                save_splices(_splice_block_path, st.session_state.get(_active_sk, []), channel=_splice_ch)
                st.success(f"Saved {len(st.session_state.get(_active_sk, []))} splice(s) for {_splice_ch_label}.")
        with col_clear:
            if st.button("🗑️ Clear All Splices"):
                st.session_state[_active_sk] = []
                save_splices(_splice_block_path, [], channel=_splice_ch)
                st.rerun()

        # Display existing splices for the active channel
        current_splices = st.session_state.get(_active_sk, [])
        if current_splices:
            st.markdown(f"**Current Splices ({_splice_ch_label}):**")
            for i, (rs, re_) in enumerate(current_splices):
                disp_s = rs - _splice_offset
                disp_e = re_ - _splice_offset
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.text(f"  #{i+1}: {disp_s:.2f}s → {disp_e:.2f}s  (duration: {re_ - rs:.2f}s)")
                with c2:
                    if st.button("🗑️", key=f"del_splice_{_splice_ch}_{i}"):
                        st.session_state[_active_sk].pop(i)
                        save_splices(_splice_block_path, st.session_state[_active_sk], channel=_splice_ch)
                        st.rerun()
    else:
        st.info("Load a TDT block in the Data Extractor tab to manage splices.")

    st.markdown("---")

    if st.button("Generate Plot"):
        if "extracted_data" not in st.session_state:
            st.error("❌ No TDT data loaded. Please use the Data Extractor tab.")
            st.stop()
        data = st.session_state.extracted_data
        try:
            pct_onset = data["pct"][graph_pct_idx]
            code12_times = [e["timestamp_s"] for e in data["events"] if e["code"] == 12]
            offset = pct_onset - code12_times[1]
        except:
            offset=cutoff_time

        # Load per-channel splices
        _block_path = st.session_state.tdt_settings.get("block_folder", "")
        raw_splices_ch1, raw_splices_ch2 = [], []
        if _block_path and os.path.isdir(_block_path):
            raw_splices_ch1 = load_splices(_block_path, channel=1)
            raw_splices_ch2 = load_splices(_block_path, channel=2)
        display_splices_ch1 = offset_splices(raw_splices_ch1, offset)
        display_splices_ch2 = offset_splices(raw_splices_ch2, offset)

        # --- Compute traces and store in session_state ---
        trace_results = {}  # label -> {time, dF, dF_z}

        def compute_trace(signal, control, label_prefix, ch_display_splices):
            # Pre-downsample length clamp (YARA indexing fix)
            min_len = min(len(signal), len(control))
            signal = signal[:min_len]
            control = control[:min_len]

            signal_ds = np.array([np.mean(signal[i_:i_+downsample_factor]) for i_ in range(0, len(signal), downsample_factor)])
            control_ds = np.array([np.mean(control[i_:i_+downsample_factor]) for i_ in range(0, len(control), downsample_factor)])
            time = np.arange(len(signal_ds)) / data["fs"] * downsample_factor

            time -= offset
            valid = time >= 0
            # 4-way guard to prevent index mismatch (YARA indexing fix)
            min_len_ds = min(len(signal_ds), len(control_ds), len(time), len(valid))
            signal_ds = signal_ds[:min_len_ds]
            control_ds = control_ds[:min_len_ds]
            time = time[:min_len_ds]
            valid = valid[:min_len_ds]

            time = time[valid]
            signal_ds = signal_ds[valid]
            control_ds = control_ds[valid]

            min_len_ds = min(len(signal_ds), len(control_ds))
            signal_ds = signal_ds[:min_len_ds]
            control_ds = control_ds[:min_len_ds]
            time = time[:min_len_ds]

            # Use splice mask to exclude spliced samples from polyfit
            if ch_display_splices:
                ds_fs = data["fs"] / downsample_factor
                smask = get_splice_mask(len(signal_ds), ds_fs, ch_display_splices)
                fit = np.polyfit(control_ds[smask], signal_ds[smask], 1)
            else:
                fit = np.polyfit(control_ds, signal_ds, 1)

            fitted = fit[0] * control_ds + fit[1]
            dF = 100 * ((signal_ds - fitted) / fitted)

            # Z-score: compute mean/std only on non-spliced samples
            if ch_display_splices:
                dF_clean = dF[smask]
            else:
                dF_clean = dF
            dF_z = (dF - np.mean(dF_clean)) / (np.std(dF_clean) if np.std(dF_clean) > 0 else 1.0)

            trace_results[label_prefix] = {
                "time": time, "dF": dF, "dF_z": dF_z
            }

        compute_trace(data["signal1"], data["control1"], "Channel 1", display_splices_ch1)
        if show_second_channel and data.get("signal2") is not None:
            compute_trace(data["signal2"], data["control2"], "Channel 2", display_splices_ch2)

        # Store computed data for persistent display and export
        st.session_state["tab2_traces"] = trace_results
        st.session_state["tab2_display_splices_ch1"] = display_splices_ch1
        st.session_state["tab2_display_splices_ch2"] = display_splices_ch2
        st.session_state["tab2_offset"] = offset
        st.session_state["tab2_plot_choice"] = plot_choice
        st.session_state["tab2_enabled_events"] = dict(enabled_events)
        st.session_state["tab2_code_colors"] = dict(code_colors) if enabled_events else {}
        st.success("✅ Plot generated. Scroll down to view.")

    # --- Render persisted plots (survives splice edits without re-generating) ---
    if "tab2_traces" in st.session_state:
        trace_results = st.session_state["tab2_traces"]
        _plot_choice = st.session_state.get("tab2_plot_choice", "ΔF/F")
        _enabled_events = st.session_state.get("tab2_enabled_events", {})
        _code_colors = st.session_state.get("tab2_code_colors", {})

        # Reload current splice state per channel from file (always up-to-date)
        _bp = st.session_state.tdt_settings.get("block_folder", "")
        _off = st.session_state.get("tab2_offset", 0)
        _ch_splices = {}  # label -> display splices
        if _bp and os.path.isdir(_bp):
            _ch_splices["Channel 1"] = offset_splices(load_splices(_bp, channel=1), _off)
            _ch_splices["Channel 2"] = offset_splices(load_splices(_bp, channel=2), _off)
        else:
            _ch_splices["Channel 1"] = st.session_state.get("tab2_display_splices_ch1", [])
            _ch_splices["Channel 2"] = st.session_state.get("tab2_display_splices_ch2", [])

        # ========== GRAPH 1: Full trace with splice overlay ==========
        st.subheader("📈 Full Trace (with splice regions highlighted)")
        fig1 = go.Figure()
        for label, td in trace_results.items():
            y_data = td["dF"] if _plot_choice == "ΔF/F" else td["dF_z"]
            color = "blue" if "1" in label else "green"
            if label == "Channel 2":
                y_data = y_data + 3  # offset channel 2
            fig1.add_trace(go.Scatter(
                x=td["time"], y=y_data,
                mode='lines', name=f"{label} {_plot_choice}",
                line=dict(color=color),
            ))

        # Per-channel splice overlays
        _splice_colors = {"Channel 1": "red", "Channel 2": "orange"}
        for ch_label in trace_results.keys():
            for s_start, s_end in _ch_splices.get(ch_label, []):
                fig1.add_vrect(
                    x0=s_start, x1=s_end,
                    fillcolor=_splice_colors.get(ch_label, "red"), opacity=0.2,
                    layer="below", line_width=1,
                    line_color=_splice_colors.get(ch_label, "red"),
                    annotation_text=f"splice ({ch_label})",
                    annotation_position="top left",
                    annotation_font_size=8,
                    annotation_font_color=_splice_colors.get(ch_label, "red"),
                )

        # Event overlays
        if "extracted_data" in st.session_state and _enabled_events:
            events_to_plot = [
                e for e in st.session_state.extracted_data.get("events", [])
                if e.get("code") in _enabled_events and _enabled_events[e["code"]]
            ]
            grouped_events = {}
            for event in events_to_plot:
                code = event["code"]
                grouped_events.setdefault(code, []).append(event["timestamp_s"])

            ymins, ymaxs = [], []
            for trace in fig1.data:
                ymins.append(float(np.min(trace.y)))
                ymaxs.append(float(np.max(trace.y)))
            ymin = min(ymins) if ymins else -1
            ymax = max(ymaxs) if ymaxs else 1

            for code, times in grouped_events.items():
                label = st.session_state.code_map.get(code, f"Code {code}")
                color = _code_colors.get(code, "black")
                x_vals, y_vals = [], []
                for t in times:
                    x_vals.extend([t, t, None])
                    y_vals.extend([ymin, ymax, None])
                fig1.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode="lines",
                    line=dict(color=color, width=1, dash="dash"),
                    name=label, hoverinfo="skip",
                ))

        fig1.update_layout(
            title=dict(text="Fiber Photometry — Full Trace with Splice Regions", font=dict(color='black')),
            template="plotly_white",
            xaxis=dict(title=dict(text="Time (s)", font=dict(color='black')),
                       color='black', tickfont=dict(color='black'),
                       rangeslider=dict(visible=True)),
            yaxis=dict(title=dict(text="ΔF/F (%)" if _plot_choice == "ΔF/F" else "Z-scored ΔF/F",
                                  font=dict(color='black')),
                       color='black', tickfont=dict(color='black')),
            font=dict(color="black"), plot_bgcolor='white',
            paper_bgcolor='white', hovermode='closest',
            height=500, legend=dict(font=dict(color="black")),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ========== GRAPH 2: Spliced trace (samples removed) ==========
        st.subheader("📉 Spliced Trace (splice regions removed)")
        fig2 = go.Figure()
        for label, td in trace_results.items():
            time_arr = td["time"]
            dF_arr = td["dF"]
            dF_z_arr = td["dF_z"]
            # Build keep mask from this channel's splices
            _this_ch_splices = _ch_splices.get(label, [])
            if _this_ch_splices:
                ds_fs_out = 1.0 / np.median(np.diff(time_arr)) if len(time_arr) > 1 else 1.0
                keep = get_splice_mask(len(time_arr), ds_fs_out, _this_ch_splices)
            else:
                keep = np.ones(len(time_arr), dtype=bool)

            time_clean = time_arr[keep]
            dF_clean = dF_arr[keep]
            dF_z_clean = dF_z_arr[keep]

            y_data = dF_clean if _plot_choice == "ΔF/F" else dF_z_clean
            color = "blue" if "1" in label else "green"
            if label == "Channel 2":
                y_data = y_data + 3
            fig2.add_trace(go.Scatter(
                x=time_clean, y=y_data,
                mode='lines', name=f"{label} {_plot_choice} (spliced)",
                line=dict(color=color),
            ))

        fig2.update_layout(
            title=dict(text="Fiber Photometry — Spliced Trace", font=dict(color='black')),
            template="plotly_white",
            xaxis=dict(title=dict(text="Time (s)", font=dict(color='black')),
                       color='black', tickfont=dict(color='black'),
                       rangeslider=dict(visible=True)),
            yaxis=dict(title=dict(text="ΔF/F (%)" if _plot_choice == "ΔF/F" else "Z-scored ΔF/F",
                                  font=dict(color='black')),
                       color='black', tickfont=dict(color='black')),
            font=dict(color="black"), plot_bgcolor='white',
            paper_bgcolor='white', hovermode='closest',
            height=500, legend=dict(font=dict(color="black")),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ========== DATA EXPORT ==========
        st.subheader("📥 Export Trace Data")
        st.caption(
            "Export the full trace with spliced regions removed. "
            "Includes time, ΔF/F, and Z-scored ΔF/F for each channel."
        )

        for label, td in trace_results.items():
            time_arr = td["time"]
            dF_arr = td["dF"]
            dF_z_arr = td["dF_z"]

            # Apply this channel's splices
            _this_ch_splices = _ch_splices.get(label, [])
            if _this_ch_splices:
                ds_fs_out = 1.0 / np.median(np.diff(time_arr)) if len(time_arr) > 1 else 1.0
                keep = get_splice_mask(len(time_arr), ds_fs_out, _this_ch_splices)
            else:
                keep = np.ones(len(time_arr), dtype=bool)

            time_export = time_arr[keep]
            dF_export = dF_arr[keep]
            dF_z_export = dF_z_arr[keep]

            safe_label = label.replace(" ", "_")

            # CSV export
            export_df = pd.DataFrame({
                "time_s": time_export,
                "dF_F_pct": dF_export,
                "dF_F_zscore": dF_z_export,
            })
            csv_buf = io.BytesIO()
            export_df.to_csv(csv_buf, index=False)
            csv_buf.seek(0)

            # Numpy export (structured array)
            npy_buf = io.BytesIO()
            np.savez_compressed(
                npy_buf,
                time_s=time_export,
                dF_F_pct=dF_export,
                dF_F_zscore=dF_z_export,
            )
            npy_buf.seek(0)

            col_csv, col_npy = st.columns(2)
            with col_csv:
                st.download_button(
                    label=f"📄 Download {label} CSV",
                    data=csv_buf,
                    file_name=f"{safe_label}_trace_spliced.csv",
                    mime="text/csv",
                    key=f"dl_csv_{safe_label}",
                )
            with col_npy:
                st.download_button(
                    label=f"💾 Download {label} NumPy (.npz)",
                    data=npy_buf,
                    file_name=f"{safe_label}_trace_spliced.npz",
                    mime="application/octet-stream",
                    key=f"dl_npy_{safe_label}",
                )

with tab4:
    st.title("📊 Multi-Block Fiber Photometry Analyzer")

    # --- Step 1: Select Directory ---
    path = st.text_input("Enter path to parent directory:", "D:\\Fiberphotometry")

    # --- Load Metadata ---
    if st.button("🔍 Load Metadata"):
        with st.spinner("Scanning for TDT blocks..."):
            tdt_paths = get_tdt_block_paths_with_events(path)
            metadata = get_sessions_by_mouseIDs(tdt_paths)
            st.session_state["metadata"] = metadata
            st.success(f"✅ Found {len(metadata)} TDT sessions!")

    # --- Step 2: Display Metadata ---
    if "metadata" in st.session_state:
        df = pd.DataFrame(st.session_state["metadata"])
        if "datetime" in df.columns:
            df = df.sort_values("datetime")

        # --- Optional Filters ---
        with st.expander("🔍 Filter Sessions"):
            selected_experiments = st.multiselect(
                "Filter by Experiment", sorted(df["Experiment"].unique())
            )
            if selected_experiments:
                df = df[df["Experiment"].isin(selected_experiments)]

            if "datetime" in df.columns and not df["datetime"].isna().all():
                date_min, date_max = st.date_input(
                    "Filter by Date Range",
                    value=(df["datetime"].min().date(), df["datetime"].max().date())
                )
                df = df[
                    (df["datetime"].dt.date >= date_min)
                    & (df["datetime"].dt.date <= date_max)
                ]

        st.dataframe(df, width='content')

        st.markdown("---")

        # --- Step 3: Load or Create Custom Groups ---
        st.header("🧩 Custom Group Builder")

        # Auto-load previously saved groups if not already loaded
        group_file = os.path.join(path, "group_assignments.json")
        if os.path.exists(group_file) and "groups" not in st.session_state:
            with open(group_file, "r") as f:
                st.session_state["groups"] = json.load(f)
            st.info("Loaded existing group assignments from disk.")
        elif "groups" not in st.session_state:
            st.session_state["groups"] = {}

        # Create a new group
        new_group = st.text_input("Create new group (e.g., Control, Lesion, Drug):")
        if st.button("➕ Add Group"):
            if new_group and new_group not in st.session_state["groups"]:
                st.session_state["groups"][new_group] = {}
                st.success(f"Added group '{new_group}'!")
            else:
                st.warning("Group name is empty or already exists.")

# --- Step 4: Assign Mice and Sessions ---
        for group_name, group_data in list(st.session_state["groups"].items()):
            cols = st.columns([4, 1])
            with cols[0]:
                st.subheader(f"📂 {group_name}")
            with cols[1]:
        # Button to delete the group
                if st.button("🗑️ Delete", key=f"delete_{group_name}"):
                    del st.session_state["groups"][group_name]
                    st.rerun()  # refresh UI after deletion

    # Available mice
            mice = sorted(df["mouseID"].unique())

    # Let user choose which mice belong to this group
            selected_mice = st.multiselect(
                f"Select mice for {group_name}",
                mice,
                default=list(group_data.keys()),
                key=f"mice_{group_name}"
            )

    # --- Remove mice that were unselected ---
            existing_mice = set(group_data.keys())
            unselected_mice = existing_mice - set(selected_mice)
            for removed_mouse in unselected_mice:
                del st.session_state["groups"][group_name][removed_mouse]

    # --- For each selected mouse, manage sessions ---
            for mouse in selected_mice:
                mouse_sessions = df[df["mouseID"] == mouse]

                selected_sessions = st.multiselect(
                    f"Sessions for mouse {mouse} ({group_name})",
                    options=mouse_sessions["path"].tolist(),
                    default=group_data.get(mouse, []),
                    key=f"{group_name}_{mouse}_sessions"
                )

        # --- Update or remove sessions based on selection ---
                if selected_sessions:
                    st.session_state["groups"][group_name][mouse] = selected_sessions
                elif mouse in st.session_state["groups"][group_name]:
            # If user deselects all sessions, remove mouse entry
                    del st.session_state["groups"][group_name][mouse]
                
            st.markdown("---")

        # --- Step 5: Group Summary ---
        if st.session_state["groups"]:
            st.subheader("📋 Group Summary")
            for group, mice in st.session_state["groups"].items():
                st.markdown(f"**🧩 {group}**")
                for mouse, sessions in mice.items():
                    st.write(f" - 🐭 {mouse}: {len(sessions)} session(s)")

            # --- Step 6: Save Group Assignments ---
            if st.button("💾 Save Group Assignments"):
                save_path = os.path.join(path, "group_assignments.json")
                with open(save_path, "w") as f:
                    json.dump(st.session_state["groups"], f, indent=2)
                st.success(f"Saved to {save_path}")

    else:
        st.info("Enter a directory and click 'Load Metadata' to begin.")
    
    if "groups" not in st.session_state:
        st.session_state["groups"] = {}

    if not st.session_state["groups"]:
        st.info("No groups defined. Create groups and assign sessions first.")
    else:
        # Build event name choices from sessions currently assigned in groups
        selected_session_paths = []
        for g, mice in st.session_state["groups"].items():
            for m, paths in mice.items():
                selected_session_paths.extend(paths)
        # gather event names from metadata for only those sessions
        event_names = set()
        if "metadata" in st.session_state:
            meta_lookup = {m['path']: m for m in st.session_state["metadata"]}
            for sp in selected_session_paths:
                meta = meta_lookup.get(sp)
                if not meta:
                    continue
                for e in meta.get("events", []):
                    event_names.add(e.get("event"))
        event_names = sorted([en for en in event_names if en is not None])
        if not event_names:
            st.error("No events found in selected sessions' metadata.")
        else:
            st.markdown("### Event & Processing Options")
            selected_event = st.selectbox("Select event (one):", event_names)
            batch_pct_choice = st.selectbox(
                "Which PCT epoch onset to use for alignment?",
                ["1st PCT onset", "2nd PCT onset"],
                index=1,
                key="batch_pct_choice"
            )
            batch_pct_idx = 0 if batch_pct_choice == "1st PCT onset" else 1
            pre_t = st.number_input("Peri-event PRE time (s)", min_value=0.0, value=5.0)
            post_t = st.number_input("Peri-event POST time (s)", min_value=0.0, value=10.0)
            baseline_lower = st.number_input("Baseline window start (s, relative to T0, negative)", value=-4.0)
            baseline_upper = st.number_input("Baseline window end (s, relative to T0, negative or <0)", value=-1.0)
            downsample_factor = st.number_input("Downsample factor (integer ≥1)", min_value=1, value=1, step=1)
            metric_start = st.number_input("Metric window start (s, relative to T0)", value=0.0)
            metric_end = st.number_input("Metric window end (s, relative to T0)", value=5.0)

            if metric_end <= metric_start:
                st.error("Metric end must be > metric start.")
            else:
                # Output folder base (timestamped to avoid overwrite)
                timestamp_str = dt.now().strftime("%Y%m%d_%H%M%S")
                plots_base = os.path.join(path, "plots", f"{_safe_name(selected_event)}_{timestamp_str}")
                os.makedirs(plots_base, exist_ok=True)

                # Prepare containers for results
                per_mouse_session_traces = defaultdict(list)   # mouse -> list of session avg traces (arrays)
                per_mouse_session_info = defaultdict(list)     # mouse -> list of (group, session_name)
                per_group_mouse_traces = defaultdict(lambda: defaultdict(list))  # group -> mouse -> mean trace
                all_metrics = []  # rows for CSV

                # Processing button
                if st.button("▶️ Run Averaging & Save Plots"):
                    with st.spinner("Running batch processing..."):
                        try:
                            summary = run_batch_processing(
                                base_path=path,
                                selected_event=selected_event,
                                pre_t=pre_t, post_t=post_t,
                                baseline_lower=baseline_lower, baseline_upper=baseline_upper,
                                downsample_factor=downsample_factor,
                                metric_start=metric_start, metric_end=metric_end,
                                metadata_list=st.session_state.get("metadata", []),
                                groups_dict=st.session_state.get("groups", {}),
                                find_stream_by_substr=find_stream_by_substr,
                                load_tdt_block=load_tdt_block,
                                verbose=True,
                                pct_onset_index=batch_pct_idx,
                            )
                            st.success(f"Processing complete. Output folder: {summary['output_files']['event_folder']}")
                            st.write(summary)  # visible run summary in the UI
                        except Exception as e:
                            st.error(f"Processing failed: {e}")
                            import traceback
                            st.text(traceback.format_exc())

with tab5:
    st.title("📊 Advanced Graphing")
    st.markdown("Post-hoc visualization of batch-processed data")

    # ── 1. Select Data Folder ──────────────────────────────────────────────
    st.header("1. Select Data Folder")

    adv_base_path = st.text_input(
        "Enter path to parent directory:",
        value="D:\\Fiberphotometry",
        key="adv_path",
    )

    if not os.path.isdir(adv_base_path):
        st.warning("⚠️ Please enter a valid folder path.")
    else:
        adv_folders = sorted(
            f for f in os.listdir(adv_base_path)
            if os.path.isdir(os.path.join(adv_base_path, f))
            and not f.startswith(".")
        )

        if not adv_folders:
            st.info("No subfolders found in this directory.")
        else:
            adv_selected_folder = st.selectbox(
                "Available Folders:",
                adv_folders,
                key="adv_folder_select",
            )
            adv_full_path = os.path.join(adv_base_path, adv_selected_folder)

            # ── 2. Select Event Folder ─────────────────────────────────────
            st.header("2. Select Event Folder")

            adv_event_folders = discover_event_folders(adv_full_path)

            if not adv_event_folders:
                st.info(
                    "No `plots/` subfolder found. Run batch processing "
                    "first (Tab 4) or select a different folder."
                )
            else:
                st.success(f"Found {len(adv_event_folders)} event folder(s)")

                adv_selected_event = st.selectbox(
                    "Select batch processing output:",
                    adv_event_folders,
                    key="adv_event_folder",
                )

                adv_event_path = os.path.join(
                    adv_full_path, "plots", adv_selected_event
                )

                # Debug: show what's inside the event folder
                with st.expander("📁 Event folder contents (debug)"):
                    st.caption(f"Path: `{adv_event_path}`")
                    if os.path.isdir(adv_event_path):
                        for item in sorted(os.listdir(adv_event_path)):
                            full_item = os.path.join(adv_event_path, item)
                            if os.path.isdir(full_item):
                                st.text(f"📁 {item}/")
                            else:
                                st.text(f"   {item}")
                    else:
                        st.error(f"Path does not exist!")

                # ── 3. Choose Graph Type ───────────────────────────────────
                st.header("3. Choose Graph Type")

                adv_graph_type = st.radio(
                    "What kind of graph do you want to make?",
                    [
                        "Signal Mean Bar Plot",
                        "AUC Bar Plot",
                        "Heatmap",
                    ],
                    key="adv_graph_type",
                )

                # Map display name → internal key
                _graph_type_map = {
                    "Signal Mean Bar Plot": "signal_mean",
                    "AUC Bar Plot": "auc",
                    "Heatmap": "heatmap",
                }
                adv_internal_type = _graph_type_map[adv_graph_type]

                # Determine which CSVs to show based on graph type
                needs_long = adv_internal_type == "heatmap"

                if needs_long:
                    st.info(
                        "Heatmaps require **trial-level LONG CSVs** "
                        "(from `mice/` folder)."
                    )
                    adv_csv_map = discover_long_trace_sessions(adv_event_path)
                    csv_label = "LONG trace CSVs"
                else:
                    st.info(
                        f"{adv_graph_type}s use **session prism CSVs** "
                        f"(from `prism_tables/sessions/`)."
                    )
                    adv_csv_map = discover_prism_sessions(adv_event_path)
                    csv_label = "prism session CSVs"

                # ── 4. Select Sessions ─────────────────────────────────────
                st.header("4. Select Sessions")

                if not adv_csv_map:
                    st.warning(f"No {csv_label} found in this event folder.")
                    # Show diagnostic info
                    if needs_long:
                        check_dir = os.path.join(adv_event_path, "mice")
                    else:
                        check_dir = os.path.join(
                            adv_event_path, "prism_tables", "sessions"
                        )
                    st.caption(f"Looked in: `{check_dir}`")
                    st.caption(
                        f"Directory exists: {os.path.isdir(check_dir)}"
                    )
                else:
                    csv_labels = sorted(adv_csv_map.keys())
                    st.success(f"Found {len(csv_labels)} {csv_label}")

                    adv_selected_csvs = st.multiselect(
                        f"Select {csv_label} to process:",
                        csv_labels,
                        key="adv_csv_select",
                    )

                    if not adv_selected_csvs:
                        st.info("Select one or more sessions to continue.")
                    else:
                        st.success(
                            f"{len(adv_selected_csvs)} session(s) selected"
                        )

                        # ── 5. Parameters ──────────────────────────────────
                        st.header("5. Parameters")

                        adv_signal_type = st.radio(
                            "Signal type:",
                            ["z-score", "raw ΔF/F"],
                            key="adv_signal_type",
                        )

                        col_bl, col_rsp = st.columns(2)

                        with col_bl:
                            st.markdown("**Baseline window (s)**")
                            adv_bl_start = st.number_input(
                                "Baseline start (s):",
                                value=-4.0,
                                key="adv_bl_start",
                            )
                            adv_bl_end = st.number_input(
                                "Baseline end (s):",
                                value=-1.0,
                                key="adv_bl_end",
                            )

                        with col_rsp:
                            st.markdown("**Response window (s)**")
                            adv_rsp_start = st.number_input(
                                "Response start (s):",
                                value=0.0,
                                key="adv_rsp_start",
                            )
                            adv_rsp_end = st.number_input(
                                "Response end (s):",
                                value=5.0,
                                key="adv_rsp_end",
                            )

                        adv_window_valid = True
                        if adv_bl_end <= adv_bl_start:
                            st.error(
                                "Baseline end must be > baseline start"
                            )
                            adv_window_valid = False
                        if adv_rsp_end <= adv_rsp_start:
                            st.error(
                                "Response end must be > response start"
                            )
                            adv_window_valid = False

                        # ── 6. Generate ────────────────────────────────────
                        st.header("6. Generate")

                        generate_disabled = not adv_window_valid

                        if st.button(
                            "🚀 Generate Plots",
                            type="primary",
                            disabled=generate_disabled,
                            key="adv_generate",
                        ):
                            with st.spinner("Generating plots…"):
                                try:
                                    output_root = os.path.join(
                                        adv_event_path,
                                        "advanced_graphs",
                                    )

                                    if needs_long:
                                        # Heatmap: process each LONG CSV
                                        # Build discovered dict with
                                        # matching prism CSVs
                                        prism_map = discover_prism_sessions(
                                            adv_event_path
                                        )
                                        discovered = {}
                                        for label in adv_selected_csvs:
                                            long_path = adv_csv_map[label]
                                            # Find matching prism CSV by
                                            # session key overlap
                                            session_key = label.replace(
                                                "_peri_event_traces_LONG.csv",
                                                "",
                                            )
                                            match_prism = None
                                            for pname, ppath in prism_map.items():
                                                if session_key in pname:
                                                    match_prism = ppath
                                                    break
                                            discovered[label] = {
                                                "prism_path": match_prism,
                                                "long_csv_path": long_path,
                                            }
                                    else:
                                        # Bar plots: process each prism CSV
                                        long_map = discover_long_trace_sessions(
                                            adv_event_path
                                        )
                                        discovered = {}
                                        for label in adv_selected_csvs:
                                            prism_path = adv_csv_map[label]
                                            prism_key = label.replace(
                                                "_prism.csv", ""
                                            )
                                            match_long = None
                                            # LONG name: {session}_LONG.csv
                                            # prism key: {mouse}_{session}
                                            # Check long_key in prism_key
                                            for lname, lpath in long_map.items():
                                                long_key = lname.replace(
                                                    "_peri_event_traces_LONG.csv",
                                                    "",
                                                )
                                                if long_key and long_key in prism_key:
                                                    match_long = lpath
                                                    break
                                            discovered[label] = {
                                                "prism_path": prism_path,
                                                "long_csv_path": match_long,
                                            }

                                    adv_summary = run_advanced_graphing_from_discovered(
                                        discovered_sessions=discovered,
                                        selected_basenames=adv_selected_csvs,
                                        output_folder=output_root,
                                        signal_type=adv_signal_type,
                                        baseline_window=(
                                            adv_bl_start, adv_bl_end,
                                        ),
                                        response_window=(
                                            adv_rsp_start, adv_rsp_end,
                                        ),
                                        plot_types=[adv_internal_type],
                                    )

                                    st.success(
                                        f"✅ Done! Processed "
                                        f"{adv_summary['n_sessions_processed']}"
                                        f" session(s)."
                                    )
                                    st.info(
                                        f"Output folder: "
                                        f"{adv_summary['output_folder']}"
                                    )

                                    for sr in adv_summary.get(
                                        "session_results", []
                                    ):
                                        for w in sr.get("warnings", []):
                                            st.warning(w)

                                    if adv_summary["n_errors"] > 0:
                                        st.error(
                                            f"{adv_summary['n_errors']}"
                                            f" session(s) had errors:"
                                        )
                                        for err in adv_summary["errors"]:
                                            st.text(
                                                f"  {err['session']}:"
                                                f" {err['error']}"
                                            )

                                    agg_csv = adv_summary.get(
                                        "aggregated_csv"
                                    )
                                    if agg_csv and os.path.exists(agg_csv):
                                        st.subheader(
                                            "Aggregated Session Stats"
                                        )
                                        st.dataframe(
                                            pd.read_csv(agg_csv),
                                            use_container_width=True,
                                        )

                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
                                    import traceback
                                    st.text(traceback.format_exc())
