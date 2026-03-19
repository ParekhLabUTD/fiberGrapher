import streamlit as st
import numpy as np
import tempfile
import os
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
            apply_pct_alignment = st.checkbox("Align signal to PCT's 2nd onset (code 12)", value=False)
            if apply_pct_alignment:
                try:
                    pct = st.session_state.extracted_data.get("pct", [])
                    events = st.session_state.extracted_data.get("events", [])
                    if len(pct) < 2:
                        st.warning("Less than 2 PCT onsets found — skipping alignment.")
                    else:
                        pct_onset = float(pct[1])
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
                            st.info(f"Applied PCT alignment offset of {offset:.3f} s")
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

            # ΔF/F calculation
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
                    # Global z-scoring (whole trace)
                    # ----------------------------
                    trace_mean = np.mean(dFF)
                    trace_std = np.std(dFF)
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
    cutoff_time_from_pct = st.checkbox("Use PCT's 2nd event (code 12) to align all events?", value=False)
    if cutoff_time_from_pct:
        cutoff_time = 0
    else:
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

    if st.button("Generate Plot"):
        if "extracted_data" not in st.session_state:
            st.error("❌ No TDT data loaded. Please use the Data Extractor tab.")
            st.stop()
        data = st.session_state.extracted_data
        try:
            pct_onset = data["pct"][1]
            code12_times = [e["timestamp_s"] for e in data["events"] if e["code"] == 12]
            offset = pct_onset - code12_times[1]
        except:
            #add logic when skipping the pct line up
            offset=cutoff_time

        fig = go.Figure()

        def process_trace(signal, control, color, label_prefix):
            signal_ds = np.array([np.mean(signal[i:i+downsample_factor]) for i in range(0, len(signal), downsample_factor)])
            control_ds = np.array([np.mean(control[i:i+downsample_factor]) for i in range(0, len(control), downsample_factor)])
            time = np.arange(len(signal_ds)) / data["fs"] * downsample_factor

            time -= offset
            valid = time >= 0
            time = time[valid]
            signal_ds = signal_ds[valid]
            control_ds = control_ds[valid]

            min_len = min(len(signal), len(control))
            if len(signal) != len(control):
                signal = signal[:min_len]
                control = control[:min_len]


            fit = np.polyfit(control_ds,signal_ds, 1)
            fitted = fit[0] * control_ds + fit[1]
            dF = 100 * ((signal_ds - fitted) / fitted)
            dF_z = (dF - np.mean(dF)) / np.std(dF)

            if label_prefix == "channel 2":
                dF = dF + 3

            fig.add_trace(go.Scatter(
                x=time,
                y=dF if plot_choice == "ΔF/F" else dF_z,
                mode='lines',
                name=f"{label_prefix} {plot_choice}",
                line=dict(color=color),
                legendgrouptitle=dict(font=dict(color='black'))
            ))

        process_trace(data["signal1"], data["control1"], "blue", "Channel 1")

        if show_second_channel:
            process_trace(data["signal2"], data["control2"], "green", "Channel 2")

        fig.update_layout(
            title=dict(text="Fiber Photometry with Behavioral Events", font=dict(color='black')),
            template="plotly_white",
            xaxis=dict(
                title=dict(text="Time (s)", font=dict(color='black')),
                color='black',
                tickfont=dict(color='black'),
                rangeslider=dict(visible=True)
            ),
            yaxis=dict(
                title=dict(text="ΔF/F (%)" if plot_choice == "ΔF/F" else "Z-scored ΔF/F", font=dict(color='black')),
                color='black',
                tickfont=dict(color='black'),
            ),
            font=dict(color="black"),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest',
            height=500,
            legend=dict(font=dict(color="black"))
        )

        # --- Overlay Events (Optimized) ---
        events_to_plot = [
            e for e in st.session_state.extracted_data.get("events", [])
            if e.get("code") in enabled_events and enabled_events[e["code"]]
        ]

        # Group events by code
        grouped_events = {}
        for event in events_to_plot:
            code = event["code"]
            grouped_events.setdefault(code, []).append(event["timestamp_s"])

        # Get y-limits from data traces
        ymins, ymaxs = [], []
        for trace in fig.data:
            ymins.append(np.min(trace.y))
            ymaxs.append(np.max(trace.y))
        ymin = float(np.min(ymins)) if ymins else -1
        ymax = float(np.max(ymaxs)) if ymaxs else 1

        # Add one trace per event code (fast)
        for code, times in grouped_events.items():
            label = st.session_state.code_map.get(code, f"Code {code}")
            color = code_colors.get(code, "black")

            x_vals, y_vals = [], []
            for t in times:
                x_vals.extend([t, t, None])
                y_vals.extend([ymin, ymax, None])

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines",
                line=dict(color=color, width=1, dash="dash"),
                name=label,
                hoverinfo="skip"
            ))

        st.plotly_chart(fig, width='stretch')

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
