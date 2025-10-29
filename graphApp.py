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

from OHRBETsEventExtractor import parse_ohrbets_serial_log
from nxtEventExtrator import readfile, get_events, get_event_codes
from sessionInfoExtractor import *

color_cycle = itertools.cycle([
    "red", "green", "purple", "orange", "brown", "pink", "gray"
])


import math
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
tab1, tab2, tab3, tab4 = st.tabs(["📦 Data Extractor", "📈 Graph Viewer", "🧪 Peri Event Plots","Average Sessions"])

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
    return hashlib.md5(data).hexdigest()

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

# =====================
# TAB 2: Graph Viewer
# =====================
with tab2:
    st.title("Fiber Photometry Grapher (Interactive)")

    downsample_factor = st.number_input("Downsampling factor", min_value=1, value=100)
    cutoff_time_from_pct = st.checkbox("Use PCT's 2nd event (code 12) to align all events?", value=False)
    if cutoff_time_from_pct:
        cutoff_time = st.session_state["extracted_data"]["pct"][1]
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
        pct_onset = data["pct"][1]              # or however your PCT onset is stored
        code12_times = [e["timestamp_s"] for e in data["events"] if e["code"] == 12]
        offset = pct_onset - code12_times[1]

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
        st.info("Create groups and assign sessions in the UI above before processing.")
        st.stop()

    if not st.session_state["groups"]:
        st.info("No groups defined. Create groups and assign sessions first.")
        st.stop()

    # Build event name choices from sessions currently assigned in groups
    selected_session_paths = []
    for g, mice in st.session_state["groups"].items():
        for m, paths in mice.items():
            selected_session_paths.extend(paths)
    # gather event names from metadata for only those sessions
    event_names = set()
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
        st.stop()

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
        st.stop()

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
        st.info("Starting batch processing... (warnings will appear in server console)")

        # iterate groups -> mice -> sessions
        for group_name, mice in st.session_state["groups"].items():
            for mouse_id, sessions in mice.items():
                for s_path in sessions:
                    meta = next((m for m in st.session_state["metadata"] if m['path']==s_path and m['mouseID']==mouse_id), None)
                    if not meta:
                        print(f"WARNING: metadata missing for session path {s_path}", flush=True)
                        continue

                    # find event timestamps for selected_event
                    event_times = [e['timestamp_s'] for e in meta.get("events", []) if e.get("event") == selected_event]
                    if not event_times:
                        print(f"WARNING: no '{selected_event}' events in session {s_path} (mouse {mouse_id})", flush=True)
                        continue

                    # load block
                    try:
                        block = load_tdt_block(s_path)
                    except Exception as e:
                        print(f"ERROR: failed to read block {s_path}: {e}", flush=True)
                        continue

                    # pick streams by signalChannelSet using substring match of exact patterns with underscore
                    scs = meta.get("signalChannelSet", 1)
                    if scs == 1:
                        sig_sub, ctrl_sub = "_465A", "_415A"
                    else:
                        sig_sub, ctrl_sub = "_465C", "_415C"

                    sig_key = find_stream_by_substr(block, sig_sub)
                    ctrl_key = find_stream_by_substr(block, ctrl_sub)
                    if sig_key is None or ctrl_key is None:
                        print(f"WARNING: cannot find streams for session {s_path} (expected {sig_sub} and {ctrl_sub}).", flush=True)
                        continue

                    sig = np.asarray(block.streams[sig_key].data).flatten()
                    ctrl = np.asarray(block.streams[ctrl_key].data).flatten()
                    fs_orig = float(block.streams[sig_key].fs)

                    if meta.get('interpretor',1) ==2: 
                        if meta.get('signalChannelSet',1) == 1:
                            pct_name = 'PtC1'
                        else:
                            pct_name = 'PtC2'
                        onset_list = block.epocs[pct_name].onset
                        try:
                            pct_onset = float(onset_list[1])  # confirm this index corresponds to onset
                            code12_times = [e["timestamp_s"] for e in meta.get("events", []) if e.get("code") == 12]
                            if len(code12_times) < 2:
                                print(f"WARNING: less than 2 code-12 events for session {s_path}; skipping alignment.", flush=True)
                            else:
                                offset = pct_onset - code12_times[1]
                                time_full = np.arange(len(sig)) / fs_orig
                                time_full -= offset
                                valid = time_full >= 0
                                sig = sig[valid]
                                ctrl = ctrl[valid]
                        except Exception as e:
                            print(f"WARNING: alignment failed for {s_path}: {e}", flush=True)

                    # Downsample BEFORE dF/F if requested (simple decimation)
                    ds = int(downsample_factor)
                    if ds > 1:
                        sig = sig[::ds]
                        ctrl = ctrl[::ds]
                        fs = fs_orig / ds
                    else:
                        fs = fs_orig

                    # Build time vector (seconds)
                    time_vec = np.arange(len(sig)) / fs

                    # Compute ΔF/F via linear regression (control -> signal)
                    try:
                        p = np.polyfit(ctrl, sig, 1)
                        fitted = p[0] * ctrl + p[1]
                        dff = 100.0 * (sig - fitted) / (fitted + 1e-12)
                    except Exception as e:
                        print(f"ERROR computing regression ΔF/F for {s_path}: {e}", flush=True)
                        continue

                    # extract peri-event snippets for this session
                    n_samples = int(round((pre_t + post_t) * fs))
                    peri_traces = []
                    peri_times = np.linspace(-pre_t, post_t, n_samples)

                    for ts in event_times:
                        start_idx = int(round((ts - pre_t) * fs))
                        end_idx = start_idx + n_samples
                        if start_idx < 0 or end_idx > len(dff):
                            print(f"WARNING: event at {ts}s in {s_path} out of bounds after windowing; skipping that trial.", flush=True)
                            continue
                        snippet = dff[start_idx:end_idx].astype(float)

                        # baseline z-score using baseline_lower..baseline_upper
                        bstart = int(round((ts + baseline_lower) * fs))
                        bend = int(round((ts + baseline_upper) * fs))
                        bstart = max(0, bstart)
                        bend = min(len(dff), bend)
                        if bend > bstart:
                            baseline_vals = dff[bstart:bend]
                            mu = np.mean(baseline_vals)
                            sigma = np.std(baseline_vals)
                            if sigma > 0:
                                snippet = (snippet - mu) / sigma
                            else:
                                snippet = snippet - mu
                        else:
                            # if baseline invalid, subtract global mean
                            mu = np.mean(dff)
                            snippet = snippet - mu

                        peri_traces.append(snippet)

                        # compute metrics per trial (peak, auc, latency) over metric window
                        mstart_idx = int(round((pre_t + metric_start) * fs))
                        mend_idx = int(round((pre_t + metric_end) * fs))
                        # ensure indices in bounds
                        mstart_idx = max(0, mstart_idx)
                        mend_idx = min(len(snippet), mend_idx)
                        if mend_idx > mstart_idx:
                            post_segment = snippet[mstart_idx:mend_idx]
                            peak = float(np.max(post_segment))
                            auc = float(np.trapz(post_segment, dx=1.0/fs))
                            latency_idx = int(np.argmax(post_segment))
                            latency = float((latency_idx) / fs + metric_start)  # relative to T0
                        else:
                            peak, auc, latency = np.nan, np.nan, np.nan

                        all_metrics.append({
                            "group": group_name,
                            "mouse": mouse_id,
                            "session": os.path.basename(s_path),
                            "event_time": ts,
                            "peak": peak,
                            "auc": auc,
                            "latency": latency
                        })

                    if len(peri_traces) == 0:
                        print(f"WARNING: no valid peri-event snippets for session {s_path}", flush=True)
                        continue

                    peri_traces = np.vstack(peri_traces)  # trials x samples
                    session_mean = np.mean(peri_traces, axis=0)
                    session_sem = peri_traces.std(axis=0) / np.sqrt(peri_traces.shape[0])

                    # Save per-session plot (and also store for mouse-level averaging)
                    ev_folder = os.path.join(plots_base, _safe_name(selected_event))
                    sess_folder = os.path.join(ev_folder, "per_mouse_sessions", _safe_name(group_name), _safe_name(mouse_id))
                    os.makedirs(sess_folder, exist_ok=True)
                    sess_fname = f"{_safe_name(mouse_id)}_{_safe_name(os.path.basename(s_path))}_session_avg.png"
                    sess_path = os.path.join(sess_folder, sess_fname)

                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.plot(peri_times, session_mean, color="blue")
                    ax.fill_between(peri_times, session_mean - session_sem, session_mean + session_sem, alpha=0.3)
                    ax.axvline(0, color="red", linestyle="--")
                    ax.set_title(f"{group_name} | {mouse_id} | {os.path.basename(s_path)}")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("ΔF/F (z-scored baseline)")
                    fig.tight_layout()
                    fig.savefig(sess_path)
                    plt.close(fig)

                    per_mouse_session_traces[mouse_id].append(session_mean)
                    per_mouse_session_info[mouse_id].append((group_name, os.path.basename(s_path)))

                # end sessions loop for this mouse
            # end mice loop for this group
        # end groups loop

        # --- Per-mouse averages and saving per-mouse plots (including each session plot already saved) ---
        mouse_avg_folder = os.path.join(plots_base, _safe_name(selected_event), "per_mouse_avgs")
        os.makedirs(mouse_avg_folder, exist_ok=True)

        per_group_mean_traces = defaultdict(list)  # group -> list of mean traces from mice

        for mouse_id, traces in per_mouse_session_traces.items():
            if not traces:
                continue
            # average across sessions for that mouse
            traces_arr = np.vstack(traces)  # sessions x samples
            mouse_mean = np.mean(traces_arr, axis=0)
            mouse_sem = traces_arr.std(axis=0) / math.sqrt(traces_arr.shape[0])

            # save mouse average
            mouse_fname = f"{_safe_name(mouse_id)}_avg.png"
            mouse_path = os.path.join(mouse_avg_folder, mouse_fname)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(peri_times, mouse_mean, color="green")
            ax.fill_between(peri_times, mouse_mean - mouse_sem, mouse_mean + mouse_sem, alpha=0.3)
            ax.axvline(0, color="red", linestyle="--")
            ax.set_title(f"Mouse avg: {mouse_id}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ΔF/F (z-scored baseline)")
            fig.tight_layout()
            fig.savefig(mouse_path)
            plt.close(fig)

            # find group(s) for this mouse (a mouse may be assigned to a single group in UI)
            # We will attach this mouse_mean to each group it belongs to
            for gname, mice in st.session_state["groups"].items():
                if mouse_id in mice:
                    per_group_mean_traces[gname].append(mouse_mean)

        # --- Per-group averages and saving ---
        group_avg_folder = os.path.join(plots_base, _safe_name(selected_event), "per_group_avgs")
        os.makedirs(group_avg_folder, exist_ok=True)
        combined_fig, combined_ax = plt.subplots(figsize=(8,5))

        color_iter = itertools.cycle(["red","blue","green","orange","purple","brown","cyan"])
        for gname, mouse_means in per_group_mean_traces.items():
            if not mouse_means:
                print(f"WARNING: no mouse means for group {gname}", flush=True)
                continue
            arr = np.vstack(mouse_means)  # mice x samples
            g_mean = np.mean(arr, axis=0)
            g_sem = arr.std(axis=0) / math.sqrt(arr.shape[0])

            # save group avg plot
            g_fname = f"{_safe_name(gname)}_avg.png"
            g_path = os.path.join(group_avg_folder, g_fname)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(peri_times, g_mean, label=gname)
            ax.fill_between(peri_times, g_mean - g_sem, g_mean + g_sem, alpha=0.3)
            ax.axvline(0, color="red", linestyle="--")
            ax.set_title(f"Group avg: {gname}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ΔF/F (z-scored baseline)")
            fig.tight_layout()
            fig.savefig(g_path)
            plt.close(fig)

            # add to combined comparison plot
            color = next(color_iter)
            combined_ax.plot(peri_times, g_mean, label=gname, color=color)
            combined_ax.fill_between(peri_times, g_mean - g_sem, g_mean + g_sem, alpha=0.15, color=color)

        combined_ax.axvline(0, color="black", linestyle="--")
        combined_ax.set_xlabel("Time (s)")
        combined_ax.set_ylabel("ΔF/F (z-scored baseline)")
        combined_ax.set_title(f"Group comparison — Event: {selected_event}")
        combined_ax.legend()
        combined_path = os.path.join(plots_base, _safe_name(selected_event), f"{_safe_name(selected_event)}_group_comparison.png")
        combined_fig.tight_layout()
        combined_fig.savefig(combined_path)
        plt.close(combined_fig)

        # --- Save metrics CSV ---
        metrics_df = pd.DataFrame(all_metrics)
        metrics_fname = f"photometry_metrics_{_safe_name(selected_event)}_{timestamp_str}.csv"
        metrics_path = os.path.join(plots_base, _safe_name(selected_event), metrics_fname)
        metrics_df.to_csv(metrics_path, index=False)

        st.success(f"Processing complete. Plots and metrics saved to: {os.path.join(plots_base, _safe_name(selected_event))}")
        print(f"Saved plots & metrics at {os.path.join(plots_base, _safe_name(selected_event))}", flush=True)