import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objs as go
import tdt
import hashlib
import matplotlib.pyplot as plt
import itertools

from OHRBETsEventExtractor import parse_ohrbets_serial_log, load_data
from nxtEventExtrator import readfile, get_events, get_event_codes

color_cycle = itertools.cycle([
    "red", "green", "purple", "orange", "brown", "pink", "gray"
])

# Set up Streamlit
st.set_page_config(layout="wide")
tab1, tab2, tab3 = st.tabs(["📦 Data Extractor","📈 Graph Viewer", "🧪 Peri Event Plots"])

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
        event_name = st.selectbox("Select event code for alignment:", sorted(set(e["event"] for e in st.session_state["extracted_data"]["events"] if "event" in e)))
        signal_set = st.radio("Select signal/control pair:", ["1", "2"], horizontal=True)
        PRE_TIME = st.number_input("Seconds before event", value=5.0)
        POST_TIME = st.number_input("Seconds after event", value=10.0)
        downsample_factor = st.number_input("Downsample factor", min_value=1, value=1, step=1)
        apply_zscore = st.checkbox("Z-score ΔF/F before peri-event extraction")

        # Load data
        key_prefix = "" if signal_set == "1" else "2"
        signal = st.session_state.extracted_data.get(f"signal{signal_set}")
        control = st.session_state.extracted_data.get(f"control{signal_set}")
        fs = st.session_state.extracted_data.get("fs")

        if signal is None or control is None or fs is None:
            st.error("Signal or control data missing.")
        else:
            # Downsample
            if downsample_factor > 1:
                signal = signal[::downsample_factor]
                control = control[::downsample_factor]
                fs = fs / downsample_factor

            # ΔF/F calculation
            fit = np.polyfit(control, signal, 1)
            fitted = fit[0] * control + fit[1]
            dFF = 100 * (signal - fitted) / fitted

            if apply_zscore:
                dFF = (dFF - np.mean(dFF)) / np.std(dFF)

            time = np.arange(len(dFF)) / fs
            TRANGE = [int(-PRE_TIME * fs), int(POST_TIME * fs)]

            snippets = []

            for event in st.session_state.extracted_data["events"]:
                if event.get("event") != event_name:
                    continue
                on = event.get("timestamp_s")
                if on is None or on * fs < -TRANGE[0] or (on * fs + TRANGE[1]) >= len(dFF):
                    continue
                idx = int(on * fs)
                snippet = dFF[idx + TRANGE[0]:idx + TRANGE[1]]
                if len(snippet) == (TRANGE[1] - TRANGE[0]):
                    snippets.append(snippet)

            if not snippets:
                st.warning("No valid event-aligned snippets found.")
            else:
                snippets = np.array(snippets)
                mean_snip = np.mean(snippets, axis=0)
                std_snip = np.std(snippets, axis=0)
                peri_time = np.linspace(TRANGE[0], TRANGE[1], len(mean_snip)) / fs

                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(111)

                for snip in snippets:
                    ax.plot(peri_time, snip, linewidth=0.5, color='gray', alpha=0.5)
                ax.plot(peri_time, mean_snip, color='green', linewidth=2, label='Mean ΔF/F')
                ax.fill_between(peri_time, mean_snip - std_snip, mean_snip + std_snip, color='green', alpha=0.2, label='±STD')
                ax.axvline(0, color='slategray', linestyle='--', linewidth=2, label=f'{event_name} Onset')
                ax.set_title("Peri-Event ΔF/F Response (Z-scored)" if apply_zscore else "Peri-Event ΔF/F Response")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("ΔF/F (Z-score)" if apply_zscore else "ΔF/F (%)")
                ax.legend()
                ax.grid(True)

                st.pyplot(fig,width='content')

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
                events_data = load_data(tmp_path)
                parsed = parse_ohrbets_serial_log(events_data)
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

        fig = go.Figure()

        def process_trace(signal, control, color, label_prefix):
            signal_ds = np.array([np.mean(signal[i:i+downsample_factor]) for i in range(0, len(signal), downsample_factor)])
            control_ds = np.array([np.mean(control[i:i+downsample_factor]) for i in range(0, len(control), downsample_factor)])
            time = np.arange(len(signal_ds)) / data["fs"] * downsample_factor

            fit = np.polyfit(control_ds, signal_ds, 1)
            fitted = fit[0] * control_ds + fit[1]
            dF = 100 * ((signal_ds - fitted) / fitted)
            dF_z = (dF - np.mean(dF)) / np.std(dF)

            idx = np.searchsorted(time, cutoff_time)
            time = time[idx:] - time[idx]
            dF = dF[idx:]
            dF_z = dF_z[idx:]

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
            grouped_events.setdefault(code, []).append(event["timestamp_s"] - cutoff_time)

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
