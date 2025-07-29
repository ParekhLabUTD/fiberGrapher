import streamlit as st
import os
import tdt
import tempfile
import hashlib

from OHRBETsEventExtractor import parse_ohrbets_serial_log, load_data
from nxtEventExtrator import readfile, get_events, get_event_codes

st.set_page_config(layout="wide")

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
        "extract_pct": False
    }

@st.cache_resource
def load_tdt_block(path):
    return tdt.read_block(path)

def get_file_hash(file):
    file.seek(0)
    data = file.read()
    file.seek(0)
    return hashlib.md5(data).hexdigest()

st.title("📂 TDT + Events Data Extractor")

# --- TDT Block Selection ---
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

        if st.button("📥 Extract Data"):
            tdt_data = {
                "signal1": data.streams[st.session_state.signal_stream].data if st.session_state.signal_stream else None,
                "control1": data.streams[st.session_state.control_stream].data if st.session_state.control_stream else None,
                "signal2": data.streams[st.session_state.signal2_stream].data if st.session_state.signal2_stream else None,
                "control2": data.streams[st.session_state.control2_stream].data if st.session_state.control2_stream else None,
                "pct": data.epocs['PtC1'].onset if pctBool and 'PtC1' in data.epocs else None,
                "fs": data.streams[st.session_state.signal_stream].fs if st.session_state.signal_stream else None,
                "time": data.streams[st.session_state.signal_stream].time if st.session_state.signal_stream else None
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
