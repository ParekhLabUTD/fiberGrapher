import streamlit as st
import numpy as np
import tempfile
import os
import plotly.graph_objs as go

from OHRBETsEventExtractor import parse_ohrbets_serial_log, load_data 
from nxtEventExtrator import readfile, get_events

st.title("Fiber Photometry Grapher (Interactive)")
code_map = {
    1:"Foraging",
    2:"Reward Approach",
    3:"Entering Foraging Arena",
    4:"Exiting Foraging Arena",
    5:"Home Approach",
    6:"Entering Home Arena",
    7:"Exiting Home Arena",
    8:"Grooming",
    9:"Eating"
}
# --- Upload Files ---
npz_file = st.file_uploader("Upload photometry .npz file", type="npz")
log_file = st.file_uploader("Upload event log (.csv or .nxt)", type=["csv", "nxt"])

# --- Parameters ---
downsample_factor = st.number_input("Downsampling factor", min_value=1, value=100)
cutoff_time = st.number_input("Start time (seconds)", min_value=0.0, value=2.0)
plot_choice = st.radio("Plot Type", ["ΔF/F", "Z-scored ΔF/F"])

# --- Event Display Toggles ---
st.markdown("**Event Overlays**")
show_sucrose = st.checkbox("Show Sucrose Admin (code 1)", value=True)
show_enter = st.checkbox("Show Entering Home Arena (code 6)", value=True)
show_exit = st.checkbox("Show Exiting Foraging Arena (code 4)", value=True)

# --- Run Processing and Plot ---
if st.button("Generate Plot") and npz_file and log_file:
    # --- Load NPZ photometry ---
    with np.load(npz_file) as data:
        signal_raw = data['signal']
        control_raw = data['control']
        time_raw = data['time']

    # --- Handle temp file for event log ---
    suffix = log_file.name.split(".")[-1].lower()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}", mode='wb') as tmp:
            tmp.write(log_file.read())
            tmp_path = tmp.name

        # Load events
        if suffix == "csv":
            events_data = load_data(tmp_path)
            events = parse_ohrbets_serial_log(events_data)
        elif suffix == "nxt":
            events_data = readfile(tmp_path)
            events = get_events(events_data,code_map)
        else:
            st.error("Unsupported file type.")
            events = []

    finally:
        os.remove(tmp_path)

    # --- Downsample ---
    signal = np.array([np.mean(signal_raw[i:i+downsample_factor]) for i in range(0, len(signal_raw), downsample_factor)])
    control = np.array([np.mean(control_raw[i:i+downsample_factor]) for i in range(0, len(control_raw), downsample_factor)])
    time = time_raw[::downsample_factor][:len(signal)]

    # --- ΔF/F calculation ---
    fit = np.polyfit(control, signal, 1)
    fitted = fit[0] * control + fit[1]
    dF = 100 * ((signal - fitted) / fitted)
    dF_z = (dF - np.mean(dF)) / np.std(dF)

    # --- Trim based on cutoff time ---
    idx = np.searchsorted(time, cutoff_time)
    time = time[idx:] - time[idx]
    dF = dF[idx:]
    dF_z = dF_z[idx:]

    # --- Create interactive plot ---
    trace = go.Scatter(
        x=time,
        y=dF if plot_choice == "ΔF/F" else dF_z,
        mode='lines',
        name=plot_choice,
        line=dict(color="blue")
    )

    layout = go.Layout(
        title=dict(text="Fiber Photometry with Behavioral Events", font=dict(color='black')),
        template="plotly_white",
        xaxis=dict(
            title=dict(text ="Time (s)",font=dict(color='black')),
            color='black',
            tickfont=dict(color='black'),
            rangeslider=dict(visible=True)
        ),
        yaxis=dict(
            title=dict(text = "ΔF/F (%)" if plot_choice == "ΔF/F" else "Z-scored ΔF/F",font=dict(color='black')),
            color='black',
            tickfont=dict(color='black'),
        ),
        font=dict(color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        height=500
    )

    fig = go.Figure(data=[trace], layout=layout)

    # --- Add vertical event lines with black annotation text ---
    for event in events:
        t = event.get('timestamp_s')
        code = event.get('code')
        if t is not None and t >= 0:
            if show_sucrose and code == 1:
                fig.add_vline(
                    x=t - 2,
                    line=dict(color='red', dash='dash'),
                    annotation_text="Sucrose",
                    annotation_position="top left",
                    annotation_font=dict(color='black'),
                    annotation=dict(bgcolor='white')
                )
            if show_enter and code == 6:
                fig.add_vline(
                    x=t,
                    line=dict(color='green', dash='dash'),
                    annotation_text="Enter",
                    annotation_position="top left",
                    annotation_font=dict(color='black'),
                    annotation=dict(bgcolor='white')
                )
            if show_exit and code == 4:
                fig.add_vline(
                    x=t,
                    line=dict(color='purple', dash='dash'),
                    annotation_text="Exit",
                    annotation_position="top left",
                    annotation_font=dict(color='black'),
                    annotation=dict(bgcolor='white')
                )

    # --- Show plot ---
    st.plotly_chart(fig, use_container_width=True)



2/2

