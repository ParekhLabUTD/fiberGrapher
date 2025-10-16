import os

import glob

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
 
# ----------------------------

# Parameters to configure

# ----------------------------

data_dir = "data"  # folder with your CSV and event files

fs = 20            # target sampling rate in Hz

window = (-5, 10)  # peri-event window in seconds

rolling_baseline_s = 60  # baseline window (seconds) for ΔF/F

export_file = "photometry_summary.xlsx"
 
# ----------------------------

# Helper functions

# ----------------------------
 
def preprocess_signals(df, fs, rolling_baseline_s=60):

    """Correct motion artifact (405 regression), then compute ΔF/F."""

    reg = LinearRegression().fit(df[["signal_405"]], df["signal_470"])

    fitted = reg.predict(df[["signal_405"]])

    corrected = df["signal_470"] - fitted
 
    win = int(rolling_baseline_s * fs)

    baseline = pd.Series(corrected).rolling(win, center=True, min_periods=1).median()

    dff = (corrected - baseline) / baseline

    df["dff"] = dff

    return df
 
def extract_peri_event(df, event_times, window=(-5, 10), fs=20):

    """Extract peri-event windows aligned to each event."""

    n_samples = int((window[1] - window[0]) * fs)

    peri_traces = []

    for et in event_times:

        start, end = et + window[0], et + window[1]

        mask = (df["time"] >= start) & (df["time"] <= end)

        trace = df.loc[mask, "dff"].values

        if len(trace) > 0:

            trace = np.interp(

                np.linspace(0, len(trace)-1, n_samples),

                np.arange(len(trace)),

                trace

            )

            peri_traces.append(trace)

    return np.array(peri_traces)
 
def compute_metrics(peri_traces, fs, window=(0, 5)):

    """Compute peak, AUC, and latency metrics for each peri-event trace."""

    n_samples = peri_traces.shape[1]

    t = np.linspace(-5, 10, n_samples)

    mask = (t >= window[0]) & (t <= window[1])

    peaks, aucs, latencies = [], [], []

    for trace in peri_traces:

        post = trace[mask]

        peaks.append(post.max())

        aucs.append(np.trapz(post, dx=1/fs))

        latencies.append(t[mask][np.argmax(post)])

    return peaks, aucs, latencies
 
# ----------------------------

# Main loop over sessions

# ----------------------------

results = []

metrics_rows = []
 
for csv_file in glob.glob(os.path.join(data_dir, "*.csv")):

    base = os.path.splitext(os.path.basename(csv_file))[0]

    event_file = os.path.join(data_dir, base + "_events.txt")

    if not os.path.exists(event_file):

        print(f"Skipping {csv_file} (no event file found).")

        continue

    df = pd.read_csv(csv_file)

    fs_est = 1.0 / np.median(np.diff(df["time"]))

    print(f"{base}: estimated fs = {fs_est:.2f} Hz")

    df = preprocess_signals(df, fs=int(fs_est), rolling_baseline_s=rolling_baseline_s)

    event_times = np.loadtxt(event_file).tolist()

    if isinstance(event_times, float):

        event_times = [event_times]

    peri_traces = extract_peri_event(df, event_times, window=window, fs=fs)

    if peri_traces.size:

        avg_trace = peri_traces.mean(axis=0)

        results.append({"session": base, "peri_traces": peri_traces, "avg_trace": avg_trace})

        # Compute metrics

        peaks, aucs, latencies = compute_metrics(peri_traces, fs, window=(0, 5))

        for i, et in enumerate(event_times[:len(peaks)]):

            metrics_rows.append({

                "session": base,

                "event_time": et,

                "peak": peaks[i],

                "auc": aucs[i],

                "latency": latencies[i]

            })

        # Plot session

        t = np.linspace(window[0], window[1], peri_traces.shape[1])

        mean_trace = avg_trace

        sem_trace = peri_traces.std(axis=0) / np.sqrt(peri_traces.shape[0])

        plt.figure(figsize=(6,4))

        plt.plot(t, mean_trace, color="blue")

        plt.fill_between(t, mean_trace - sem_trace, mean_trace + sem_trace,

                         color="blue", alpha=0.3)

        plt.axvline(0, color="red", linestyle="--")

        plt.xlabel("Time (s)")

        plt.ylabel("ΔF/F")

        plt.title(f"{base} (n={peri_traces.shape[0]} events)")

        plt.tight_layout()

        plt.show()
 
print("Finished processing all sessions.")
 
# ----------------------------

# Export to Excel

# ----------------------------

if metrics_rows:

    metrics_df = pd.DataFrame(metrics_rows)

    with pd.ExcelWriter(export_file) as writer:

        metrics_df.to_excel(writer, sheet_name="EventMetrics", index=False)

        for r in results:

            peri_df = pd.DataFrame(r["avg_trace"]).T

            peri_df.to_excel(writer, sheet_name=r["session"], index=False)

    print(f"Exported results to {export_file}")

 