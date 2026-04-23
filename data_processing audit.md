# Full Data Audit — Every Step the Data Goes Through

> [!NOTE]
> Last updated: 2026-04-23. Reflects per-channel signal splicing across all tabs.

---

## Splice System Overview

Splice definitions are stored per-channel inside each TDT block folder:

| File | Purpose |
|---|---|
| `splices_ch1.txt` | Channel 1 splice regions |
| `splices_ch2.txt` | Channel 2 splice regions |
| `splices.txt` | Legacy fallback (treated as Channel 1) |

Each line: `start_seconds,end_seconds` (raw TDT block time, pre-alignment).

```mermaid
graph LR
    RAW["splices_chN.txt (raw TDT seconds)"] --> OFF["offset_splices: subtract alignment offset"]
    OFF --> DISP["display_splices (graph coordinates)"]
    DISP --> MASK["get_splice_mask → boolean array (True=keep)"]
    DISP --> OVERLAP["check_snippet_overlaps_splice → skip trial?"]
```

---

## Pipeline 1: Tab 2 — Graph Viewer (`graphApp.py` `compute_trace`)

### Step-by-step data flow

```mermaid
graph TD
    A["TDT Block (raw arrays)"] --> A1["signal = extracted_data.signal{1|2}"]
    A --> A2["control = extracted_data.control{1|2}"]
    A --> A3["fs = extracted_data.fs"]
    A1 --> B["Pre-downsample clamp: min_len = min(len sig, len ctrl)"]
    A2 --> B
    B --> C["Downsample by averaging blocks of N samples:<br/>signal_ds[i] = mean(signal[i*N : (i+1)*N])<br/>control_ds[i] = mean(control[i*N : (i+1)*N])"]
    C --> D["time = arange(len signal_ds) / fs * N"]
    D --> E["PCT alignment: time -= offset<br/>Keep only time ≥ 0 (trim start)"]
    E --> F["Truncate signal_ds, control_ds, time to equal length"]
```

### Regression + ΔF/F + Z-score

```mermaid
graph TD
    F["Aligned, downsampled arrays"] --> G{"Channel has splices?"}
    G -->|yes| H["smask = get_splice_mask(len, ds_fs, display_splices)<br/>ds_fs = fs / downsample_factor"]
    G -->|no| I["Use all samples"]
    H --> J["p = polyfit(control_ds[smask], signal_ds[smask], 1)"]
    I --> J2["p = polyfit(control_ds, signal_ds, 1)"]
    J --> K["fitted = p[0] × control_ds + p[1]<br/>(computed for ALL samples)"]
    J2 --> K
    K --> L["dF = 100 × (signal_ds − fitted) / fitted<br/>(computed for ALL samples — spliced values meaningless)"]
    L --> M{"Has splices?"}
    M -->|yes| N["dF_clean = dF[smask] (non-spliced only)<br/>μ = mean(dF_clean), σ = std(dF_clean)"]
    M -->|no| N2["μ = mean(dF), σ = std(dF)"]
    N --> O["dF_z = (dF − μ) / σ  (applied to ALL samples)"]
    N2 --> O
```

> [!IMPORTANT]
> **polyfit** uses only non-spliced samples → clean regression line.
> **ΔF/F** is computed for ALL samples (spliced values exist but are meaningless).
> **Z-score mean/std** are computed only from non-spliced ΔF/F values.
> **Spliced samples are removed** in Graph 2 and exports (post-computation).

### Graph 1: Full Trace

- Plots `time` vs `dF` or `dF_z` for each channel
- Red `vrect` overlays for Channel 1 splices, orange for Channel 2
- Dashed event lines overlaid

### Graph 2: Spliced Trace

```mermaid
graph LR
    A["dF / dF_z (all samples)"] --> B["keep = get_splice_mask(len, ds_fs, ch_splices)"]
    B --> C["time_clean = time[keep]<br/>dF_clean = dF[keep]<br/>dF_z_clean = dF_z[keep]"]
    C --> D["Plot cleaned trace"]
```

### Exports (CSV / NPZ)

Same masking as Graph 2. Each channel uses its own splice definitions.

| Column | Content |
|---|---|
| `time_s` | Timepoints with spliced samples removed |
| `dF_F_pct` | ΔF/F (%) at each kept timepoint |
| `dF_F_zscore` | Z-scored ΔF/F at each kept timepoint |

---

## Pipeline 2: Tab 3 — Peri-Event Plots (`graphApp.py`)

### Raw Data → ΔF/F

```mermaid
graph TD
    A["signal = extracted_data.signal{signal_set}<br/>control = extracted_data.control{signal_set}<br/>fs = extracted_data.fs"]
    A --> B{"PCT alignment enabled?"}
    B -->|yes| C["offset = pct_onset − code12_times[1]<br/>Trim signal/control where time < 0"]
    B -->|no| C2["No trimming"]
    C --> D["Downsample: signal[::ds], control[::ds], fs /= ds"]
    C2 --> D
    D --> E["Truncate to min(len signal, len control)"]
    E --> F["Load splices: load_splices(block, channel=signal_set)<br/>Adjust: offset_splices(raw, pct_offset)"]
    F --> G{"Has splices?"}
    G -->|yes| H["mask = get_splice_mask(min_len, fs, splices)<br/>fit = polyfit(control[mask], signal[mask], 1)"]
    G -->|no| I["fit = polyfit(control, signal, 1)"]
    H --> J["fitted = fit[0] × control + fit[1]<br/>dFF = 100 × (signal − fitted) / fitted"]
    I --> J
```

### ΔF/F → Per-Event Snippets

```mermaid
graph TD
    J["dFF (full trace, all samples)"] --> K["For each event matching event_name:"]
    K --> L{"Peri-event window overlaps a splice?"}
    L -->|yes| SKIP["SKIP this trial entirely"]
    L -->|no| M["snippet = dFF[event − pre : event + post]"]
    M --> N{"Baseline mode?"}
    N -->|yes| O["baseline_vals = dFF[event+lowerBound : event+upperBound]<br/>μ = mean(baseline_vals), σ = std(baseline_vals)<br/>snippet = (snippet − μ) / σ"]
    N -->|no| P{"Has splices?"}
    P -->|yes| Q["dFF_clean = dFF[mask] (non-spliced only)<br/>μ = mean(dFF_clean), σ = std(dFF_clean)"]
    P -->|no| Q2["μ = mean(dFF), σ = std(dFF)"]
    Q --> R["snippet = (snippet − μ) / σ"]
    Q2 --> R
    O --> S["Append to snippets"]
    R --> S
```

> [!IMPORTANT]
> **Global z-score** mean/std exclude spliced samples from the full trace.
> **Baseline z-score** uses each trial's own baseline window (trials overlapping splices are already skipped, so baseline windows are clean).

### Final Plot

```mermaid
graph LR
    S["snippets array (n_trials × n_timepoints)"] --> T["mean_snip = mean(snippets, axis=0)"]
    S --> U["std_snip = std(snippets, axis=0)"]
    T --> V["Plot: mean ± std, individual traces in gray"]
    U --> V
```

---

## Pipeline 3: Tab 4 — Batch Processing (`batchProcessing.py`)

### Raw Data → ΔF/F

```mermaid
graph TD
    A["TDT Block (raw)"] --> B["scs = meta.signalChannelSet (1 or 2)"]
    B --> C["scs=1: sig=_465A, ctrl=_415A<br/>scs=2: sig=_465C, ctrl=_415C"]
    C --> D["Read streams: sig, ctrl, fs_orig"]
    D --> E{"interpretor == 2?<br/>(PCT alignment)"}
    E -->|yes| F["offset = pct_onset − code12_times[1]<br/>Trim signal/control where time < 0<br/>Shift event_times by −offset"]
    E -->|no| F2["offset = 0.0"]
    F --> G["Load splices: load_splices(block_path, channel=scs)"]
    F2 --> G
    G --> H["block_splices = offset_splices(raw_splices, offset)"]
    H --> I["Downsample: sig[::ds], ctrl[::ds], fs = fs_orig / ds"]
    I --> J["Truncate to min(len sig, len ctrl)"]
    J --> K{"Has splices?"}
    K -->|yes| L["splice_mask = get_splice_mask(min_len, fs, block_splices)<br/>p = polyfit(ctrl[splice_mask], sig[splice_mask], 1)"]
    K -->|no| M["p = polyfit(ctrl, sig, 1)"]
    L --> N["fitted = p[0] × ctrl + p[1]<br/>dff = 100 × (sig − fitted) / (fitted + 1e-12)"]
    M --> N
```

> [!NOTE]
> The `1e-12` epsilon in the denominator prevents division by zero. Tab 2 and Tab 3 do NOT use this epsilon — they divide by `fitted` directly.

### ΔF/F → Per-Trial Z-Scored Snippets

```mermaid
graph TD
    N["dff (full session)"] --> O["For each event timestamp ts:"]
    O --> P{"start_idx < 0 OR end_idx > len(dff)?"}
    P -->|yes| SKIP1["Skip: out of bounds"]
    P -->|no| Q{"Peri-event window overlaps a splice region?"}
    Q -->|yes| SKIP2["Skip: splice overlap"]
    Q -->|no| R["snippet = dff[ts−pre : ts+post]"]
    R --> S["baseline_vals = snippet[bl_start_rel : bl_end_rel]"]
    S --> T["μ = mean(baseline_vals), σ = std(baseline_vals)"]
    T --> U{"σ > 0?"}
    U -->|yes| V["snippet_z = (snippet − μ) / σ"]
    U -->|no| V2["snippet_z = snippet − μ (not z-scored)"]
```

> [!IMPORTANT]
> Batch processing always uses **per-trial baseline z-scoring** (never global). Trials overlapping splice regions are skipped entirely, so baseline windows cannot be contaminated by spliced data.

### Per-Trial Metrics

```mermaid
graph TD
    V["snippet_z (per trial)"] --> W["metric_segment = snippet_z[metric_start : metric_end]"]
    W --> X["peak = max(metric_segment)"]
    W --> Y["auc = trapz(metric_segment, dx=1/fs)"]
    W --> Z["latency = argmax(metric_segment) / fs + metric_start"]
```

> [!WARNING]
> **AUC uses `dx=1/fs`** (uniform spacing), NOT `np.trapz(y, x)` with time values. Units are **signal_z × seconds**.

---

### Batch Output Files

#### 1. Session Prism CSV (`prism_tables/sessions/{mouse}_{session}_prism.csv`)

```mermaid
graph LR
    A["All valid trial snippets_z"] --> B["peri_arr = vstack(snippets)"]
    B --> C["session_mean = mean(peri_arr, axis=0)"]
    B --> D["session_sem = std(peri_arr, axis=0) / √n_trials"]
    C --> E["CSV: time_s, mean, sem"]
    D --> E
```

#### 2. Session Metrics CSV (`mice/{mouse}/{session}_metrics.csv`)

**1 row per valid trial**: group, mouse, session, trial_index, peak, auc, latency, n_timepoints, zscored, fs, signalChannelSet.

#### 3. LONG Traces CSV (`mice/{mouse}/{session}_peri_event_traces_LONG.csv`)

**1 row per timepoint per trial**: trial_index, global_time_s, time_idx, time_s, signal_z, mouse, group, session, fs, signalChannelSet.

#### 4. Session Average Plot (`mice/{mouse}/{session}_session_avg.png`)

Line plot: session_mean ± session_sem with vertical line at t=0.

#### 5. Mouse Prism CSV (`prism_tables/mice/{mouse}_combined_prism.csv`)

```mermaid
graph LR
    A["ALL trials across ALL sessions for this mouse"] --> B["trials_arr = vstack(all_trials)"]
    B --> C["mouse_mean = mean(trials_arr, axis=0)"]
    B --> D["mouse_sem = std(trials_arr, axis=0) / √n_total_trials"]
    C --> E["CSV: time_s, mean, sem"]
    D --> E
```

> [!IMPORTANT]
> Pools individual trials, not session means. Sessions with more valid trials carry more weight.

#### 6. Group Summary Metrics CSV (`groups/{group}/{group}_summary_metrics.csv`)

**1 row per mouse**: mean_peak, mean_auc, mean_latency, n_trials — means across all that mouse's trials.

#### 7. Group Mean Trace Plot (`groups/{group}/{group}_mean_trace.png`)

```mermaid
graph LR
    A["Per-mouse mean traces"] --> B["arr = vstack(mouse_means)"]
    B --> C["group_mean = mean(arr, axis=0) — mean across mice"]
    B --> D["group_sem = std(arr, axis=0) / √n_mice"]
    C --> E["Plot: group mean ± SEM + individual mouse traces"]
    D --> E
```

> [!IMPORTANT]
> Group mean = mean of mouse means (each mouse weighted equally).

#### 8. Group Prism CSV (`prism_tables/groups/{group}_group_mean_prism.csv`)

Same data as plot #7: `time_s`, `mean`, `sem`.

#### 9. Group Comparison Plot + CSV

All group means overlaid. CSV columns: `time_s, {group1}_mean, {group1}_sem, ...`

#### 10. Master CSVs

- `all_groups_trial_metrics.csv` — 1 row per trial across everything
- `all_groups_peri_event_traces_LONG.csv` — all LONG DataFrames concatenated

---

## Pipeline 4: Tab 5 — Advanced Graphing (`advanced_graphing.py`)

Operates entirely on CSV outputs from Pipeline 3. **No raw TDT data is re-read.** Splices were already applied during batch processing, so all data is inherently clean.

```mermaid
graph TD
    P["Prism CSV: time_s, mean, sem"] --> PS["Session mean trace"]
    L["LONG CSV: trial_index, time_s, signal_z, ..."] --> LT["Trial-level data"]
    PS --> F{"Has LONG CSV?"}
    LT --> F
    F -->|yes| T["Use trial-level stats (n = n_trials)"]
    F -->|no| FB["Fallback: session mean only (n = 1)"]
```

#### Signal Mean Bar Plot

**Math**: For each trial → `mean(signal_z)` in baseline/response window → `mean(trial_means)`, `stddev(trial_means, ddof=0)`.

#### AUC Bar Plot

**Math**: For each trial → `trapz(signal_z, time_s)` in window → `mean(trial_aucs)`, `stddev(trial_aucs, ddof=0)`.

> [!NOTE]
> Uses `np.trapz(y, x)` with actual time values (vs batch processing which uses `dx=1/fs`).

#### Heatmap

Pivot trial data → matrix (rows=trials, cols=timepoints). Color limits: `±percentile(|values|, 98th)`. Colormap: `RdBu_r`.

---

## Known Design Decisions

| Item | Behavior |
|---|---|
| `std(ddof=0)` | Population std used everywhere (NumPy default) |
| Mouse pooling | Trials pooled, not session means — unequal trial counts cause unequal weighting |
| Splice masking vs deletion | Polyfit excludes spliced samples; ΔF/F computed for all; spliced samples removed post-hoc for graphs/exports |
| ΔF/F denominator | Batch uses `fitted + 1e-12`; Tabs 2/3 use `fitted` directly |
| AUC method | Batch: `trapz(y, dx=1/fs)`; Advanced: `trapz(y, x)` |
