# Full Math Audit & Data Flow Documentation

## Pipeline 1: Batch Processing ([batchProcessing.py](file:///d:/Aryan/fiberGrapher/batchProcessing.py))

### Raw Data → dF/F

```mermaid
graph TD
    A["TDT Block (raw)"] --> B["Read streams: sig = _465A/C, ctrl = _415A/C"]
    B --> C["Downsample: sig[::ds], ctrl[::ds], fs = fs_orig / ds"]
    C --> D["Truncate to equal length: min(len sig, len ctrl)"]
    D --> E["Linear regression: p = polyfit(ctrl, sig, 1)"]
    E --> F["Fitted control: fitted = p[0] * ctrl + p[1]"]
    F --> G["dF/F = 100 × (sig − fitted) / (fitted + 1e-12)"]
```

### dF/F → Per-Trial Z-Scored Snippets

```mermaid
graph TD
    G["dF/F (full session)"] --> H["For each event timestamp ts:"]
    H --> I["Extract snippet: dff[ts−pre : ts+post]"]
    I --> J{"Baseline window indices valid?"}
    J -->|yes| K["baseline_vals = snippet[bl_start : bl_end]"]
    J -->|no| K2["Fallback: use snippet[0 : pre_t] or global mean"]
    K --> L["μ = mean(baseline_vals), σ = std(baseline_vals)"]
    L --> M{"σ > 0?"}
    M -->|yes| N["snippet_z = (snippet − μ) / σ"]
    M -->|no| N2["snippet_z = snippet − μ (not z-scored)"]
    K2 --> L
```

> [!NOTE]
> **Baseline z-scoring**: `std()` uses `ddof=0` (NumPy default = population std). Each trial is independently z-scored against its own baseline window.

### Per-Trial Metrics

```mermaid
graph TD
    N["snippet_z (per trial)"] --> O["metric_segment = snippet_z[metric_start : metric_end]"]
    O --> P["peak = max(metric_segment)"]
    O --> Q["auc = trapz(metric_segment, dx=1/fs)"]
    O --> R["latency = argmax(metric_segment) / fs + metric_start"]
```

> [!WARNING]
> **AUC uses `dx=1/fs`** (uniform spacing), NOT `np.trapz(y, x)` with actual time values. This is correct IF the snippet is uniformly sampled (which it is after downsampling). However, it means AUC units are **signal_z × seconds**.

---

### Output File Flowcharts

#### 1. Session Prism CSV (`prism_tables/sessions/{mouse}_{session}_prism.csv`)

```mermaid
graph LR
    A["All valid trial snippets_z"] --> B["peri_arr = vstack(snippets)"]
    B --> C["session_mean = mean(peri_arr, axis=0)"]
    B --> D["session_sem = std(peri_arr, axis=0) / √n_trials"]
    C --> E["CSV: time_s, mean, sem"]
    D --> E
```

**Columns**: `time_s`, [mean](file:///d:/Aryan/fiberGrapher/advanced_graphing.py#481-528), `sem` — the session-averaged peri-event trace.

#### 2. Session Metrics CSV (`mice/{mouse}/{session}_metrics.csv`)

```mermaid
graph LR
    A["Per-trial: peak, auc, latency"] --> B["DataFrame of all trials"]
    B --> C["CSV: group, mouse, session, trial_index, peak, auc, latency, n_timepoints, zscored, fs, signalChannelSet"]
```

**1 row per trial** with: peak, auc (trapz), latency, zscored flag, fs, signalChannelSet.

#### 3. LONG Traces CSV (`mice/{mouse}/{session}_peri_event_traces_LONG.csv`)

```mermaid
graph LR
    A["peri_arr: trials × timepoints"] --> B["Reshape to long format"]
    B --> C["CSV: trial_index, global_time_s, time_idx, time_s, signal_z, mouse, group, session, fs, signalChannelSet"]
```

**1 row per timepoint per trial**. The `signal_z` column contains z-scored dF/F values.

#### 4. Session Average Plot (`mice/{mouse}/{session}_session_avg.png`)

```mermaid
graph LR
    A["session_mean ± session_sem"] --> B["Line plot with SEM shading"]
    B --> C["Vertical line at t=0"]
```

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
> **This pools individual trials, not session means.** If Mouse1 has 10 trials from Session1 and 5 from Session2, the mouse mean is computed across all 15 trials — Session1 carries double the weight.

#### 6. Group Summary Metrics CSV (`groups/{group}/{group}_summary_metrics.csv`)

```mermaid
graph LR
    A["Per-mouse, per-group trial metrics"] --> B["mean_peak = mean of all trial peaks for this mouse"]
    A --> C["mean_auc = mean of all trial AUCs for this mouse"]
    A --> D["mean_latency = mean of all trial latencies"]
    B --> E["CSV: group, mouse, mean_peak, mean_auc, mean_latency, n_trials"]
    C --> E
    D --> E
```

**1 row per mouse** in the group. Metrics are means across all trials (not session means).

#### 7. Group Mean Trace Plot (`groups/{group}/{group}_mean_trace.png`)

```mermaid
graph LR
    A["Per-mouse mean traces (trial-pooled)"] --> B["arr = vstack(mouse_means)"]
    B --> C["group_mean = mean(arr, axis=0) — mean across mice"]
    B --> D["group_sem = std(arr, axis=0) / √n_mice"]
    C --> E["Line plot: group mean ± SEM"]
    A --> F["Overlay: each mouse mean trace"]
    E --> G["Combined plot"]
    F --> G
```

> [!IMPORTANT]
> **Group mean is mean of mouse means** (each mouse weighted equally), which is correct for between-subjects analysis.

#### 8. Group Prism CSV (`prism_tables/groups/{group}_group_mean_prism.csv`)

Same data as plot #7: `time_s`, [mean](file:///d:/Aryan/fiberGrapher/advanced_graphing.py#481-528), `sem` — group-averaged trace.

#### 9. Group Comparison Plot (`comparison/{event}_group_comparison.png`)

```mermaid
graph LR
    A["Group means + SEMs from each group"] --> B["Overlay all groups on one plot"]
    B --> C["Each group: mean ± SEM shading, different color"]
```

#### 10. Group Comparison Prism CSV (`prism_tables/comparison/{event}_group_comparison_prism.csv`)

**Columns**: `time_s, {group1}_mean, {group1}_sem, {group2}_mean, {group2}_sem, ...`

#### 11. Master Trial Metrics CSV (`all_groups_trial_metrics.csv`)

```mermaid
graph LR
    A["All trial metric rows from all sessions"] --> B["Concatenated DataFrame"]
    B --> C["CSV: group, mouse, session, trial_index, peak, auc, latency, ..."]
```

**1 row per trial across ALL mice and sessions.**

#### 12. Master LONG Traces CSV (`all_groups_peri_event_traces_LONG.csv`)

All individual session LONG DataFrames concatenated. **1 row per timepoint per trial across everything.**

---

## Pipeline 2: Advanced Graphing ([advanced_graphing.py](file:///d:/Aryan/fiberGrapher/advanced_graphing.py))

This pipeline reads CSVs from Pipeline 1 and generates additional visualizations.

### Input Data Sources

```mermaid
graph TD
    P["Prism CSV: time_s, mean, sem"] --> PS["Session mean trace"]
    L["LONG CSV: trial_index, time_s, signal_z, ..."] --> LT["Trial-level data"]
    PS --> F{"Has LONG CSV?"}
    LT --> F
    F -->|yes| T["Use trial-level stats (n = n_trials)"]
    F -->|no| FB["Fallback: session mean only (n = 1)"]
```

#### Signal Mean Bar Plot (`*_signal_mean_bars.png`)

```mermaid
graph TD
    A["LONG CSV trial data"] --> B["For each trial: mean(signal_z) in baseline window"]
    A --> C["For each trial: mean(signal_z) in response window"]
    B --> D["baseline_stats: mean, std, n of per-trial means"]
    C --> E["response_stats: mean, std, n of per-trial means"]
    D --> F["Bar chart: Baseline vs Response ± stddev"]
    E --> F
```

**Math**: `trial_means[i] = mean(signal_z[t ∈ window])` per trial → `mean = mean(trial_means)`, `stddev = std(trial_means, ddof=0)`

#### AUC Bar Plot (`*_auc_bars.png`)

```mermaid
graph TD
    A["LONG CSV trial data"] --> B["For each trial: trapz(signal_z, time_s) in baseline window"]
    A --> C["For each trial: trapz(signal_z, time_s) in response window"]
    B --> D["baseline_auc: mean, std, n of per-trial AUCs"]
    C --> E["response_auc: mean, std, n of per-trial AUCs"]
    D --> F["Bar chart: Baseline vs Response AUC ± stddev"]
    E --> F
```

**Math**: `trial_aucs[i] = trapz(signal_z, time_s)` per trial → `mean = mean(trial_aucs)`, `stddev = std(trial_aucs, ddof=0)`

> [!NOTE]
> Unlike batchProcessing which uses `dx=1/fs`, advanced_graphing uses `np.trapz(y, x)` with actual time values. Both are correct but use slightly different integration methods.

#### Heatmap (`*_heatmap.png`)

```mermaid
graph TD
    A["LONG CSV trial data"] --> B["Pivot: rows=trial_index, cols=time_idx, values=signal_z"]
    B --> C["Color limits: ±percentile(abs(values), 98th)"]
    C --> D["imshow: RdBu_r colormap, symmetric scaling"]
    D --> E["Vertical line at t=0"]
```

#### Session Stats CSV (`*_stats.csv`)

```mermaid
graph LR
    A["baseline_signal + response_signal"] --> B["Row 1: baseline — signal_mean, signal_stddev, signal_n"]
    A --> C["Row 2: response — signal_mean, signal_stddev, signal_n"]
    D["baseline_auc + response_auc"] --> E["+ auc_mean, auc_stddev, auc_n"]
    B --> F["CSV with metadata: mouse, session, group, signal_type, windows"]
    C --> F
    E --> F
```

#### Aggregated Stats CSV (`aggregated_session_stats.csv`)

```mermaid
graph TD
    A["All session stats dicts"] --> B["1 row per session: baseline/response signal_mean, stddev, n + auc_mean, stddev, n"]
    B --> C["+ MEAN_ACROSS_SESSIONS summary row"]
    B --> D["+ STDDEV_ACROSS_SESSIONS summary row"]
    C --> E["CSV"]
    D --> E
```

> [!WARNING]
> The summary rows compute [mean()](file:///d:/Aryan/fiberGrapher/advanced_graphing.py#481-528)/`std()` of the per-session **n** column too, which is meaningless (it averages the trial counts). Consider dropping `n` columns from the summary computation.

---

## Issues Found

### 1. Group summary metrics source data mismatch

In [batchProcessing.py](file:///d:/Aryan/fiberGrapher/batchProcessing.py) lines 534–540, the `per_group_mouse_metrics_rows` are built from `per_mouse_summary_metrics_by_group`, which stores the **per-trial** metric rows (peak, auc, latency). These metrics come from the **z-scored snippet** computed per-trial during batch processing.

The user requested these metrics come from the **prism data** instead. Currently the prism CSV only stores `time_s, mean, sem` (the session-averaged trace) — it doesn't contain per-trial metrics like peak/auc/latency.

**To use prism data for summary metrics**: We could compute peak/auc/latency from the prism session mean trace, but this would give one value per session (not per trial). This changes the meaning of `n_trials` in the summary.

### 2. `std(ddof=0)` used throughout

Both pipelines use population standard deviation. This is consistent but worth noting — some scientific conventions prefer `ddof=1` (sample std) or SEM.

### 3. Mouse-level pooling weights trials, not sessions

When computing mouse-mean traces, all trials are pooled equally. A session with 20 trials has 4× the weight of a session with 5 trials. This is a design choice, not a bug — but different from computing session means first, then averaging (which weights sessions equally).
