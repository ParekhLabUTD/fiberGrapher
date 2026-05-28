"""
Shared math and signal-processing utilities for fiber photometry analysis.

Every function here was extracted from inline code in graphApp.py (Tabs 2/3)
and batchProcessing.py (Tab 4).  The order of operations and numerical
behaviour is preserved exactly.

Reference: Lerner et al. 2015 (block-averaging downsampling).
"""

import math
import numpy as np

from splice_utils import get_splice_mask


# ---------------------------------------------------------------------------
# 1. Downsampling
# ---------------------------------------------------------------------------

def downsample_block_average(signal, factor):
    """Block-averaging downsample (Lerner et al. 2015).

    Each output sample is the mean of ``factor`` consecutive input samples.

    Parameters
    ----------
    signal : np.ndarray
        1-D input array.
    factor : int
        Downsample factor (must be >= 1).

    Returns
    -------
    np.ndarray
        Downsampled 1-D array.
    """
    if factor <= 1:
        return signal
    return np.array([
        np.mean(signal[i:i + factor])
        for i in range(0, len(signal), factor)
    ])


# ---------------------------------------------------------------------------
# 2. Length matching
# ---------------------------------------------------------------------------

def match_lengths(*arrays):
    """Truncate all arrays to the length of the shortest.

    Parameters
    ----------
    *arrays : np.ndarray
        One or more 1-D arrays.

    Returns
    -------
    tuple of np.ndarray
        Truncated arrays, all with the same length.
    """
    min_len = min(len(a) for a in arrays)
    return tuple(a[:min_len] for a in arrays)


# ---------------------------------------------------------------------------
# 3. ΔF/F calculation
# ---------------------------------------------------------------------------

def compute_dff(signal, control, splice_mask=None, epsilon=0.0):
    """Compute ΔF/F using polyfit regression.

    Steps (order preserved from original inline code):
      1. ``polyfit(control, signal, 1)`` — optionally restricted to
         non-spliced samples when *splice_mask* is provided.
      2. ``fitted = p[0] * control + p[1]`` — computed for ALL samples.
      3. ``dff = 100 * (signal - fitted) / (fitted + epsilon)``

    Parameters
    ----------
    signal : np.ndarray
        1-D signal (e.g. 465 nm).
    control : np.ndarray
        1-D control (e.g. 415 nm), same length as *signal*.
    splice_mask : np.ndarray or None
        Boolean mask (True = keep) to exclude spliced samples from the
        polyfit.  If None, all samples are used.
    epsilon : float
        Small constant added to the denominator to prevent division by zero.
        Use ``0.0`` for Tab 2/3 behaviour, ``1e-12`` for Tab 4 behaviour.

    Returns
    -------
    dff : np.ndarray
        ΔF/F in percent (same length as input).
    fitted : np.ndarray
        Fitted control line (same length as input).
    coefficients : np.ndarray
        Polyfit coefficients ``[slope, intercept]``.
    """
    if splice_mask is not None:
        coefficients = np.polyfit(control[splice_mask], signal[splice_mask], 1)
    else:
        coefficients = np.polyfit(control, signal, 1)

    fitted = coefficients[0] * control + coefficients[1]
    dff = 100.0 * (signal - fitted) / (fitted + epsilon)
    return dff, fitted, coefficients


# ---------------------------------------------------------------------------
# 4. Z-scoring
# ---------------------------------------------------------------------------

def zscore_global(dff, splice_mask=None):
    """Global z-score computed from the entire trace.

    Mean and std are computed only from non-spliced samples (if a mask is
    provided), but the z-score is applied to ALL samples.

    Used by Tab 2 and Tab 3 (global mode).

    Parameters
    ----------
    dff : np.ndarray
        Full ΔF/F trace.
    splice_mask : np.ndarray or None
        Boolean mask (True = keep).  If None, all samples are used.

    Returns
    -------
    np.ndarray
        Z-scored trace (same shape as *dff*).
    """
    if splice_mask is not None:
        clean = dff[splice_mask]
    else:
        clean = dff
    mu = np.mean(clean)
    sigma = np.std(clean)
    return (dff - mu) / (sigma if sigma > 0 else 1.0)


def zscore_baseline(snippet, baseline_start_idx, baseline_end_idx):
    """Per-event baseline z-score.

    Used by Tab 3 (baseline mode).

    Parameters
    ----------
    snippet : np.ndarray
        1-D peri-event snippet (already extracted from the dFF trace).
    baseline_start_idx : int
        Start index of the baseline window within the snippet.
    baseline_end_idx : int
        End index of the baseline window within the snippet.

    Returns
    -------
    z_snippet : np.ndarray
        Z-scored (or mean-subtracted) snippet.
    was_zscored : bool
        True if sigma > 0 and a proper z-score was applied.
    """
    baseline_start_idx = max(0, baseline_start_idx)
    baseline_end_idx = min(len(snippet), baseline_end_idx)

    if baseline_end_idx > baseline_start_idx:
        baseline_vals = snippet[baseline_start_idx:baseline_end_idx]
        mu = float(np.mean(baseline_vals))
        sigma = float(np.std(baseline_vals))
        if sigma > 0:
            return (snippet - mu) / sigma, True
        else:
            return snippet - mu, False
    else:
        # Degenerate window — return unchanged
        return snippet.copy(), False


def zscore_baseline_with_fallback(snippet, baseline_start_idx, baseline_end_idx,
                                  fallback_end_idx=None, global_mean=None):
    """Per-event baseline z-score with a three-tier fallback cascade.

    Used by Tab 4 (batchProcessing.py).

    Cascade:
      1. Use the requested baseline window.
      2. If that window is empty, fall back to ``snippet[:fallback_end_idx]``
         (the pre-event portion).
      3. If that is also empty, subtract *global_mean* only (no z-scoring).

    Parameters
    ----------
    snippet : np.ndarray
        1-D peri-event snippet.
    baseline_start_idx : int
        Start index of the baseline window within the snippet.
    baseline_end_idx : int
        End index of the baseline window within the snippet.
    fallback_end_idx : int or None
        End index for the pre-event fallback window (``snippet[:fallback_end_idx]``).
    global_mean : float or None
        Global dFF mean used as last-resort subtraction.

    Returns
    -------
    z_snippet : np.ndarray
        Z-scored / mean-subtracted snippet.
    was_zscored : bool
        True if sigma > 0 and a proper z-score was applied.
    """
    baseline_start_idx = max(0, baseline_start_idx)
    baseline_end_idx = min(len(snippet), baseline_end_idx)

    if baseline_end_idx > baseline_start_idx:
        baseline_vals = snippet[baseline_start_idx:baseline_end_idx]
        mu = float(np.mean(baseline_vals))
        sigma = float(np.std(baseline_vals))
        if sigma > 0:
            return (snippet - mu) / sigma, True
        else:
            return snippet - mu, False

    # Fallback 1: pre-event portion
    if fallback_end_idx is not None and fallback_end_idx > 0:
        fallback_vals = snippet[:fallback_end_idx]
        mu = float(np.mean(fallback_vals))
        sigma = float(np.std(fallback_vals))
        if sigma > 0:
            return (snippet - mu) / sigma, True
        else:
            return snippet - mu, False

    # Fallback 2: global mean subtraction
    if global_mean is not None:
        return snippet - global_mean, False

    # Nothing to do
    return snippet.copy(), False


def zscore_pooled_baseline(snippet, pooled_mean, pooled_std):
    """Z-score a snippet using externally-computed pooled baseline statistics.

    All trials in a session share the same mean and std, computed from the
    concatenation of every trial's baseline window.

    Parameters
    ----------
    snippet : np.ndarray
        1-D peri-event snippet (raw dF/F).
    pooled_mean : float
        Mean of all concatenated baseline windows.
    pooled_std : float
        Std of all concatenated baseline windows.

    Returns
    -------
    z_snippet : np.ndarray
        Z-scored snippet.
    was_zscored : bool
        True if pooled_std > 0 and a proper z-score was applied.
    """
    if pooled_std > 0:
        return (snippet - pooled_mean) / pooled_std, True
    else:
        return snippet - pooled_mean, False


# ---------------------------------------------------------------------------
# 5. SEM calculation
# ---------------------------------------------------------------------------

def compute_sem(traces_array):
    """Compute the standard error of the mean across trials.

    Parameters
    ----------
    traces_array : np.ndarray
        2-D array of shape ``(n_trials, n_timepoints)``.

    Returns
    -------
    np.ndarray
        1-D SEM array of shape ``(n_timepoints,)``.
    """
    return traces_array.std(axis=0) / math.sqrt(traces_array.shape[0])


# ---------------------------------------------------------------------------
# 6. PCT alignment
# ---------------------------------------------------------------------------

def compute_pct_offset(pct_onset_time, code12_times, code12_index=1):
    """Compute the time offset for PCT alignment.

    ``offset = pct_onset_time - code12_times[code12_index]``

    Parameters
    ----------
    pct_onset_time : float
        PCT onset timestamp (from TDT epocs).
    code12_times : list of float
        Timestamps of code-12 events.
    code12_index : int
        Which code-12 event to use (default: 1, i.e. the second one).

    Returns
    -------
    float
        Offset in seconds.
    """
    return pct_onset_time - code12_times[code12_index]


def apply_pct_alignment(signal, control, fs, offset):
    """Trim signal and control to remove samples before t=0 after PCT offset.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal array.
    control : np.ndarray
        1-D control array (same length as *signal*).
    fs : float
        Sampling rate in Hz.
    offset : float
        PCT offset in seconds (subtracted from the time axis).

    Returns
    -------
    signal_trimmed : np.ndarray
    control_trimmed : np.ndarray
    """
    time_full = np.arange(len(signal)) / fs
    time_full -= offset
    valid = time_full >= 0
    return signal[valid], control[valid]


# ---------------------------------------------------------------------------
# 7. Code 0 cutoff
# ---------------------------------------------------------------------------

def apply_code0_cutoff(signal, control, fs, code0_time):
    """Truncate signal and control at code-0 timestamp.

    Parameters
    ----------
    signal : np.ndarray
    control : np.ndarray
    fs : float
        Sampling rate in Hz.
    code0_time : float
        Timestamp of the first code-0 event.

    Returns
    -------
    signal_truncated : np.ndarray
    control_truncated : np.ndarray
    """
    time_arr = np.arange(len(signal)) / fs
    keep = time_arr <= code0_time
    return signal[keep], control[keep]


def apply_code0_cutoff_with_time(time_arr, signal, control, code0_time):
    """Truncate signal, control, and a pre-computed time array at code-0.

    Used by Tab 2's ``compute_trace`` where the time array is already built.

    Parameters
    ----------
    time_arr : np.ndarray
    signal : np.ndarray
    control : np.ndarray
    code0_time : float

    Returns
    -------
    time_truncated : np.ndarray
    signal_truncated : np.ndarray
    control_truncated : np.ndarray
    """
    keep = time_arr <= code0_time
    return time_arr[keep], signal[keep], control[keep]


# ---------------------------------------------------------------------------
# 8. Peri-event extraction
# ---------------------------------------------------------------------------

def compute_peri_time_vector(pre_time, post_time, fs):
    """Generate the peri-event time axis.

    ``peri_times = (arange(n_samples) / fs) - pre_time``

    Parameters
    ----------
    pre_time : float
        Seconds before event onset.
    post_time : float
        Seconds after event onset.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    peri_times : np.ndarray
        1-D time vector in seconds.
    n_samples : int
        Total number of samples in the peri-event window.
    """
    n_samples = int(round((pre_time + post_time) * fs))
    peri_times = (np.arange(n_samples) / fs) - pre_time
    return peri_times, n_samples


def extract_peri_snippet(dff, event_time, pre_time, post_time, fs, n_samples=None):
    """Extract a peri-event snippet from the full dFF trace.

    Parameters
    ----------
    dff : np.ndarray
        Full ΔF/F trace.
    event_time : float
        Event onset timestamp in seconds.
    pre_time : float
        Seconds before event.
    post_time : float
        Seconds after event.
    fs : float
        Sampling rate.
    n_samples : int or None
        Expected snippet length.  If None, computed from pre/post/fs.

    Returns
    -------
    snippet : np.ndarray or None
        Peri-event snippet, or None if out of bounds.
    start_idx : int
        Start index in *dff*.
    end_idx : int
        End index in *dff*.
    """
    if n_samples is None:
        n_samples = int(round((pre_time + post_time) * fs))

    start_idx = int(round((event_time - pre_time) * fs))
    end_idx = start_idx + n_samples

    if start_idx < 0 or end_idx > len(dff):
        return None, start_idx, end_idx

    snippet = dff[start_idx:end_idx].astype(float)
    if len(snippet) != n_samples:
        return None, start_idx, end_idx

    return snippet, start_idx, end_idx


# ---------------------------------------------------------------------------
# 9. Trial metrics
# ---------------------------------------------------------------------------

def compute_trial_metrics(snippet_z, fs, metric_start, metric_end, pre_time,
                          detect_inhibitory=True):
    """Compute peak, AUC, and latency for a z-scored peri-event snippet.

    Parameters
    ----------
    snippet_z : np.ndarray
        Z-scored peri-event snippet.
    fs : float
        Sampling rate.
    metric_start : float
        Start of the metric window relative to event onset (seconds).
    metric_end : float
        End of the metric window relative to event onset (seconds).
    pre_time : float
        Pre-event time used to offset into the snippet array.
    detect_inhibitory : bool
        If True (batchProcessing.py default), detect whether the response is
        excitatory or inhibitory by comparing abs(max) vs abs(min).
        If False, always use np.max (excitatory-only).

    Returns
    -------
    dict
        ``{"peak": float, "auc": float, "latency": float}``
        All values are ``np.nan`` if the metric window is empty.
    """
    mstart_idx = int(round((pre_time + metric_start) * fs))
    mend_idx = int(round((pre_time + metric_end) * fs))
    mstart_idx = max(0, mstart_idx)
    mend_idx = min(len(snippet_z), mend_idx)

    if mend_idx <= mstart_idx:
        return {"peak": np.nan, "auc": np.nan, "latency": np.nan}

    post_segment = snippet_z[mstart_idx:mend_idx]
    auc = float(np.trapz(post_segment, dx=1.0 / fs))

    if detect_inhibitory:
        # Detect excitatory vs inhibitory response:
        # use whichever extreme has the larger absolute value
        seg_max = float(np.max(post_segment))
        seg_min = float(np.min(post_segment))
        if abs(seg_min) > abs(seg_max):
            # Inhibitory: nadir is the dominant peak
            peak = seg_min
            latency_idx = int(np.argmin(post_segment))
        else:
            # Excitatory (or flat): max is the dominant peak
            peak = seg_max
            latency_idx = int(np.argmax(post_segment))
    else:
        peak = float(np.max(post_segment))
        latency_idx = int(np.argmax(post_segment))

    latency = float(latency_idx / fs + metric_start)

    return {"peak": peak, "auc": auc, "latency": latency}
