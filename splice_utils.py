"""
Signal splice utilities for fiber photometry data.

Splice files are plain-text files stored inside each TDT block folder.
Each non-comment line contains: start_seconds,end_seconds

The masking approach:
  - Spliced samples are EXCLUDED from the polyfit regression so the fit
    isn't biased by artifacts.
  - The fitted line and ΔF/F are still computed for ALL samples (time axis
    unchanged), so event timestamps don't need shifting.
  - Peri-event snippets whose windows overlap a splice region are skipped
    entirely, preventing contaminated trials.
"""

import os
import numpy as np

SPLICE_FILENAME = "splices.txt"


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_splices(block_path: str) -> list:
    """Load splice definitions from a TDT block folder.

    Returns a sorted list of (start_seconds, end_seconds) tuples.
    Returns an empty list if no splice file exists.
    """
    splice_file = os.path.join(block_path, SPLICE_FILENAME)
    splices = []
    if not os.path.exists(splice_file):
        return splices
    with open(splice_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    start = float(parts[0].strip())
                    end = float(parts[1].strip())
                    if end > start >= 0:
                        splices.append((start, end))
                except ValueError:
                    continue
    splices.sort(key=lambda x: x[0])
    return splices


def save_splices(block_path: str, splices: list):
    """Write splice definitions into the TDT block folder."""
    splice_file = os.path.join(block_path, SPLICE_FILENAME)
    with open(splice_file, "w") as f:
        f.write(
            f"# Signal splice definitions for block: "
            f"{os.path.basename(block_path)}\n"
        )
        f.write("# Each line: start_seconds,end_seconds\n")
        f.write("# Lines starting with # are comments\n")
        for start, end in sorted(splices, key=lambda x: x[0]):
            f.write(f"{start},{end}\n")


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def get_splice_mask(n_samples: int, fs: float, splices: list) -> np.ndarray:
    """Return a boolean mask (True = keep, False = spliced out).

    Parameters
    ----------
    n_samples : int
        Length of the signal array.
    fs : float
        Sampling rate in Hz.
    splices : list of (start_s, end_s)
        Splice regions in seconds.
    """
    mask = np.ones(n_samples, dtype=bool)
    for start_s, end_s in splices:
        si = max(0, int(np.floor(start_s * fs)))
        ei = min(n_samples, int(np.ceil(end_s * fs)))
        if si < ei:
            mask[si:ei] = False
    return mask


def check_snippet_overlaps_splice(
    event_time: float,
    pre_t: float,
    post_t: float,
    splices: list,
) -> bool:
    """Return True if the peri-event window [event-pre, event+post]
    overlaps any splice region."""
    win_start = event_time - pre_t
    win_end = event_time + post_t
    for s_start, s_end in splices:
        if s_start < win_end and s_end > win_start:
            return True
    return False


def offset_splices(splices: list, offset: float) -> list:
    """Shift splice times by subtracting *offset* (raw → aligned).

    Splices that fall entirely before time 0 after shifting are dropped.
    Splices that partially overlap time 0 are clipped.
    """
    adjusted = []
    for start, end in splices:
        a_start = start - offset
        a_end = end - offset
        if a_end <= 0:
            continue
        a_start = max(0.0, a_start)
        adjusted.append((a_start, a_end))
    return adjusted
