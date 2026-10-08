"""Common AFT label contract: observed [t,t], right-censored [t,+inf]."""
import numpy as np


def aft_bounds(duration, event_observed):
    lower = np.asarray(duration, dtype=float)
    observed = np.asarray(event_observed)
    if lower.ndim != 1 or lower.shape != observed.shape:
        raise ValueError("Duration and event_observed must be aligned one-dimensional arrays.")
    if not np.isfinite(lower).all() or (lower <= 0).any():
        raise ValueError("AFT durations must be finite and positive.")
    if not np.isin(observed, [0, 1]).all():
        raise ValueError("event_observed must contain only 0 (censored) or 1 (observed).")
    return lower, np.where(observed == 1, lower, np.inf)
