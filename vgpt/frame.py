"""SignalFrame: an aligned block of samples (numpy) for a set of registry concepts over one or more time intervals, plus the provenance of every column.
Built by the executor from the read-only database through compiled, whitelisted SELECTs (see query_planner.compile_select); consumed by derived.py, sensor_quality.py and the aggregations."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    ts: np.ndarray                                   # int64 epoch seconds, ascending
    cols: dict = field(default_factory=dict)         # concept -> float array (NaN = missing sample)
    res: int = 60                                    # seconds per sample (1 = 1 Hz tier, 60 = per-minute tier)
    tier: str = "1m"
    source: dict = field(default_factory=dict)       # concept -> "table.column"
    intervals: list = field(default_factory=list)    # [(start, end)] the samples were loaded from

    def __len__(self): return len(self.ts)

    def mask_between(self, lo, hi): return (self.ts >= lo) & (self.ts < hi)

    def take(self, mask):
        m = np.asarray(mask)
        if m.dtype == bool: m = np.flatnonzero(m)                                                                  # the positions once, not one scan of the mask per column
        f = Frame(self.ts[m], {k: v[m] for k, v in self.cols.items()}, self.res, self.tier, dict(self.source), list(self.intervals)); return f

    def column(self, name):
        v = self.cols.get(name)
        return None if v is None else np.asarray(v, dtype=float)

    @property
    def span(self): return (int(self.ts[0]), int(self.ts[-1]) + self.res) if len(self.ts) else (0, 0)
