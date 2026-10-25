"""CAN signal-level anomaly monitor (stdlib, O(#IDs x #signals) memory - a few KB). Learns from CLEAN decoded traffic, then flags:
  UNKNOWN_ID  an arbitration id never seen in the clean recording
  RANGE       a signal outside its clean min/max (+ small margin)
  JUMP        a step between consecutive values much larger than any clean step
  FROZEN      a signal that normally moves stuck on one value far longer than it ever was in clean data (plateau / replay-of-constant)
  FAST        a message arriving much sooner than the clean minimum interval (flooding / injection)
  GAP         a message missing for much longer than the clean maximum interval (suppression) - reported when the next one arrives
Input rows are (time_seconds, can_id, [signal values]) - exactly what the DBC decoder produces. It is a monitor: it never transmits or blocks."""
import math
from collections import defaultdict


class _Id:
    __slots__ = ("n", "lo", "hi", "step", "run", "max_run", "last", "last_t", "dts", "min_gap", "max_gap", "distinct", "cur_run", "med")

    def __init__(self, k):
        self.n = 0; self.lo = [math.inf] * k; self.hi = [-math.inf] * k; self.step = [0.0] * k; self.max_run = [0] * k; self.cur_run = [0] * k
        self.last = [None] * k; self.last_t = None; self.dts = []; self.min_gap = self.max_gap = self.med = None; self.distinct = [set() for _ in range(k)]


class SignalHealth:
    def __init__(self, margin=0.03, jump_factor=2.0, run_factor=3.0, fast_factor=0.4, gap_factor=3.0):
        self.ids, self.margin, self.jf, self.rf, self.ff, self.gf = {}, margin, jump_factor, run_factor, fast_factor, gap_factor
        self.frozen = False

    # ---- learning -------------------------------------------------------------
    def learn(self, t, cid, vals):
        s = self.ids.get(cid)
        if s is None: s = self.ids[cid] = _Id(len(vals))
        s.n += 1
        if s.last_t is not None and t > s.last_t:
            if len(s.dts) < 20000: s.dts.append(t - s.last_t)
        s.last_t = t
        for j, v in enumerate(vals):
            if v is None or v != v: continue
            if v < s.lo[j]: s.lo[j] = v
            if v > s.hi[j]: s.hi[j] = v
            if len(s.distinct[j]) < 8: s.distinct[j].add(v)
            p = s.last[j]
            if p is not None:
                d = abs(v - p)
                if d > s.step[j]: s.step[j] = d
                if v == p:
                    s.cur_run[j] += 1
                    if s.cur_run[j] > s.max_run[j]: s.max_run[j] = s.cur_run[j]
                else: s.cur_run[j] = 0
            s.last[j] = v

    def finish(self):
        for s in self.ids.values():
            d = sorted(s.dts)
            if d:
                s.min_gap = d[max(0, int(0.01 * len(d)))]; s.max_gap = d[min(len(d) - 1, int(0.999 * len(d)))]; s.med = d[len(d) // 2]
            s.last = [None] * len(s.last); s.last_t = None; s.cur_run = [0] * len(s.cur_run); s.dts = []
        self.frozen = True

    # ---- monitoring -----------------------------------------------------------
    def check(self, t, cid, vals):
        """-> list of (kind, detail) for this message (empty = normal)."""
        s = self.ids.get(cid)
        if s is None: return [("UNKNOWN_ID", cid)]
        out = []
        if s.last_t is not None and s.med:
            dt = t - s.last_t
            if s.min_gap is not None and dt < self.ff * s.min_gap and dt >= 0: out.append(("FAST", round(dt, 6)))
            elif s.max_gap is not None and dt > max(self.gf * s.max_gap, 0.05): out.append(("GAP", round(dt, 4)))
        s.last_t = t
        for j, v in enumerate(vals):
            if v is None or v != v or j >= len(s.lo) or s.lo[j] == math.inf: continue
            span = s.hi[j] - s.lo[j]
            m = self.margin * span + (1e-9 if span else 1e-6)
            if v < s.lo[j] - m or v > s.hi[j] + m: out.append(("RANGE", j))
            p = s.last[j]
            moving = len(s.distinct[j]) >= 3
            if p is not None and moving:
                if abs(v - p) > self.jf * s.step[j] + m: out.append(("JUMP", j))
                if v == p:
                    s.cur_run[j] += 1
                    if s.cur_run[j] > max(self.rf * s.max_run[j], s.max_run[j] + 20): out.append(("FROZEN", j))
                else: s.cur_run[j] = 0
            s.last[j] = v
        return out

    def reset_stream(self):
        for s in self.ids.values(): s.last = [None] * len(s.last); s.last_t = None; s.cur_run = [0] * len(s.cur_run)
