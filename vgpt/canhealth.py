"""CAN bus-health monitor (stdlib, O(#IDs) memory). Works on RAW frames of any bus, no DBC needed - so it also covers IDs the DBC-lite adapter
does not decode. Learns each arbitration ID's normal per-second rate from a clean recording, then flags 1 s windows with
  * UNKNOWN_ID  an ID never seen in the baseline (typical of fuzzing / diagnostic probing)
  * RATE_SPIKE  an ID far above its baseline peak rate (typical of DoS / flooding / replay-injection of a legitimate ID)
Emits CAN_ANOMALY events for the store. It is a monitor, not a gateway: it never transmits and never blocks a frame.
"""
from collections import Counter, defaultdict


class CanHealth:
    def __init__(self, spike_factor=2.0, spike_floor=3, min_unknown=1):
        self.base_max, self.base_n = {}, 0
        self.k, self.floor, self.min_unknown = spike_factor, spike_floor, min_unknown
        self.cur, self.sec = Counter(), None
        self.learning = True; self.hist = defaultdict(list)

    # ---- learning (clean traffic) -------------------------------------------------
    def learn(self, ts, can_id):
        s = int(ts)
        if self.sec is None: self.sec = s
        if s != self.sec:
            self._close_learn()
            self.sec = s
        self.cur[can_id] += 1

    def _close_learn(self):
        for i, n in self.cur.items(): self.hist[i].append(n)
        self.cur = Counter(); self.base_n += 1

    def finish_learning(self):
        if self.cur: self._close_learn()
        for i, v in self.hist.items(): self.base_max[i] = max(v)
        self.hist = defaultdict(list); self.cur, self.sec, self.learning = Counter(), None, False

    # ---- monitoring ---------------------------------------------------------------
    def feed(self, ts, can_id):
        """Returns a list of (second, kind, can_id, count) for the window that just closed, else []."""
        s = int(ts); out = []
        if self.sec is None: self.sec = s
        if s != self.sec:
            out = self._close(); self.sec = s
        self.cur[can_id] += 1
        return out

    def _close(self):
        out = []
        for i, n in self.cur.items():
            b = self.base_max.get(i)
            if b is None:
                if n >= self.min_unknown: out.append((self.sec, "UNKNOWN_ID", i, n))
            elif n > max(self.k * b, b + self.floor): out.append((self.sec, "RATE_SPIKE", i, n))
        self.cur = Counter(); return out

    def flush(self): return self._close() if self.cur else []
