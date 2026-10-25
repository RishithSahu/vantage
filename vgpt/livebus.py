"""The live bus: what is arriving at this moment, straight from the ingest, before it is stored.

    data from the vehicle's sensors  ->  VANTAGE  ->  the dataset (always)  and  a direct answer (when the person asks about the present)

Everything that arrives is stored.  But a value that is arriving this second is worth more than the same value read back from the dataset a moment later: the dataset is behind by the second still being
written, a bus value is as fresh as its message.  So when somebody asks about the PRESENT - "what is my speed", "how hot is the battery", "what are the tyre pressures" - the answer is taken from the incoming
stream.  When the instant asked about is more than `WINDOW_S` (20 s) older than the live moment, or the question is about a stretch of time (a trip, an average, a count, history), the dataset answers and
nothing here is consulted.  Only what is genuinely a live reading is offered live: a plain signal the bus carries (never an estimate, a total or a derived figure).

    bus = LiveBus(runtime);  vgpt.live_snapshot.LIVE_BUS = bus       (the application attaches it while a source is delivering)
    bus.value("speed", now)  ->  {"value": 74.0, "ts": 1792364339.6, "age_s": 0.4}  or None"""
import time

WINDOW_S = 20.0                                    # an instant older than this is a question for the dataset
MAX_AGE_S = 5.0                                    # a reading older than this is not "now" any more (slow signals have their own, longer, validity from the profile)


class LiveBus:
    def __init__(self, runtime): self.rt = runtime; self.hits = 0; self.t_hit = None                                                  # hits: readings handed out (the dashboard reads the bus too)

    # ------------------------------------------------------------------------------------------------ the live moment
    def moment(self):
        """The timestamp of the newest message that has arrived (None before the first)."""
        return self.rt.ts or None

    def covers(self, ts):
        """True when the instant `ts` is the live moment or at most WINDOW_S seconds before it - the only instants this bus answers."""
        m = self.moment()
        return m is not None and m - WINDOW_S <= float(ts) <= m + 2.0

    # ------------------------------------------------------------------------------------------------ readings
    def _fresh(self, concept):
        item = self.rt.ing.latest.get(concept)
        if item is None: return None
        v, ts = item; m = self.rt.ts or ts
        limit = max(MAX_AGE_S, float(getattr(self.rt.ing, "ttl_over", {}).get(concept, 0.0)))
        return (float(v), float(ts)) if 0 <= m - ts <= limit else None

    def value(self, concept, now=None):
        """The live reading of one concept, or None (not on this vehicle's buses, not fresh, or `now` is not the live moment)."""
        if now is not None and not self.covers(now): return None
        with self.rt.lock:
            r = self._fresh(concept) if concept not in ("battery_power", "drive_mode") else self._derived(concept)
        if r is None: return None
        self.hits += 1; self.t_hit = time.time(); return {"value": r[0], "ts": r[1], "age_s": round(max(0.0, (self.rt.ts or r[1]) - r[1]), 2)}

    def _derived(self, concept):
        """The two core readings the ingest itself combines from several signals: battery power from voltage x current, the drive mode from ignition / charging / speed."""
        if concept == "battery_power":
            v, a = self._fresh("battery_voltage"), self._fresh("battery_current")
            return None if v is None or a is None else (v[0] * a[0] / 1000.0, min(v[1], a[1]))
        ign, chg, spd = self._fresh("_ignition"), self._fresh("_charging_active"), self._fresh("speed")
        if chg is not None and chg[0] >= 0.5: return (3.0, chg[1])
        if ign is not None and ign[0] < 0.5: return (0.0, ign[1])
        if ign is not None and spd is not None: return (2.0 if spd[0] > 75 else 1.0, min(ign[1], spd[1]))
        return None

    def values(self, concepts, now=None):
        """{concept: reading} for the concepts that are live now."""
        out = {}
        for c in concepts:
            r = self.value(c, now)
            if r is not None: out[c] = r
        return out

    def status(self):
        m = self.moment()
        return {"live_moment": m, "window_s": WINDOW_S, "reads": self.hits, "signals": len(self.rt.ing.latest)}
