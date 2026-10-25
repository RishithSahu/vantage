"""The car on the other end of the local pipe, as the application sees it.

Vantage listens on the one-way pipe (vgpt/pipe/transport.py), the receiver (vgpt/pipe/receiver.py) works out what the car is and stores what it sends in a database of that vehicle, and this class hands the
application three things: which database to read, what time the data has reached, and what to say about the link.  The application never writes to that database and never talks to the car.

    LinkedVehicle(home).start()      begin listening (a car may connect at any time, and another may replace it)
    .seq()                           changes whenever a different vehicle has been recognised
    .vehicle()                       {key, id, model, db, how, vin} of the vehicle now recognised, or the one stored last time, or None
    .now()                           the time the stored data reaches (the car's own clock)
    .view()                          what the data-link chip shows
"""
import glob
import os
import sqlite3
import threading
import time


class LinkedVehicle:
    def __init__(self, home, endpoint=None, tier=None):
        from vgpt.pipe.receiver import LinkService
        from vgpt.pipe.transport import Listener
        self.home = home; self.svc = LinkService(home, tier=tier); self.listener = Listener(endpoint); self._stop = threading.Event(); self._t = None
        self.errors = 0; self._last_seen = None; self._stored = self._latest_stored()

    # ------------------------------------------------------------------------------------------------ listening
    def start(self):
        self._t = threading.Thread(target=self._accept_loop, daemon=True, name="vantage-link"); self._t.start(); return self

    def _accept_loop(self):
        while not self._stop.is_set():
            try: fd = self.listener.accept()
            except OSError: self.errors += 1; time.sleep(1.0); continue
            if fd is None: break
            try: reader = self.svc.attach(fd)
            except Exception: self.errors += 1; continue
            while reader.is_alive() and not self._stop.is_set(): reader.join(0.5)          # one car at a time: a FIFO that already has a writer lets a second reader in at once, so the next car is waited for only after this one has gone

    def close(self):
        self._stop.set()
        try: self.listener.close()
        except Exception: pass
        self.svc.stop()

    # ------------------------------------------------------------------------------------------------ what the application reads
    def _latest_stored(self):
        """The newest vehicle database an earlier session left (what is stored stays available without the car)."""
        best = None
        for p in glob.glob(os.path.join(self.home, "vehicles", "*", "vantage.db")):
            try: m = os.path.getmtime(p)
            except OSError: continue
            if best is None or m > best[0]: best = (m, p)
        return best[1] if best else None

    def seq(self):
        return self.svc.vehicle_seq

    def vehicle(self):
        v = self.svc.status().get("vehicle")
        if v: return v
        if self._stored:
            d = os.path.dirname(self._stored); name = os.path.basename(d)
            return {"key": name, "id": name, "model": name.split("-")[0].replace("_", " "), "db": self._stored, "how": "stored", "vin": None}
        return None

    def connected(self):
        s = self.svc.status(); return s.get("state") in ("running", "listening", "unidentified") and not s.get("closed")

    def now(self):
        s = self.svc.session
        if s is not None and s.last_ts: self._last_seen = s.last_ts; return s.last_ts
        if self._last_seen: return self._last_seen
        v = self.vehicle()
        if v:
            try:
                c = sqlite3.connect("file:" + v["db"].replace("\\", "/") + "?mode=ro", uri=True)
                try: r = max((x for x in (c.execute("SELECT MAX(ts) FROM sig_1s").fetchone()[0], c.execute("SELECT MAX(ts) FROM sig_1m").fetchone()[0]) if x), default=None)      # an instantly loaded history has minute rows only
                finally: c.close()
                if r: self._last_seen = float(r); return self._last_seen
            except sqlite3.Error: pass
        return time.time()

    def runtime(self): return self.svc.runtime()
    def picture(self): return self.svc.picture()

    def view(self):
        """The data-link chip: waiting for a car, the car's traffic, or the car gone."""
        s = self.svc.status(); state = s.get("state"); v = s.get("vehicle")
        if v is None:
            if state in ("listening", "unidentified"): return {"kind": "wire", "waiting": True, "t": 0, "lag_s": 0, "speed": 1.0, "error": None, "state": state, "records": s.get("records", 0),
                                                              "vantage": {"connected": True, "car": "recognising", "age_s": None}, "sources": [], "decision": (s.get("decision") or {}).get("mode"), "import": s.get("import"), "inventory": s.get("inventory")}
            return {"kind": "wire", "waiting": self.vehicle() is None, "offline": self.vehicle() is not None, "t": 0, "lag_s": 0, "speed": 1.0, "error": None, "state": state or "waiting", "vantage": {"connected": False, "car": "", "age_s": None}, "sources": []}
        age = time.time() - (self.svc.session.last_ts or 0) if (self.svc.session and self.svc.session.last_ts and self.svc.session.last_ts > 1e9) else None
        out = {"kind": "wire", "t": s.get("time", {}).get("last"), "lag_s": 0, "speed": 1.0, "error": None, "msgs": s.get("records", 0), "state": state, "model": v["model"], "decision": (s.get("decision") or {}).get("mode"),
               "sources": [{"id": c["channel"], "name": f"channel {c['channel']} ({c['kind']})", "type": c["kind"], "state": "ok", "frames": c["blocks"]} for c in s.get("channels", [])], "dropped": s.get("dropped", 0),
               "vantage": {"connected": state == "running", "car": v["model"], "age_s": age}, "import": s.get("import"), "inventory": s.get("inventory"), "how": s.get("how") or (s.get("decision") or {}).get("mode"), "why": [str(x)[:200] for x in ((s.get("decision") or {}).get("why") or [])[:8]]}
        if state in ("disconnected",) or s.get("closed"): out["offline"] = True
        return out

    def camera_state(self):
        n = self.svc.status().get("pictures", 0)
        return {"state": "OK" if n else "CALIBRATING", "reason": "", "cond": "", "blocked_frac": 0.0, "sharp_ratio": 1.0, "luma": 0, "lane_conf": 0.0, "low_light": False, "lost": None, "fps": 0.0, "ms": 0.0,
                "source": "the car's camera over the link" if n else "no camera picture from the car yet", "fault": "none", "error": ""}


class LinkPipe:
    """What `App.pipe` is while a car is linked: the same few members the data feed of a recorded history has (kind, status, rt, stop), answered from the link."""
    kind = "wire"

    def __init__(self, linked): self.linked = linked
    @property
    def rt(self): return self.linked.runtime()
    def status(self): return self.linked.view()
    def now(self): return self.linked.now()
    def stop(self): pass
