"""Many vehicles, one machine: the fleet / validation hub.

Each vehicle streams into its own receiver process (its own pipe, its own database, its own failure domain); a hub starts them, restarts one that dies and reports on all of them.  One receiver is about 40 MB and
decodes tens of thousands of messages a second, so a hub's capacity is its cores: the receivers are independent processes, nothing is shared between vehicles, and the assistant opens any vehicle's database read-only.

    hub = FleetHub(home, slots=16)                      # 16 pipes, 16 receiver processes
    ep  = hub.endpoint(3)                               # the pipe the 4th car connects to
    hub.status()                                        # per vehicle: state, what it was recognised as, records, drops
    hub.close()

`slots=0` sizes the hub from the hardware plan (one receiver per vehicle, bounded by the plan's ingest workers x 4)."""
import multiprocessing as mp
import os
import queue
import threading
import time


def _slot_main(home, endpoint, idx, out, tier):
    """One receiver process: a LinkedVehicle on its own endpoint; reports a status line every second."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from ui.linked import LinkedVehicle
    lv = LinkedVehicle(home, endpoint=endpoint, tier=tier).start(); t = 0
    try:
        while True:
            time.sleep(1.0); t += 1
            try: v = lv.view()
            except Exception as e: v = {"error": repr(e)}
            st = lv.svc.status()
            out.put({"slot": idx, "t": time.time(), "state": st.get("state"), "records": st.get("records", 0), "dropped": st.get("dropped", 0), "vehicle": (st.get("vehicle") or {}).get("model") if isinstance(st.get("vehicle"), dict) else None,
                     "decision": (st.get("decision") or {}).get("mode"), "pid": os.getpid()})
    except KeyboardInterrupt: pass
    finally: lv.close()


class FleetHub:
    def __init__(self, home, slots=4, tier=None, name="fleet", restart=True):
        from ..pipe import transport as PT
        self.home, self.slots, self.tier, self.restart = home, int(slots), tier, restart; self.ctx = mp.get_context("spawn"); self.q = self.ctx.Queue(); self.procs = {}; self.last = {}; self.restarts = {}
        self.endpoints = [PT.endpoint(f"{name}{i}") for i in range(self.slots)]; self._stop = threading.Event()
        for i in range(self.slots): self._start(i)
        self._t = threading.Thread(target=self._pump, daemon=True, name="fleet-hub"); self._t.start()

    def endpoint(self, i): return self.endpoints[i % self.slots]

    def _start(self, i):
        p = self.ctx.Process(target=_slot_main, args=(os.path.join(self.home, f"slot{i}"), self.endpoints[i], i, self.q, self.tier), daemon=True, name=f"fleet-slot-{i}"); p.start(); self.procs[i] = p

    def _pump(self):
        while not self._stop.is_set():
            try: m = self.q.get(timeout=0.5); self.last[m["slot"]] = m
            except queue.Empty: pass
            if self.restart:
                for i, p in list(self.procs.items()):
                    if not p.is_alive() and not self._stop.is_set(): self.restarts[i] = self.restarts.get(i, 0) + 1; self._start(i)

    def status(self):
        now = time.time(); rows = []
        for i in range(self.slots):
            m = self.last.get(i); p = self.procs.get(i)
            rows.append({"slot": i, "alive": bool(p and p.is_alive()), "pid": p.pid if p else None, "restarts": self.restarts.get(i, 0), "state": m["state"] if m else "starting", "vehicle": m["vehicle"] if m else None, "decision": m["decision"] if m else None,
                         "records": m["records"] if m else 0, "dropped": m["dropped"] if m else 0, "age_s": round(now - m["t"], 1) if m else None})
        return rows

    def totals(self):
        rows = self.status(); return {"slots": self.slots, "connected": sum(1 for r in rows if r["state"] == "running"), "records": sum(r["records"] for r in rows), "dropped": sum(r["dropped"] for r in rows), "restarts": sum(r["restarts"] for r in rows)}

    def kill(self, i):
        """For the fault test: end one receiver the hard way."""
        p = self.procs.get(i)
        if p and p.is_alive(): p.kill()

    def close(self):
        self._stop.set()
        for p in self.procs.values():
            try: p.terminate()
            except Exception: pass
        for p in self.procs.values():
            try: p.join(5)
            except Exception: pass
