"""Commands for the two deliberately separate halves of the offline test bench.

``receive`` owns the read-only end of the local vehicle link and writes one
database per detected vehicle.  ``car`` owns only the write end and emits the
selected model's native-looking bus traffic.  Neither side imports the other;
the launcher is merely a convenient place from which to start either process.
"""
import json
import os
import sys
import time


def home(path=None):
    if path:
        return os.path.abspath(path)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~")
    return os.path.join(base, "Vantage", "link")


def list_models(out=print):
    from .car.models import catalogue
    rows = catalogue(home())
    for m in rows:
        out(f"{m['id']:<28} {m.get('name', m['id'])}  [{', '.join(m.get('interfaces', []))}]")
    return rows


def receive(path=None, endpoint=None, tier=None, json_status=False):
    """Run Vantage's collector until interrupted.  A closed car connection is
    followed by a fresh accept, so models can be changed without restarting it.
    """
    from .pipe.receiver import LinkService
    from .pipe.transport import Listener
    root = home(path); listener = Listener(endpoint); svc = LinkService(root, tier=tier)
    print(f"Vantage vehicle link ready: {listener.path}")
    print(f"Vehicle histories: {os.path.join(root, 'vehicles')}")
    try:
        while True:
            fd = listener.accept()
            if fd is None:
                break
            reader = svc.attach(fd)
            while reader.is_alive():
                reader.join(0.5)
                st = svc.status()
                if json_status:
                    print(json.dumps(st, separators=(",", ":")), flush=True)
                elif st.get("vehicle"):
                    v = st["vehicle"]
                    print(f"\r{st['state']}: {v['model']} | {st['records']} records | {v['db']}      ", end="", flush=True)
            if not json_status:
                print("\ncar disconnected; waiting for another car")
    except KeyboardInterrupt:
        pass
    finally:
        listener.close(); svc.stop(); svc.join()
    return 0


def run_car(model_id, dataset=None, endpoint=None, speed=1.0, seed=7, root=None, instant=False):
    """Run only the simulated car.  It cannot read from the link."""
    from .car import models
    from .car.car import Car
    from .car.transport import connect
    from .car.writer import Writer
    m = models.find(model_id, home(root))
    if not m:
        print(f"Unknown car model: {model_id}")
        print("Available models:"); list_models()
        return 2
    if dataset and not os.path.isfile(dataset):
        print(f"Dataset not found: {dataset}")
        return 2
    print(f"Car: {m.get('name', model_id)}")
    print("Waiting for Vantage's local one-way pipe...")
    fd = connect(endpoint, stop=None)
    if fd is None:
        return 2
    if instant and not dataset:
        print("Instant load needs --dataset (a recorded history).")
        return 2
    car = Car(m["dir"], Writer(fd), dataset=dataset, speed=speed, seed=seed, instant=instant)
    car.start()
    try:
        while car.is_alive():
            car.join(1.0)
            s = car.status()
            print(f"\r{s['drive']} | {s['sent']} records | {s['speed_x']:g}x | {s['vin']}      ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        car.stop()
    print()
    if car.error:
        print(car.error)
        return 1
    return 0


def run_aaos(adb="adb", serial=None, endpoint=None, seconds=None, drive=False):
    """The car is an Android Automotive device reached with adb (vgpt/car/aaos.py): its Vehicle HAL values go onto the pipe."""
    from .car.aaos import run
    from .car.transport import connect
    print("Waiting for Vantage's local one-way pipe...")
    fd = connect(endpoint)
    if fd is None: return 2
    try: n = run(fd, adb, serial, seconds, 1.0, drive)
    except KeyboardInterrupt: n = 0
    print(); print(f"{n} property values written")
    return 0


INSTANT = "Instant - load the whole dataset now"
SPEEDS = ["1", "2", "5", "10", "30", "60", "120", "300", "600", INSTANT]


def datasets(root=None):
    """The recorded histories this computer holds, by the car model each belongs to: <repo>/data/world4/<car>/<car>_<n>yr.db, and <home>/datasets/*.db.  [(model id or '', path, megabytes)], the car's own first."""
    import glob
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); out = []; seen = set()
    for pat in (os.path.join(repo, "data", "world4", "*", "*yr.db"), os.path.join(home(root), "datasets", "*.db")):
        for p in sorted(glob.glob(pat)):
            p = os.path.abspath(p)
            if p in seen or os.path.basename(p).startswith("_") or os.path.basename(os.path.dirname(p)).startswith("_"): continue
            seen.add(p); parent = os.path.basename(os.path.dirname(p)); out.append((parent if os.path.dirname(os.path.dirname(p)).endswith("world4") else "", p, os.path.getsize(p) / 1e6))
    return out


def car_selector(root=None):
    """The Car program's window: choose a vehicle and (optionally) a recorded history, set how fast it is sent - or load the whole history into Vantage at once, for testing - start, pause, watch the link, press a
    fault or the ignition.  The car runs in THIS process and knows nothing about Vantage except the local pipe it writes to; the pipe is opened when Start is pressed (Vantage may be started before or after)."""
    import threading
    import time as _t
    import tkinter as tk
    from tkinter import filedialog, ttk
    from .car.models import catalogue
    cars = catalogue(home(root)); sets = datasets(root)
    win = tk.Tk(); win.title("Vantage Car Simulator"); win.geometry("780x640"); win.minsize(740, 600)
    frame = ttk.Frame(win, padding=18); frame.pack(fill="both", expand=True); frame.columnconfigure(1, weight=1); frame.columnconfigure(2, weight=1)
    ttk.Label(frame, text="Car simulator", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
    ttk.Label(frame, text="The car sends its own bus traffic to Vantage over the local one-way pipe. Nothing leaves this computer.", wraplength=720).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 12))
    labels = [f"{m.get('name', m['id'])}  -  {m.get('powertrain', 'vehicle')}, {m.get('region', '')}".rstrip(", ") for m in cars]
    chosen = tk.StringVar(value=labels[0] if labels else "")
    ttk.Label(frame, text="Vehicle").grid(row=2, column=0, sticky="w"); box = ttk.Combobox(frame, textvariable=chosen, values=labels, state="readonly", width=70); box.grid(row=2, column=1, columnspan=3, sticky="we", pady=3)
    notes = tk.StringVar(); ttk.Label(frame, textvariable=notes, wraplength=600, foreground="#555").grid(row=3, column=1, columnspan=3, sticky="w")
    dataset = tk.StringVar(); ttk.Label(frame, text="Recorded history").grid(row=4, column=0, sticky="w")
    dbox = ttk.Combobox(frame, textvariable=dataset, width=62); dbox.grid(row=4, column=1, columnspan=2, sticky="we", pady=3)
    ttk.Button(frame, text="Browse...", command=lambda: dataset.set(filedialog.askopenfilename(filetypes=[("Vehicle history", "*.db *.sqlite"), ("All files", "*.*")]) or dataset.get())).grid(row=4, column=3, sticky="e")
    hint = tk.StringVar(); ttk.Label(frame, textvariable=hint, wraplength=600, foreground="#555").grid(row=5, column=1, columnspan=3, sticky="w")
    speed = tk.StringVar(value="1"); ttk.Label(frame, text="Speed").grid(row=6, column=0, sticky="w")
    sbox = ttk.Combobox(frame, textvariable=speed, values=SPEEDS, width=36, state="readonly"); sbox.grid(row=6, column=1, sticky="w", pady=3)
    ttk.Label(frame, text="times real time (1 = the car's own pace). Changes at once while sending.", wraplength=300, foreground="#555").grid(row=6, column=2, columnspan=2, sticky="w")
    state = {"car": None, "thread": None, "stop": threading.Event(), "link": "stopped"}
    status = tk.StringVar(value="Stopped."); detail = tk.StringVar(); link = tk.StringVar(value="Pipe: not connected"); where = tk.StringVar()
    ttk.Separator(frame).grid(row=7, column=0, columnspan=4, sticky="we", pady=10)
    ttk.Label(frame, textvariable=status, font=("Segoe UI", 12, "bold")).grid(row=8, column=0, columnspan=4, sticky="w")
    bar = ttk.Progressbar(frame, maximum=1000); bar.grid(row=9, column=0, columnspan=4, sticky="we", pady=(6, 2)); ttk.Label(frame, textvariable=where, foreground="#555").grid(row=10, column=0, columnspan=4, sticky="w")
    ttk.Label(frame, textvariable=link, foreground="#555").grid(row=11, column=0, columnspan=4, sticky="w"); ttk.Label(frame, textvariable=detail, wraplength=720, foreground="#555").grid(row=12, column=0, columnspan=4, sticky="w", pady=(0, 8))
    fault = tk.StringVar(); fbox = ttk.Combobox(frame, textvariable=fault, state="readonly", width=44); fbox.grid(row=13, column=1, columnspan=2, sticky="w")
    ttk.Label(frame, text="Situation").grid(row=13, column=0, sticky="w")

    def current():
        i = labels.index(chosen.get()) if chosen.get() in labels else 0
        return cars[i] if cars else None

    def own_sets():
        m = current(); mine = [p for mid, p, mb in sets if m and mid == m["id"]]
        return mine, [p for mid, p, mb in sets if not (m and mid == m["id"])]

    def describe(*_):
        m = current()
        if m: notes.set(", ".join(m.get("interfaces", []))[:300])
        mine, rest = own_sets(); sizes = {p: mb for _, p, mb in sets}
        dbox.configure(values=[""] + mine + rest)
        if mine and (not dataset.get() or dataset.get() not in mine and dataset.get() in sizes): dataset.set(mine[0])
        elif not mine and dataset.get() in sizes: dataset.set("")
        hint.set((f"{len(mine)} history file(s) found for this car; the first one is chosen. Leave empty for a simulated drive." if mine else "No recorded history for this car here: it will drive a simulated route. Browse... for one of your own.") if m else "")
    box.bind("<<ComboboxSelected>>", describe); describe()

    def is_instant(): return speed.get() == INSTANT
    def speed_value():
        try: return 600.0 if is_instant() else float(speed.get())
        except ValueError: return 1.0

    def on_speed(*_):
        c = state["car"]
        if c is not None and not c.instant:
            try: c.set_speed(speed_value())
            except Exception: pass
    sbox.bind("<<ComboboxSelected>>", on_speed)

    def say(msg): win.after(0, status.set, msg)

    def run_car(m):
        from .car.car import Car
        from .car.transport import connect
        from .car.writer import Writer
        state["link"] = "waiting"; say("Waiting for Vantage's pipe...  (start Vantage, or the Vehicle Receiver)")
        fd = connect(None, stop=state["stop"])
        if fd is None: state["link"] = "stopped"; say("Stopped."); return
        ds = dataset.get().strip() or None
        if is_instant() and not (ds and os.path.isfile(ds)): state["link"] = "stopped"; say("Instant load needs a recorded history: choose or browse for one."); return
        try: car = Car(m["dir"], Writer(fd), dataset=ds, speed=speed_value(), instant=is_instant()); car.start()
        except Exception as e: state["link"] = "stopped"; say(f"Could not start: {type(e).__name__}: {e}"); return
        state["car"] = car; state["link"] = "connected"; state["since"] = _t.monotonic()
        from .live.car import FAULTS
        win.after(0, lambda: (fbox.configure(values=[f"{k}: {v[0]}" for k, v in FAULTS.items()] if not ds else []), fbox.set("")))
        while car.is_alive() and not state["stop"].is_set(): _t.sleep(0.3)
        car.stop(); state["car"] = None; state["link"] = "closed by Vantage" if car.link_closed else "stopped"
        say("Stopped." if not car.error else "Stopped: " + car.error)

    def start():
        m = current()
        if not m or (state["thread"] and state["thread"].is_alive()): return
        state["stop"] = threading.Event(); t = threading.Thread(target=run_car, args=(m,), daemon=True); state["thread"] = t; t.start()

    def stop():
        state["stop"].set(); c = state["car"]
        if c is not None: c.stop(join=False)

    pause_text = tk.StringVar(value="Pause")
    def pause():
        c = state["car"]
        if c is not None and c.is_alive(): c.pause(not c.paused); pause_text.set("Resume" if c.paused else "Pause")

    def restart():
        c = state["car"]
        stop(); t = state["thread"]
        if t is not None: win.after(300, lambda: wait_then_start(t))

    def wait_then_start(t):
        if t.is_alive(): win.after(300, lambda: wait_then_start(t)); return
        start()

    def refresh():
        c = state["car"]
        if c is not None and c.is_alive():
            s = c.status(); v = (s.get("state") or {}); lag = s.get("lag_s") or 0.0
            if s["instant"]: status.set(f"Instant load: Vantage was told where the history is; the car now drives on from its end  |  {s['sent']:,} messages")
            else: status.set(f"{'Paused' if s['paused'] else 'Sending'}: {s['sent']:,} messages  |  {s['speed_x']:g}x  |  {s['drive']}")
            prog = v.get("progress") if isinstance(v, dict) else None
            if prog is not None:
                bar.configure(value=int(prog * 1000)); sp = v.get("span") or [0, 0]
                import datetime as _d
                f = lambda x: _d.datetime.fromtimestamp(x).strftime("%d %b %Y %H:%M") if x else "?"
                where.set(f"History {prog * 100:.1f}% played  |  now {f(v.get('dataset_t'))}  |  from {f(sp[0])} to {f(sp[1])}")
            else: bar.configure(value=0); where.set("Simulated drive (no recorded history)")
            link.set(f"Pipe: connected for {int(_t.monotonic() - state.get('since', _t.monotonic()))} s  |  {s['bytes'] / 1024:.0f} KiB written  |  " + ("keeping up" if lag < 1.0 else f"Vantage is the limit: {lag:.0f} s behind"))
            bits = [f"VIN {s['vin']}"] + [f"{k} {n:,}" for k, n in sorted(s["by_medium"].items())]
            if isinstance(v, dict) and v.get("speed") is not None: bits.insert(0, f"{v['speed']:.0f} km/h")
            detail.set("   ".join(bits)); pause_text.set("Resume" if s["paused"] else "Pause")
        else: link.set("Pipe: " + {"waiting": "waiting for Vantage", "closed by Vantage": "closed by Vantage"}.get(state["link"], "not connected"))
        win.after(500, refresh)

    def inject():
        c = state["car"]
        if c is not None and fault.get():
            try: c.inject(fault.get().split(":")[0])
            except Exception as e: status.set(f"Cannot apply: {e}")

    def clear():
        c = state["car"]
        if c is not None:
            try: c.clear(None)
            except Exception: pass

    def ignition(on):
        c = state["car"]
        if c is not None:
            try: c.control("ignition", on=on)
            except Exception as e: status.set(f"Cannot change: {e}")

    row = ttk.Frame(frame); row.grid(row=14, column=0, columnspan=4, sticky="we", pady=10)
    ttk.Button(row, text="Start car", command=start).pack(side="left"); ttk.Button(row, textvariable=pause_text, command=pause).pack(side="left", padx=6); ttk.Button(row, text="Stop car", command=stop).pack(side="left"); ttk.Button(row, text="Restart", command=restart).pack(side="left", padx=6)
    row2 = ttk.Frame(frame); row2.grid(row=15, column=0, columnspan=4, sticky="we")
    ttk.Button(row2, text="Apply situation", command=inject).pack(side="left"); ttk.Button(row2, text="Clear", command=clear).pack(side="left", padx=6)
    ttk.Button(row2, text="Ignition on", command=lambda: ignition(True)).pack(side="left", padx=(24, 0)); ttk.Button(row2, text="Ignition off", command=lambda: ignition(False)).pack(side="left", padx=6)

    def close():
        stop(); win.destroy()
    win.protocol("WM_DELETE_WINDOW", close); refresh(); win.mainloop(); return 0
