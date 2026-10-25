"""Build data/real_can.db: REAL CAN-bus attack incidents from the HCRL Car-Hacking recordings (read in place from the zip; nothing copied).
The CanHealth monitor is learned on the clean run, then streamed over the attack recordings; consecutive flagged seconds are merged into incidents,
stored as CAN_ANOMALY events - exactly what the ingest daemon would store live. Usage: python tools_import_hcrl.py [zip]"""
import io, os, re, sys, zipfile, time
from vgpt import store, canhealth
ZIP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "Downloads", "9) Car-Hacking Dataset.zip")
LINE = re.compile(r"Timestamp:\s+([\d.]+)\s+ID:\s+([0-9a-fA-F]+)")
z = zipfile.ZipFile(ZIP); mon = canhealth.CanHealth()
with z.open("normal_run_data/normal_run_data.txt") as f:
    for raw in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
        m = LINE.search(raw)
        if m: mon.learn(float(m[1]), m[2].lower())
mon.finish_learning()
for e in ("", "-wal", "-shm"):
    try: os.remove("data/real_can.db" + e)
    except OSError: pass
c = store.open_rw("data/real_can.db"); events = []; hi = 0
KIND = {"DoS_dataset.csv": "DoS", "Fuzzy_dataset.csv": "Fuzzing", "RPM_dataset.csv": "RPM spoofing", "gear_dataset.csv": "gear spoofing"}
for name, label in KIND.items():
    m2 = canhealth.CanHealth(); m2.base_max = dict(mon.base_max); m2.learning = False; alerts = []
    with z.open(name) as f:
        for raw in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
            p = raw.strip().split(",")
            if len(p) < 4: continue
            try: ts = float(p[0])
            except ValueError: continue
            alerts += m2.feed(ts, p[1].lower())
    alerts += m2.flush(); inc = []
    for (sec, kind, i, n) in alerts:                     # merge alerts within 3 s into one incident
        if inc and sec - inc[-1]["end"] <= 3: inc[-1]["end"] = sec; inc[-1]["n"] += n; inc[-1]["ids"].add(i); inc[-1]["kinds"].add(kind)
        else: inc.append({"start": sec, "end": sec, "n": n, "ids": {i}, "kinds": {kind}})
    for k in inc:
        top = sorted(k["ids"])[:3]; hi = max(hi, k["end"])
        events.append((k["start"], "CAN_ANOMALY", "warn", f"{label}: {'/'.join(sorted(k['kinds']))} on ID {', '.join(top)}{'...' if len(k['ids']) > 3 else ''} for {k['end'] - k['start'] + 1} s ({k['n']} frames above baseline)"))
    print(f"{label}: {len(alerts)} alerts -> {len(inc)} incidents")
c.executemany("INSERT INTO events VALUES(?,?,?,?)", events)
c.execute("INSERT OR REPLACE INTO meta VALUES('end',?)", (str(hi + 3600),)); c.execute("INSERT OR REPLACE INTO meta VALUES('t0','0')")
c.execute("INSERT INTO sig_1s(ts,speed,soc,power,batt,flow,amb,hvac,motor,mode) VALUES(?,0,0,0,0,0,0,0,0,0)", (hi,)); c.close()
print("events stored:", len(events))
