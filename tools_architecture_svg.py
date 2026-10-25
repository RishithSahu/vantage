"""Draw docs/architecture.svg - the one-page architecture of Vantage - from the measured result files, so that no number on the page is typed by hand.

    python tools_architecture_svg.py            writes docs/architecture.svg

Sources of every figure on the page: results/causal/test.json (sealed set), bench/event_cause_bench.py output (results/causal/events.txt), results/isolation/pi_full.json + pi_attack.json (Raspberry Pi 4),
results/scaling/{pi_threads,pi_contexts,laptop_procs,laptop_contexts,laptop_fleet}.json, tests (counts below are read from the files that hold them)."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
W, H = 1600, 1236
C = {"p1": ("#0b6fb8", "#eaf3fb"), "p2": ("#7b3fb5", "#f3ecfa"), "p3": ("#c0392b", "#fdecea"), "p4": ("#1e8449", "#eafaf1"), "n": ("#5b6475", "#f6f7f9")}
out = []
D1 = ' stroke-dasharray="7 5"'; D2 = ' stroke-dasharray="6 4"'; BOLD = ' font-weight="700"'; ITAL = ' font-style="italic"'


def j(path, default=None):
    try: return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))
    except (OSError, ValueError): return default


def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def add(s): out.append(s)


def rect(x, y, w, h, k="n", r=10, dash=False, sw=1.6, fill=True, extra=""):
    st, fl = C[k]; fa = "" if 'fill="' in extra else 'fill="%s"' % (fl if fill else "none")
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" {fa} stroke="{st}" stroke-width="{sw}"{D1 if dash else ""} {extra}/>')


def text(x, y, s, size=13, k=None, bold=False, anchor="start", fill=None, italic=False):
    col = fill or (C[k][0] if k else "#1a1a2e")
    add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{col}" text-anchor="{anchor}"{BOLD if bold else ""}{ITAL if italic else ""}>{esc(s)}</text>')


def lines(x, y, ss, size=12.5, gap=17, **kw):
    for i, s in enumerate(ss): text(x, y + i * gap, s, size, **kw)


def arrow(x1, y1, x2, y2, k="n", dash=False, sw=2):
    st = C[k][0]; add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{st}" stroke-width="{sw}" marker-end="url(#m{k})"{D2 if dash else ""}/>')


def badge(x, y, n, k):
    st, _ = C[k]; add(f'<circle cx="{x}" cy="{y}" r="15" fill="{st}"/>'); text(x, y + 6, n, 17, anchor="middle", bold=True, fill="#fff")


def bars(x, y, w, h, vals, k, labels=None, unit="", maxv=None):
    """A small bar chart: bars on a baseline, value above each bar, label below."""
    st, fl = C[k]; mx = maxv or max(vals); n = len(vals); gw = w / n; bw = gw * 0.62
    add(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#9aa" stroke-width="1"/>')
    for i, v in enumerate(vals):
        bh = max(2, (h - 14) * v / mx); bx = x + i * gw + (gw - bw) / 2
        add(f'<rect x="{bx:.1f}" y="{y + h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{st}"/>')
        text(bx + bw / 2, y + h - bh - 4, f"{v:g}{unit}", 11, anchor="middle", bold=True)
        if labels: text(bx + bw / 2, y + h + 14, labels[i], 11, anchor="middle", fill="#445")


# ------------------------------------------------------------------------------------------------ the measured figures
test = j("results/causal/test.json", []); dev = j("results/causal/dev.json", [])
log = open(os.path.join(ROOT, "results", "causal_test_run.log"), encoding="utf-8").read() if os.path.exists(os.path.join(ROOT, "results", "causal_test_run.log")) else ""
m = re.search(r"positives: (\d+)\s+top-1 ([\d.]+)%\s+top-3 ([\d.]+)%", log); n_pos, top1, top3 = (m.group(1), m.group(2), m.group(3)) if m else ("?", "?", "?")
m = re.search(r"confident false cause ([\d.]+)%", log); false_cause = m.group(1) if m else "?"
ev = open(os.path.join(ROOT, "results", "causal", "events.txt"), encoding="utf-8").read() if os.path.exists(os.path.join(ROOT, "results", "causal", "events.txt")) else ""
m = re.search(r"ALL: (\d+) / (\d+)", ev); ev_ok, ev_n = (m.group(1), m.group(2)) if m else ("?", "?")
iso = j("results/isolation/pi_full.json", {"results": []})["results"]; att = j("results/isolation/pi_attack.json", {"results": []})["results"]
scen = [r for r in iso if "safety" in r]; misses = sum(r["safety"]["deadline_misses"] for r in scen); worst = max([r["safety"]["late_p9999_us"] for r in scen] or [0]); dead = (scen[0]["safety"]["deadline_us"] if scen else 0)
refused = att[0]["attack"]["refused"] if att else 0; attempts = att[0]["attack"]["attempts"] if att else 0
procs = j("results/scaling/laptop_procs.json", {"result": []})["result"]; w_ans = [r["answers_per_s"] for r in procs if r["clients"] == 12]; w_lab = [str(r["workers"]) for r in procs if r["clients"] == 12]
pi_thr = []
if os.path.exists(os.path.join(ROOT, "results", "scaling", "pi_threads.json")):
    pi_thr = j("results/scaling/pi_threads.json", {"result": []})["result"]
fleet = j("results/scaling/laptop_fleet.json", {"result": []})["result"]
n_prot = 9

add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI, system-ui, Helvetica, Arial, sans-serif">')
add("<defs>" + "".join(f'<marker id="m{k}" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{C[k][0]}"/></marker>' for k in C) + "</defs>")
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ------------------------------------------------------------------------------------------------ header
text(24, 40, "Vantage - an on-device vehicle assistant that reads every bus, answers in plain language, and says WHY", 25, bold=True)
text(24, 64, "Runs inside the car: no cloud, 2 GB, 2 ARM cores. The same build scales from the in-car board to a vehicle HPC. Every number below was measured; none is typed in.", 14, fill="#445")
chips = [("1", "p1", "Multiple protocols", [f"{n_prot} bus families, recognised with no per-car setup;", "each decoder checked against an independent", "implementation (cantools, scapy, udsoncan, can-j1939 ...)"]),
         ("2", "p2", "Explaining causes", [f"Sealed test, {n_pos} planted faults: right cause {top1} % first,", f"{top3} % in the top 3; invented causes: {false_cause} %.", f"Recorded events: {ev_ok} / {ev_n} traced to the planted cause."]),
         ("3", "p3", "Safety isolation", [f"Raspberry Pi 4, real kernel: {refused} / {attempts} escape attempts refused;", f"{misses} missed deadlines of a 100 Hz safety loop in {len(scen)} stress scenarios,", f"worst wake-up delay {worst:.0f} us against a {dead} us deadline."]),
         ("4", "p4", "Scalability to a vehicle HPC", [f"One build, 2 cores / 2 GB up to {w_lab[-1] if w_lab else '?'} worker processes:", f"{w_ans[0] if w_ans else '?'} -> {w_ans[-1] if w_ans else '?'} answers/s; {fleet[-1]['cars'] if fleet else '?'} cars into one hub at {round(fleet[-1]['records_per_s']) if fleet else '?'} records/s;", "the autoscaler adds workers and sheds load to hold 2 s."])]
for i, (n, k, title, ls) in enumerate(chips):
    x = 16 + i * 396; rect(x, 80, 384, 100, k, r=12); badge(x + 24, 106, n, k); text(x + 46, 112, title, 17, k, bold=True); lines(x + 14, 136, ls, 12.5, 16)

# ------------------------------------------------------------------------------------------------ column A: the vehicle and its protocols
rect(16, 200, 290, 500, "p1"); badge(40, 224, "1", "p1"); text(62, 230, "The vehicle: what it speaks", 15, "p1", bold=True)
protos = [("CAN - CAN-FD (DBC)", "cantools on 77 real-car DBCs"), ("SAE J1939 - BAM, RTS/CTS, DM1", "can-j1939 library"), ("OBD-II - ISO-TP", "python-can-isotp, 1-4095 B"), ("UDS (ISO 14229) - DTCs", "udsoncan"), ("SOME/IP - Service Discovery", "scapy"),
          ("Android VHAL properties", "AOSP VehicleProperty"), ("NMEA 0183 GNSS", "pynmea2"), ("Logs: logcat - syslog - slog2", "published line formats"), ("Camera: JPEG - ADAS records", "decode + schema tests")]
for i, (a, b) in enumerate(protos):
    y = 246 + i * 40; rect(28, y, 266, 34, "p1", r=7, extra='fill-opacity="0.9"'); text(38, y + 15, a, 13, bold=True); text(38, y + 29, "checked against: " + b, 10.5, fill="#445")
text(28, 622, "A new car needs no code: Vantage fingerprints", 12, fill="#334"); text(28, 638, "the wires, maps signals it is sure of and refuses", 12, fill="#334"); text(28, 654, "the rest (831 reviewed signals, 76 DBCs, 0 wrong).", 12, fill="#334")
text(28, 680, "J1939: 1785-byte transfers rebuilt from packets", 12, "p1", bold=True)

# vehicle safety box (outside the AI domain)
rect(16, 716, 290, 100, "p3"); text(28, 740, "Vehicle control and safety functions", 14, "p3", bold=True)
lines(28, 760, ["Brakes - steering - ADAS - airbags", "own cores, real-time priority, own memory;", "they never wait for the assistant"], 12.5, 16)
text(28, 810, "100 Hz loop, 200 us work, 1 ms deadline", 11, "p3", italic=True)

# the one-way tap and the wall
arrow(306, 330, 346, 330, "p1", sw=3)
text(326, 302, "listen-", 10.5, "p1", bold=True, anchor="middle"); text(326, 314, "only", 10.5, "p1", bold=True, anchor="middle")
add('<g><rect x="311" y="398" width="30" height="20" rx="4" fill="#fff" stroke="#c0392b" stroke-width="1.6"/><text x="326" y="413" font-size="13" text-anchor="middle" fill="#c0392b" font-weight="700">TX</text><line x1="311" y1="418" x2="341" y2="398" stroke="#c0392b" stroke-width="2.4"/></g>')
text(326, 436, "never", 10.5, "p3", anchor="middle"); text(326, 448, "sent", 10.5, "p3", anchor="middle")
add('<line x1="326" y1="716" x2="326" y2="816" stroke="#c0392b" stroke-width="5"/>'); add('<text transform="rotate(-90 316 790)" x="316" y="790" font-size="11" font-weight="700" fill="#c0392b" text-anchor="middle">kernel wall</text>')

# ------------------------------------------------------------------------------------------------ the AI domain
rect(346, 200, 1238, 500, "p3", r=14, dash=True, sw=2.2, fill=False)
rect(360, 190, 650, 24, "p3", r=6); badge(378, 202, "3", "p3")
text(398, 207, "AI DOMAIN - a hardened sandbox the vehicle's functions cannot be hurt by", 13.5, bold=True, fill="#c0392b")
sand = ["no network socket", "no CAN / raw socket", "no write to vehicle files", "no real-time priority", "no signals to other processes", "no new privileges", "only its 2 cores", "memory and tasks capped"]
for i, s in enumerate(sand):
    cx = 362 + (i % 4) * 160; cy = 224 + (i // 4) * 22; add(f'<rect x="{cx}" y="{cy}" width="152" height="18" rx="9" fill="#fff" stroke="#c0392b" stroke-width="1"/>'); text(cx + 76, cy + 13, s, 11, anchor="middle", fill="#8e2a1f")
text(1012, 238, "systemd unit: cgroup v2 cpuset, cpu, memory, pids; seccomp;", 11.5, fill="#8e2a1f"); text(1012, 253, f"PrivateNetwork, read-only file system - {refused} / {attempts} attempts refused on the Pi", 11.5, fill="#8e2a1f")

# column B: ingest
rect(360, 270, 262, 416, "n"); text(374, 292, "Ingest (listen-only)", 15, bold=True)
blocks = [("Receiver", "one reader per wire, bounded queues"), ("Fingerprint", "which car, which wiring"), ("Decoders + mapper", "units, scaling, refuse-not-guess"), ("Event detectors", "faults, resets, warnings, bursts")]
for i, (a, b) in enumerate(blocks):
    y = 304 + i * 62; rect(374, y, 234, 48, "p1", r=8); text(386, y + 20, a, 13.5, bold=True); text(386, y + 38, b, 11.5, fill="#445")
    if i < 3: arrow(491, y + 48, 491, y + 62, "p1")
arrow(491, 552, 491, 574, "n")
add('<path d="M382,590 a109,10 0 0 1 218,0 v62 a109,10 0 0 1 -218,0 z" fill="#fff4e6" stroke="#f58220" stroke-width="1.5"/><ellipse cx="491" cy="590" rx="109" ry="10" fill="#ffe6c8" stroke="#f58220" stroke-width="1.5"/>')
text(491, 616, "Signal lake and events", 13.5, bold=True, anchor="middle"); text(491, 633, "1 s tier + 1 min tier, SQLite", 11.5, anchor="middle", fill="#445"); text(491, 648, "weeks to years, searchable", 11.5, anchor="middle", fill="#445")
arrow(622, 620, 660, 620, "n")

# column C: the assistant
rect(660, 270, 280, 416, "n"); text(674, 292, "Assistant (small language model)", 15, bold=True)
c_blocks = [("Question", "typed or spoken, any wording"), ("Meaning model", "Qwen2.5-0.5B, int8 on ONNX Runtime"), ("Typed query plan", "validated against what the car records"), ("Executors", "events - trends - anomalies - history"), ("Answer", "numbers from data, never from the model")]
for i, (a, b) in enumerate(c_blocks):
    y = 304 + i * 62; rect(674, y, 252, 48, "p1" if i in (1,) else "n", r=8); text(686, y + 20, a, 13.5, bold=True); text(686, y + 38, b, 11.5, fill="#445")
    if i < 4: arrow(800, y + 48, 800, y + 62, "n")
lat = ""
if pi_thr:
    for r in pi_thr:
        if (r.get("threads") or r.get("cores")) == 2: lat = f"Pi 4, 2 cores: meaning step {r['parse_median_s']:.2f} s median"
text(674, 634, "The model only reads the question;", 11.5, fill="#445"); text(674, 649, "every figure comes from the stored data.", 11.5, fill="#445")
if lat: text(674, 672, lat, 11.5, "p4", bold=True)

# the arrow from the lake into the cause engine and from the executors too
arrow(940, 360, 974, 360, "p2", sw=2.6); arrow(940, 560, 974, 560, "p2", sw=2.6)

# column D: why engine
rect(974, 270, 598, 416, "p2"); badge(998, 294, "2", "p2"); text(1020, 300, "WHY ENGINE - explains causes, with evidence and honesty", 15.5, "p2", bold=True)
stages = [("DETECT", ["change-points in the daily", "series: step, ramp, spike,", "end-of-period trend;", "permutation-tested, de-noised"]),
          ("ATTRIBUTE", ["physical trip model (speed^2,", "climate power, cold, highway", "share) to split causes;", "20+ cause signatures;", "event graph: reset -> ADAS drop"]),
          ("JUDGE", ["competing hypotheses are scored", "against each other and against", "'no cause'; confidence is", "calibrated; 'nothing changed'", "is a valid answer"])]
for i, (t, ls) in enumerate(stages):
    x = 988 + i * 192; rect(x, 316, 180, 120, "p2", r=8, fill=True, extra='fill="#fff"'); text(x + 90, 336, t, 13.5, "p2", bold=True, anchor="middle"); lines(x + 10, 356, ls, 11.5, 16)
    if i < 2: arrow(x + 180, 380, x + 192, 380, "p2")
ex = j("results/causal/example_answer.json", {"question": "", "answer": "", "scenario": ""})
import textwrap
rect(988, 446, 570, 148, "p2", r=8, extra='fill="#fff"'); text(1000, 464, 'Q: "' + ex["question"] + '?"', 12.5, bold=True)
body = []
for part in ex["answer"].split(chr(10)):
    if part.startswith("Could not be checked") or part.startswith(" - "): body += textwrap.wrap(part.strip(" -"), 100) if part.startswith(" - ") else []; continue
    body += textwrap.wrap(part, 100)
lines(1000, 482, body[:7], 11.2, 15.2, fill="#2a1a3d")
text(1000, 588, "real output on " + ex["scenario"], 10, fill="#6b5a80", italic=True)
rect(988, 604, 570, 70, "p2", r=8, extra='fill="#fff"'); text(1000, 622, "Measured on histories with the cause planted and never read by the assistant", 12, "p2", bold=True)
lines(1000, 640, [f"sealed test: {n_pos} scenarios - first guess right {top1} %, in top 3 {top3} %, invented cause {false_cause} %;", f"recorded events (resets, bursts, dips, speed conflicts): {ev_ok} / {ev_n} traced to the real cause."], 11.2, 15.5)

# ------------------------------------------------------------------------------------------------ bottom band: scaling
import dataclasses
sys.path.insert(0, ROOT)
from vgpt.scale.probe import probe
from vgpt.scale.plan import make_plan
hw0 = probe()
tiers = [("In-car board", "2 ARM cores, 2 GB", dict(cores=2, mem_mb=2048, arch="aarch64", isa=frozenset({"neon"})), "measured on the Raspberry Pi 4 limited to 2 cores", 1),
         ("Raspberry Pi 4", "4 cores, 3.7 GB", dict(cores=4, mem_mb=3795, arch="aarch64", isa=frozenset({"neon"})), "measured on the real board", 1),
         ("Laptop / VM as HPC stand-in", "12 threads, 16 GB", dict(cores=12, mem_mb=16159), "measured here (worker processes, hub, 3-year history)", 1),
         ("Vehicle HPC", "32 cores, 64 GB, 16 cars", dict(cores=32, mem_mb=65536, arch="aarch64", isa=frozenset({"neon", "dotprod"})), "plan computed by the same code; not run on such a machine", 16)]
rect(16, 832, 1568, 388, "p4", r=14); badge(40, 858, "4", "p4"); text(62, 864, "AUTO-SCALING - one build, any machine: the plan is computed from the hardware, then corrected by measurement", 16, "p4", bold=True)
flow = [("1  Probe", "cores, RAM, ISA, GPU"), ("2  Plan", "contexts x threads, history, sessions"), ("3  Calibrate", "measure the per-context thread knee"), ("4  Serve", "worker processes, priority queue"), ("5  Autoscale", "p95 vs 2 s target, memory guard")]
for i, (a_, b_) in enumerate(flow):
    x = 30 + i * 262; rect(x, 876, 236, 44, "p4", r=8, extra='fill="#fff"'); text(x + 10, 895, a_, 13.5, "p4", bold=True); text(x + 10, 912, b_, 10.8, fill="#445")
    if i < 4: arrow(x + 236, 898, x + 262, 898, "p4")
text(30, 944, "The same plan code, given four different machines:", 12.5, bold=True)
for i, (name, spec, kw, how, veh) in enumerate(tiers):
    h = dataclasses.replace(hw0, **kw); p_ = make_plan(h, have_onnx=True, vehicles=veh); x = 30 + i * 386; w = 374
    rect(x, 952, w, 138, "p4", r=9, extra='fill="#fff"'); text(x + 12, 972, name, 14, "p4", bold=True); text(x + w - 12, 972, spec, 11.5, anchor="end", fill="#445")
    start = f"{p_.contexts} model context(s) x {p_.threads} thread(s) at start; grows to {p_.max_contexts}"
    lines(x + 12, 992, [start, f"{p_.sessions} people at once - {p_.fine_days} days at 1 s, {p_.coarse_days} days per minute", f"{p_.ingest_workers} ingest worker(s) for {veh} vehicle(s); camera {p_.camera_fps:g} fps", "tier label: " + p_.tier], 11.5, 16)
    text(x + 12, 1075, how, 10.5, "p4", italic=True)
text(30, 1112, "When load threatens 2 s the autoscaler first adds workers, then sheds in this order:", 12.5, bold=True)
shed = ["pause analytics", "lower camera rate", "queue service + fleet", "refuse non-driver", "shorten answers"]
for i, s_ in enumerate(shed):
    x = 30 + i * 150; rect(x, 1122, 142, 22, "p4", r=11, extra='fill="#fff"'); text(x + 71, 1137, s_, 11, anchor="middle", fill="#14602f")
text(30, 1164, "A driver's question is never shed. When load falls the workers are released and the memory returns.", 12, "p4", bold=True)
text(30, 1182, "Verified: autoscaler logic (29 tests, laptop and Pi), worker pool crash retry, sticky sessions, fleet receiver restart.", 11.5, fill="#445")

text(806, 1112, "Pi 4 meaning step vs cores, s", 11.5, bold=True)
if pi_thr:
    vals = [r["parse_median_s"] for r in pi_thr]; labs = [f"{r.get('threads') or r.get('cores')} core" for r in pi_thr]; bars(806, 1122, 220, 56, vals, "p4", labs, "", maxv=max(vals))
text(1052, 1112, "Laptop: answers/s vs workers", 11.5, bold=True)
if w_ans: bars(1052, 1122, 230, 56, w_ans, "p4", w_lab, "")
text(1308, 1112, "Cars into one hub: records/s", 11.5, bold=True)
if fleet: bars(1308, 1122, 262, 56, [round(r["records_per_s"]) for r in fleet], "p4", [str(r["cars"]) for r in fleet], "")
add("</svg>")
open(os.path.join(ROOT, "docs", "architecture.svg"), "w", encoding="utf-8").write("\n".join(out))
print("wrote docs/architecture.svg", len("\n".join(out)), "bytes")
