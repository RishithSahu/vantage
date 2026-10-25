"""Does the data that ARRIVES through the pipeline give the same answers as the finished recording?

    python bench/meaning_eval/pipeline_fidelity.py --seconds 3600 [--sets novel_v1 messy_v1 ...] [--vehicle world_aster.db]

Two databases of the same vehicle at the same moment T:
  A  the finished recording (everything the world holds, read at T)
  B  the application's own database after the source delivered the recording from the replay start to T as bus messages (wire path), decoded and stored by the ingest runtime
Every frozen question is asked of both with the CORRECT reading supplied (the same trick as e2e_world.py), so a difference can only come from the data.  Output: the share of identical answers, and the
differences grouped by what they are about.  results/meaning/pipeline_fidelity.json keeps every row."""
import argparse
import collections
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import evalio                                                     # noqa: E402

norm = lambda s: " ".join(str(s).split())
EVIDENCE = ("Based on", "Evidence:", "Data quality:", "Only ", "Coverage")


def core(a):
    """The answer without its evidence lines (sample counts, coverage): what the person is told, not how many rows stood behind it."""
    return norm(" ".join(l for l in str(a).splitlines() if not l.strip().startswith(EVIDENCE)))


class GoldReader:
    def __init__(self, real): self.real, self.frame = real, None
    def parse(self, text, hint=None, prev=None): return self.frame, {"raw": self.frame.to_text(), "pmin": 0.99, "ms": 0.0, "hint": ""}
    def __getattr__(self, k): return getattr(self.real, k)


class Sched: laps = 0


def build_arrived(src, seconds, begin=None):
    """-> (path of the database built by delivery, begin, T)."""
    import importlib.util
    from vgpt.live import replay as RP
    from vgpt.vehicle.profile import Profile
    spec = importlib.util.spec_from_file_location("uiserver_fid", os.path.join(ROOT, "ui", "server.py")); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    work = os.path.join(tempfile.gettempdir(), "fidelity_" + os.path.basename(src))
    for e in ("", "-wal", "-shm"):
        try: os.remove(work + e)
        except OSError: pass
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True); d = sqlite3.connect(work); s.backup(d); d.close()
    c0 = sqlite3.connect(f"file:{src}?mode=ro", uri=True); plan = S.make_plan(c0); meta = dict(c0.execute("SELECT k, v FROM meta")); c0.close()
    begin = int(begin or plan.begin); c = sqlite3.connect(work, isolation_level=None); RP.cut_after(c, begin); c.close()
    w = RP.WireSource(Sched(), src, work, Profile.load(meta["vehicle.id"]), begin); T = w.deliver(begin + seconds); w.publish(); w.close()
    return work, begin, T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vehicle", default="world_aster.db"); ap.add_argument("--seconds", type=int, default=3600); ap.add_argument("--begin", type=int, default=None)
    ap.add_argument("--sets", nargs="+", default=["novel_v1", "messy_v1", "user_v1", "platform_v1", "platform_v2", "adv_v1", "sealed_v2", "hard_v1"]); ap.add_argument("--adapter", default="models/meaning-lora-full_v11.gguf"); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    from vgpt.product import make_meaning_brain
    from vgpt.vehicle import lake as LK
    src = os.path.join(ROOT, "data", a.vehicle); t0 = time.time(); work, begin, T = build_arrived(src, a.seconds, a.begin); print(f"delivered {T - begin} s of the recording in {time.time() - t0:.1f} s; asking at second {T}", flush=True)
    LK.adopt_path(src)
    adapter = os.path.join(ROOT, a.adapter)
    A = make_meaning_brain(src, now=T, threads=a.threads, adapter=adapter); LK.release(); LK.adopt_path(work); B = make_meaning_brain(work, now=T, threads=a.threads, adapter=adapter)
    import vgpt.meaning.pipeline as PL
    ga, gb = GoldReader(A.frame_model), GoldReader(B.frame_model); out = []
    def fresh(b):
        b.qcs.last = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []; b.units = {"p": "kpa", "d": "km", "t": "c"}; b._last_data = None
        getattr(b, "dialogue", None) and b.dialogue.reset()
    PL.arbitrate = lambda text, frame, onto, glossary: (frame, None)
    for name in a.sets:
        path = os.path.join(HERE, name + ".tsv")
        if not os.path.exists(path): continue
        for r in evalio.load_tsv(path):
            res = []
            for b, g in ((A, ga), (B, gb)):
                fresh(b); b._text = r["text"]; g.frame = r["gold"]; b.frame_model = g; b._frames.clear()
                try: res.append(b.ask(r["text"]))
                except Exception as e: res.append({"answer": f"EXCEPTION {type(e).__name__}"})
            out.append({"set": name, "id": r["id"], "text": r["text"], "same": norm(res[0].get("answer")) == norm(res[1].get("answer")), "same_core": core(res[0].get("answer")) == core(res[1].get("answer")), "recorded": str(res[0].get("answer"))[:300], "arrived": str(res[1].get("answer"))[:300],
                        "route": (res[0].get("facts") or {}).get("route") if isinstance(res[0].get("facts"), dict) else None})
        done = [o for o in out if o["set"] == name]; print(f"{name:12} n={len(done):4} identical {100 * sum(o['same'] for o in done) / max(len(done), 1):5.1f}%   same content {100 * sum(o['same_core'] for o in done) / max(len(done), 1):5.1f}%", flush=True)
    n = len(out); same = sum(o["same"] for o in out); core_ = sum(o["same_core"] for o in out); print(f"ALL          n={n} identical answers {100 * same / n:.1f}%   same content (evidence lines aside) {100 * core_ / n:.1f}%")
    by = collections.Counter((o["route"] or "?").split(":")[0] + ":" + ((o["route"] or "?").split(":")[1] if ":" in (o["route"] or "") else "") for o in out if not o["same_core"])
    print("differences by route:", by.most_common(14))
    os.makedirs(os.path.join(ROOT, "results", "meaning"), exist_ok=True)
    json.dump({"vehicle": a.vehicle, "begin": begin, "T": T, "rows": out}, open(os.path.join(ROOT, "results", "meaning", "pipeline_fidelity.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
