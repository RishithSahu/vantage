"""Check that every canonical topic phrase (vgpt/topics.py) reaches its own route in the legacy route handlers.  python bench/topic_canon_check.py"""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
os.environ["VGPT_FRONTEND"] = "legacy"
from vgpt.product import make_legacy_brain
from vgpt import topics as T
b = make_legacy_brain(os.path.join(HERE, "data", "gen_demo.db"), slm=None, now=1787625900)
bad = 0
for tid, (phrase, _desc) in T.TOPICS.items():
    b.qcs.last = None; b.sem_topic = None; b.qcs.choices = []; b.qcs.pending = None; b.qcs.options = []
    r = b.ask(phrase, use_slm=False); f = r.get("facts") if isinstance(r.get("facts"), dict) else {}
    route = f.get("route"); ok = route == tid or (route or "").replace("qc:", "") == tid
    bad += (not ok)
    print(("ok  " if ok else "BAD ") + f"{tid:26s} -> {str(route):26s} | {phrase} | {r['answer'][:60]!r}")
print("mismatches:", bad)
