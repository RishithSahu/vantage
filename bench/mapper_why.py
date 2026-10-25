"""Explain one decision of the mapper:  python bench/mapper_why.py <dbc file> MSG.SIG [MSG.SIG ...]"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
import mapper_benchmark as B  # noqa: E402
from vgpt.ingest import mapper as MP  # noqa: E402
from vgpt.protocols import dbc as DBC  # noqa: E402

db = DBC.Database.load(sys.argv[1], lenient=True)
for ref in sys.argv[2:]:
    msg, _, sig = ref.partition("."); m = db.by_name(msg); s = next(x for x in m.signals if x.name == sig)
    ev = MP.Evidence(**B.evidence(db, m, s, MP.powertrain_is_ice(db))); P = MP.parse(ev); d = MP.decide(ev)
    print(f"{ref}: toks {P.toks} ctx {P.ctx} pos {P.axle},{P.side} shape {P.shape} unit_cls {P.unit_cls} span {P.span}")
    print(f"   -> {d.concept} conf {d.confidence} {d.reasons} refused={d.refused!r}")
    print(f"   alternatives {d.alternatives}")
