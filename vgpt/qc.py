"""Query compiler front door: the whole NEW pipeline in one object.

    text --nl2ir (rules)--> Query --(SLM fallback)--> Query --validate--> capability --plan--> execute --> evidence --> compose --> verify --> answer

`QueryCompiler.execute(query)` runs everything after the parser (so tests, the training pipeline and the model path all share it); `QueryCompiler.ask(text)` adds the language side. Every stage is timed.
It never raises for a question it cannot handle: it returns None (so the caller falls back to the legacy engine) or an explained answer (missing data, no data, unsupported)."""
import os
import sqlite3
import time
from dataclasses import dataclass, field

from . import answer_composer as AC, answer_verifier as AV, contracts as CT, evidence as EV, signal_registry as R
from .capability import OK as CAP_OK
from .query_executor import Executor, context_for
from .query_ir import Query
from .query_validator import validate


@dataclass
class QCResult:
    query: Query
    answer: AC.Answer
    result: object = None
    evidence: EV.Evidence = None
    validation: object = None
    verification: object = None
    timings: dict = field(default_factory=dict)
    status: str = "answered"            # answered | missing_data | no_data | invalid | unsupported | unverified
    fallback_used: bool = False
    debug_trace: dict = field(default_factory=dict)

    def facts(self):
        out = {"ok": self.status == "answered", "route": f"qc:{self.query.op}", "qc": True, "status": self.status, "query": self.query.to_text(), "plan": (self.result.plan if self.result is not None else []),
                "evidence": self.evidence.to_dict() if self.evidence is not None else None, "timings_ms": {k: round(v, 2) for k, v in self.timings.items()}, "verified": bool(self.verification and self.verification.ok), "contract": (self.result.extra.get("contract") if self.result is not None else None)}
        if os.environ.get("VGPT_DEV_TRACE") == "1": out["debug_trace"] = self.debug_trace
        return out


class QueryCompiler:
    def __init__(self, conn, now=None, legacy=None, profile=None):
        self.conn = conn; self.profile = profile or R.Profile(conn); self.ctx = context_for(conn, now); self.legacy = legacy
        try: self._data_version = conn.execute("PRAGMA data_version").fetchone()[0]
        except sqlite3.Error: self._data_version = None
        self.executor = Executor(conn, self.profile, self.ctx, legacy=legacy)

    def refresh(self, now=None):
        """Re-read the coverage (the history can grow while the app runs)."""
        self.ctx = context_for(self.conn, now)
        try: version = self.conn.execute("PRAGMA data_version").fetchone()[0]
        except sqlite3.Error: version = self._data_version
        if version != self._data_version:
            old = self.profile; self.profile = R.Profile(self.conn); self._data_version = version; self.executor.profile = self.profile
            self.profile._avail = {k: v for k, v in getattr(old, "_avail", {}).items() if v is True}                        # a column that held data still does after new rows arrived: only absence is looked at again (a live car changes the data at every question, and proving a view of the lake non-empty can read millions of rows)
        self.executor.ctx = self.ctx

    def execute(self, q, text=None):
        T = {}; t0 = time.perf_counter()
        def trace(res=None):
            from .capability import concepts_of
            return {"raw": text, "front_end": "typed parser", "canonical_query": q.to_dict(), "query_text": q.to_text(), "signal_ids": concepts_of(q),
                    "time": q.scope.time.to_text(), "filters": [c.to_text() for c in q.scope.conditions], "source_kind": (getattr(res, "quality_kind", None) if res is not None else None),
                    "sources": list(getattr(res, "sources", []) or []), "sample_count": getattr(res, "samples", None), "observation_time": getattr(res, "when", None),
                    "unit": getattr(res, "unit", None), "calculation": getattr(res, "formula", None), "missing_data_policy": "never substitute zero or another signal"}
        v = validate(q, self.profile); T["validate"] = (time.perf_counter() - t0) * 1000
        if not v.valid:
            return QCResult(q, AC.Answer("I couldn't turn that into a valid query: " + "; ".join(i.message for i in v.issues[:2]) + ".", [], [], "unsupported"), validation=v, timings=T, status="invalid", debug_trace=trace())
        if not v.capability.ok:
            a = AC.compose(None, EV.Evidence(status="missing"), q, self.ctx.tz, self.profile, capability=v.capability)
            return QCResult(q, a, validation=v, timings=T, status="missing_data" if v.capability.status != "UNKNOWN" else "unsupported", debug_trace=trace())
        t1 = time.perf_counter()
        try: res = self.executor.run(q.normalized())
        except Exception as e:                                                       # a plan that cannot run is an internal problem: report it, let the caller fall back
            return QCResult(q, AC.Answer("I couldn't compute that.", [f"internal: {type(e).__name__}"], [], "unsupported"), validation=v, timings=T, status="invalid")
        T["execute"] = (time.perf_counter() - t1) * 1000; t2 = time.perf_counter()
        ev = EV.from_result(res, q, self.profile); ev.claim = ""
        contract = CT.assess(q, res, self.ctx, self.conn, self.profile) if res.status == "ok" else None
        if contract is not None: res.extra.update(CT.numbers(contract)); res.extra["contract"] = contract.to_dict()
        a = AC.compose(res, ev, q, self.ctx.tz, self.profile)
        if contract is not None: CT.apply(a, contract)
        T["compose"] = (time.perf_counter() - t2) * 1000; t3 = time.perf_counter()
        rep = AV.verify(a, ev, q, res, v.capability, self.ctx)
        if contract is not None:
            for issue in CT.check(a.text, q, contract): rep.fail("CONTRACT", issue)
        T["verify"] = (time.perf_counter() - t3) * 1000
        status = "answered" if res.status == "ok" else "no_data" if res.status == "no_data" else "unsupported"
        fb = False
        if not rep.ok:
            a = AV.safe_fallback(res, ev, q); fb = True; status = "unverified"
        T["total"] = sum(T.values())
        return QCResult(q, a, res, ev, v, rep, T, status, fb, trace(res))
