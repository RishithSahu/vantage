"""Diagnostic: how well does a FROZEN pretrained schema-conditioned extractor (GLiNER2) pick the intended analytical operation for a development question, from labels generated out of the plan-language definition (toolspec.FUNCS)? No generation model is involved; no training. The extractor runs offline from the local Hugging Face cache (HF_HUB_OFFLINE=1): nothing is downloaded.
    PYTHONUTF8=1 python tools_pin.py 4,5 bench/diag_extractor.py [--model fastino/gliner2-base-v1] [--label short|meaning|both] [--set dev|final]
Reports top-1 / top-2 / top-3 accuracy of the operation label on the stand-alone questions (follow-ups are judged in the combined path, not here), the confusions, and the per-question time."""
import argparse, collections, json, os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from bench.diag_function import EQUIV, gold_function
from bench.indep_io import load
from vgpt import toolspec as T

SPECIAL = {"decline": "an instruction to operate the vehicle or a request unrelated to the vehicle's data", "say": "a greeting or thanks", "ask_user": "too vague to tell what is being asked", "unresolved": "about a quantity this vehicle does not record"}


NL = {"current_value": "asks what a measurement is right now", "statistic": "asks for an average, total, median or other statistic of a measurement over a period of time", "extreme": "asks for the highest, lowest, maximum or minimum value reached",
      "count_events": "asks how many times something happened", "count_entities": "asks how many trips or charging sessions there were", "list_events": "asks to list or show the individual events or warnings", "did_event_happen": "asks yes or no whether an event or warning happened",
      "did_condition_hold": "asks yes or no whether a measurement ever went above or below a level", "time_share": "asks what percentage or share of the time something was true", "compare": "asks to compare two time periods or two driving situations",
      "when_changed": "asks when a measurement changed the most", "event_context": "asks what was happening just before or after an event or warning", "explain_change": "asks why a measurement went up or down or got worse",
      "explain_event": "asks why an event or warning happened", "trend": "asks whether a measurement is trending up or down over time", "rank_trips": "asks for the top trips or charging sessions ordered by something, such as the longest or most efficient",
      "correlate": "asks how two measurements relate to each other", "is_recorded": "asks whether the vehicle records or has data for something", "use_tool": "asks for a service, tyre, weather, health or driving-style check"}


def labels(kind):
    out = {}                                                                                           # label text -> function
    for fn in T.FUNCS:
        if fn in SPECIAL: out[SPECIAL[fn]] = fn; continue
        if fn in T.NO_TURN: continue
        txt = {"nl": NL[fn], "short": T.func_label(fn), "meaning": T.func_meaning(fn), "both": f"{T.func_label(fn)}: {T.func_meaning(fn)}"}[kind]
        out[txt] = fn
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="fastino/gliner2-base-v1"); ap.add_argument("--label", default="short"); ap.add_argument("--set", default="dev"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    path = {"dev": os.path.join(HERE, "bench", "indep_dev.tsv"), "final": os.path.join(HERE, "bench", "indep_final.tsv")}.get(a.set, a.set)
    if os.path.basename(path).startswith("acceptance"): raise SystemExit("sealed")
    from gliner2 import GLiNER2
    m = GLiNER2.from_pretrained(a.model); lab = labels(a.label); names = list(lab)
    rows = [r for r in load(path) if not r["prev_text"]]
    if a.limit: rows = rows[: a.limit]
    top = {1: 0, 2: 0, 3: 0}; n = 0; conf = collections.Counter(); t0 = time.time(); rec = []
    for r in rows:
        g = gold_function(r["ir"])
        if g is None: continue
        res = m.extract(r["text"], m.create_schema().classification("operation", names, multi_label=True, cls_threshold=0.0), include_confidence=True)["operation"]
        ranked = [lab[x["label"]] for x in sorted(res, key=lambda x: -x["confidence"])]; ok = {g} | EQUIV.get(g, set()) | ({"decline", "say", "ask_user", "unresolved"} if g == "decline" else set()); n += 1
        for k in top: top[k] += any(x in ok for x in ranked[:k])
        if not any(x in ok for x in ranked[:1]): conf[(g, ranked[0] if ranked else None)] += 1
        rec.append({"id": r["id"], "gold": g, "ranked": ranked[:3]})
    print(f"{os.path.basename(path)} label={a.label} model={a.model}: n={n} " + ", ".join(f"top-{k} {top[k]}/{n} = {100 * top[k] / n:.0f}%" for k in top) + f"  ({(time.time() - t0) / n * 1000:.0f} ms/question)")
    for (g, f), c in conf.most_common(10): print(f"   gold {g:18s} -> {f} x{c}")


if __name__ == "__main__": main()
