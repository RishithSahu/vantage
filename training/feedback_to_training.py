"""Reviewed feedback -> training data. Nothing a user typed ever reaches a training file without a human decision.

    1. collect      python -m training.feedback_to_training collect [--feedback feedback.jsonl] --out training/data/feedback_candidates.jsonl
                    Selects the questions worth looking at (thumbs-down, or answered by the fallback / refused) and writes CANDIDATES with status "pending".
                    Personal data is scrubbed (e-mail addresses, phone numbers, coordinates, long digit strings, VINs); text that looks like an attack is quarantined, not offered.
                    Each candidate carries the query our parser produced ("proposed_ir", possibly empty) to save the reviewer time.
    2. review       a person opens the JSONL and, per line, sets  "status": "approved" (+ writes the correct "ir")  or  "rejected".  Anything left "pending" is ignored.
    3. approve      python -m training.feedback_to_training approve training/data/feedback_candidates.jsonl --db data/gen_courier_ext.db --out training/data/feedback_approved.jsonl
                    Re-checks every approved line mechanically: the IR must parse, validate, EXECUTE (or refuse for a stated reason) on a real database, must be grounded in the question,
                    and must not duplicate any text already in a train or test split (no leakage into the held-out sets; approved examples only ever go to training).
    4. merge        python -m training.feedback_to_training merge --data training/data     appends feedback_approved.jsonl to train.jsonl (a new adapter is then trained and evaluated as usual).

The reviewer is the safety boundary: the model is never retrained on its own outputs or on raw user text."""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)

PII = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"), (re.compile(r"\+?\d[\d ()-]{8,}\d"), "<number>"), (re.compile(r"-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}"), "<coordinates>"), (re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"), "<vin>")]
ATTACK = re.compile(r"(;|--|/\*)\s*\w|\b(drop|delete|update|insert|attach|pragma)\s+(table|from|into|database|\w+\s*=)|<\s*script|\bignore (all )?(previous|prior) instructions\b|\brm\s+-rf\b|\bsystem prompt\b", re.I)


def scrub(text):
    for rx, rep in PII: text = rx.sub(rep, text)
    return " ".join(text.split())[:300]


def collect(feedback_path, out):
    from vgpt import feedback_report, nl2ir
    items = feedback_report.load(feedback_path); seen = set(); n = q = 0
    with open(out, "w", encoding="utf-8") as f:
        for i in items:
            text = i.get("q") or ""
            route = str(i.get("route") or i.get("path") or "")
            if not text or not (i.get("rating") == "down" or route in ("clarify", "gate", "unsupported")): continue
            clean = scrub(text)
            if clean in seen: continue
            seen.add(clean)
            if ATTACK.search(text): q += 1; continue                                                  # quarantined: never offered for training
            p = nl2ir.parse(clean, now=None, tz=0)
            rec = {"id": hashlib.sha1(clean.encode()).hexdigest()[:12], "text": clean, "status": "pending", "proposed_ir": p.query.to_text() if p.ok else "", "rating": i.get("rating"), "route": route, "note": scrub(str(i.get("note") or "")), "ir": ""}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
    print(f"{n} candidates written to {out} ({q} quarantined as attacks); open it, fill in \"ir\" and set \"status\" to approved or rejected")


def existing_texts(data_dir):
    seen = set()
    for fn in os.listdir(data_dir) if os.path.isdir(data_dir) else []:
        if fn.endswith(".jsonl") and not fn.startswith("feedback"):
            for line in open(os.path.join(data_dir, fn), encoding="utf-8"):
                try: seen.add(json.loads(line)["text"].lower())
                except Exception: pass
    return seen


def _record_text(record):
    """Return a normalized, safe reviewed text or ``None``.

    Feedback files are user-controlled input.  A malformed JSONL row must not
    abort a review/merge run or become an accidental training example.
    """
    text = record.get("text") if isinstance(record, dict) else None
    if not isinstance(text, str): return None
    text = scrub(text)
    return text or None


def approve(cand_path, db, out, data_dir):
    import sqlite3
    from vgpt.ir_slm import grounding_issues
    from vgpt.qc import QueryCompiler
    from vgpt.query_ir import IRError, Query
    from vgpt.query_validator import validate
    from training.build_dataset import DEFAULT_NOW, gold_of
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True); qc = QueryCompiler(conn, now=DEFAULT_NOW); have = existing_texts(data_dir); ok = bad = 0; reasons = {}
    def reject(r): nonlocal bad; bad += 1; reasons[r] = reasons.get(r, 0) + 1
    with open(out, "w", encoding="utf-8") as f:
        for line in open(cand_path, encoding="utf-8"):
            try: r = json.loads(line)
            except ValueError: reject("malformed candidate JSON"); continue
            if not isinstance(r, dict): reject("candidate is not an object"); continue
            if r.get("status") != "approved": continue
            text = _record_text(r)
            if text is None: reject("missing text"); continue
            ir = (r.get("ir") or "").strip()
            if text.lower() in have: reject("duplicate of an existing example"); continue
            if ATTACK.search(text): reject("attack-like text"); continue
            if ir == "op=none":
                # A refusal is valid only when a reviewer has documented why;
                # otherwise an unanswerable model output could teach the model
                # to refuse an ordinary telemetry request.
                reason = scrub(str(r.get("refusal_reason") or ""))
                if not reason: reject("op=none requires a reviewer refusal_reason"); continue
                f.write(json.dumps({"text": text, "ir": ir, "op": "none", "tags": ["feedback", "op:none"], "template": 0, "split": "train", "noisy": False, "gold": {"status": "refused", "reason": reason}}, ensure_ascii=False) + "\n"); have.add(text.lower()); ok += 1; continue
            try: q = Query.from_text(ir)
            except IRError as e: reject(f"IR does not parse: {e}"); continue
            if validate(q).codes(): reject("IR invalid"); continue
            if grounding_issues(q, text): reject("IR not grounded in the question"); continue
            res = qc.execute(q)
            if res.status == "invalid" or (res.verification is not None and not res.verification.ok): reject(f"IR does not execute cleanly ({res.status})"); continue
            f.write(json.dumps({"text": text, "ir": q.to_text(), "op": q.op, "tags": ["feedback", f"op:{q.op}"], "template": 0, "split": "train", "noisy": False, "gold": gold_of(res)}, ensure_ascii=False) + "\n"); have.add(text.lower()); ok += 1
    print(f"{ok} approved examples written to {out}; {bad} rejected {reasons if reasons else ''}")


def merge(data_dir):
    src = os.path.join(data_dir, "feedback_approved.jsonl"); dst = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(src): raise SystemExit("no feedback_approved.jsonl: run approve first")
    have = existing_texts(data_dir)
    add = []
    rejected = 0
    for line in open(src, encoding="utf-8"):
        try: record = json.loads(line)
        except ValueError: rejected += 1; continue
        text = _record_text(record)
        # Approved feedback is always training-only.  Never merge a record
        # whose split was edited to a held-out split or whose text is duplicate.
        if not text or record.get("split") != "train" or text.lower() in have:
            rejected += 1; continue
        have.add(text.lower())
        record["text"] = text
        add.append(json.dumps(record, ensure_ascii=False) + "\n")
    with open(dst, "a", encoding="utf-8") as f: f.writelines(add)
    print(f"appended {len(add)} reviewed examples to {dst}; rejected {rejected} malformed, duplicate, or non-training rows")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--feedback"); c.add_argument("--out", default=os.path.join(HERE, "training", "data", "feedback_candidates.jsonl"))
    a = sub.add_parser("approve"); a.add_argument("candidates"); a.add_argument("--db", default=os.path.join(HERE, "data", "gen_courier_ext.db")); a.add_argument("--out", default=os.path.join(HERE, "training", "data", "feedback_approved.jsonl")); a.add_argument("--data", default=os.path.join(HERE, "training", "data"))
    m = sub.add_parser("merge"); m.add_argument("--data", default=os.path.join(HERE, "training", "data"))
    x = ap.parse_args()
    if x.cmd == "collect": collect(x.feedback, x.out)
    elif x.cmd == "approve": approve(x.candidates, x.db, x.out, x.data)
    else: merge(x.data)


if __name__ == "__main__": main()
