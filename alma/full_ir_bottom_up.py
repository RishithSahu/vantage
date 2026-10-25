"""Gold-blind bottom-up recombination of complete typed-IR candidate atoms."""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import os
import re
import sqlite3
from collections import Counter

from bench.indep_io import load
from training.build_dataset import DEFAULT_NOW
from vgpt import nl2ir, signal_registry as R
from vgpt.qc import QueryCompiler
from vgpt.query_ir import (AGGS, OPS, Condition, EntitySelect, EventRef, IRError,
                           Query)
from .final_campaign import (DEV_PATH, ROOT, WORLD_NAMES, Pool, evaluate_candidate_pools,
                             full_key, sha256_file, valid_program)


UNIT_WORDS = {"mph": "mph", "miles per hour": "mph", "psi": "psi", "fahrenheit": "F", "bar": "bar",
              "kpa": "kPa", "kw": "kW", "rpm": "rpm", "nm": "Nm",
              "l/min": "L/min", "km/h": "km/h", "percent": "%", "%": "%"}

# Grammar-level signatures.  This is deliberately an ontology inventory, not a
# phrase-to-intent map: language scoring decides among these alternatives later.
NO_AGG_OPS = {"current_value", "trend", "count", "list_events", "event_context",
              "rank", "correlation", "percent_time", "explain_event",
              "capability", "explain_change", "detect_change"}
STANDARD_CONTEXT_WINDOWS = (300, 600, 900, 1800)


def atom_key(value):
    if value is None: return "null"
    if isinstance(value, tuple): return json.dumps([atom_key(item) for item in value], sort_keys=True)
    if isinstance(value, list): return json.dumps([getattr(item, "to_text", lambda: str(item))()
                                                   for item in value], sort_keys=True)
    if hasattr(value, "to_text"): return value.to_text()
    return json.dumps(value, sort_keys=True, default=str)


def add_atom(store, value, rank):
    key = atom_key(value)
    if key not in store or rank < store[key][0]: store[key] = (rank, copy.deepcopy(value))


def inferred_units(text):
    low = text.lower(); values = []
    for word, unit in UNIT_WORDS.items():
        if word == "%":
            present = "%" in low
        else:
            present = re.search(r"\b" + re.escape(word) + r"\b", low) is not None
        if present and unit not in values: values.append(unit)
    return values


def atoms(row, candidates, profile):
    question = row["text"]
    stores = {name: {} for name in ("signature", "targets", "times", "conditions",
                                     "selections", "groups", "event", "meta")}
    units = inferred_units(question); group_heads = set()
    seeded = list(candidates)
    if row.get("prev_ir"):
        seeded.append({"ir": row["prev_ir"]})
    state = nl2ir.ConvState()
    if row.get("prev_ir"):
        try: state.last = Query.from_text(row["prev_ir"])
        except IRError: pass
    parsed = nl2ir.parse(question, now=DEFAULT_NOW, tz=3600, profile=profile,
                         state=state)
    if parsed.query is not None and getattr(parsed.query, "op", ""):
        seeded.append({"ir": parsed.query.to_text()})
    for rank, item in enumerate(seeded, 1):
        try: q = Query.from_text(item["ir"])
        except IRError: continue
        add_atom(stores["signature"], (q.op, q.agg), rank)
        add_atom(stores["targets"], (q.target, q.target2), rank)
        add_atom(stores["times"], (q.scope.time, q.scope_b.time if q.scope_b else None), rank)
        add_atom(stores["conditions"], (q.scope.conditions,
                                        q.scope_b.conditions if q.scope_b else None), rank)
        add_atom(stores["selections"], (q.scope.select,
                                        q.scope_b.select if q.scope_b else None), rank)
        if q.scope.select:
            for metric in R.ENTITY_METRICS.get(q.scope.select.entity, {}):
                variant = copy.deepcopy(q.scope.select); variant.rank_by = metric
                add_atom(stores["selections"], (variant,
                                                q.scope_b.select if q.scope_b else None), rank + 2)
        add_atom(stores["groups"], q.groups, rank)
        if q.groups: group_heads.add(q.groups[0])
        add_atom(stores["event"], q.event, rank)
        add_atom(stores["meta"], (q.unit, q.refs), rank)
        # Unit copying is a generic semantic atom operation, not a phrase-to-intent rule.
        for unit in units:
            if q.scope.conditions:
                changed = copy.deepcopy(q.scope.conditions)
                for condition in changed:
                    if condition.signal and condition.value is not None: condition.unit = unit
                add_atom(stores["conditions"], (changed,
                                                q.scope_b.conditions if q.scope_b else None), rank + 1)
                for condition in q.scope.conditions:
                    if condition.signal and condition.value is not None:
                        single = copy.deepcopy(condition); single.unit = unit
                        add_atom(stores["conditions"], ([single], None), rank + 1)
        if q.event:
            event = copy.deepcopy(q.event); event.before = 0; event.after = 0
            add_atom(stores["event"], event, rank + 1)
            for selector in ("last", "first", "all"):
                event = copy.deepcopy(q.event); event.selector = selector; event.n = None
                add_atom(stores["event"], event, rank + 2)
            # Context length is a bounded typed value.  Supplying the common
            # windows lets the verifier choose it without coupling language
            # understanding to a one-off wording rule.
            for seconds in STANDARD_CONTEXT_WINDOWS:
                event = copy.deepcopy(q.event); event.before = seconds
                add_atom(stores["event"], event, rank + 3)
    # Seed atoms from the frozen schema linker/normalizers even when they cannot
    # assemble a valid full program. This is the intended division of labour.
    # These atoms are question-conditioned outputs of protected, accepted
    # components, so they receive the same priority as a top generator atom.
    normalized = nl2ir.normalise(question); semantic_rank = 1
    for time_spec in nl2ir._times(nl2ir._S(normalized), DEFAULT_NOW, 3600):
        add_atom(stores["times"], (time_spec, None), semantic_rank)
    concept_names = []
    for match in nl2ir._CONCEPT_RX.finditer(normalized):
        concept_names.extend(sorted(R.LEXICON.get(match.group(1).replace("-", " "), ())))
    for name in dict.fromkeys(concept_names):
        add_atom(stores["targets"], (name, None), semantic_rank)
    for hint in list(dict.fromkeys(concept_names)) + [None]:
        try:
            found = nl2ir._conditions(nl2ir._S(normalized), profile, hint)
        except (KeyError, ValueError):
            found = []
        if found: add_atom(stores["conditions"], (found, None), semantic_rank)
        # A parser may emit a named condition and a numeric threshold together.
        # Keep their atomic alternatives so composition can express either.
        for condition in found:
            add_atom(stores["conditions"], ([condition], None), semantic_rank + 1)
            if condition.signal and condition.value is not None:
                for unit in units:
                    changed = copy.deepcopy(condition); changed.unit = unit
                    add_atom(stores["conditions"], ([changed], None), semantic_rank + 1)
    try:
        selection = nl2ir._selection(nl2ir._S(normalized), normalized)
    except (KeyError, ValueError):
        selection = None
    if selection: add_atom(stores["selections"], (selection, None), semantic_rank)
    if selection:
        for metric in R.ENTITY_METRICS.get(selection.entity, {}):
            variant = copy.deepcopy(selection); variant.rank_by = metric
            add_atom(stores["selections"], (variant, None), semantic_rank + 2)
    predicates = []
    for match in nl2ir._PRED_RX.finditer(normalized):
        phrase = match.group(0).strip()
        value = nl2ir._PRED_PHRASES.get(phrase)
        if value and value not in predicates: predicates.append(value)
    if predicates:
        add_atom(stores["groups"], predicates, semantic_rank)
        add_atom(stores["conditions"], ([Condition(predicate=value) for value in predicates], None), semantic_rank)
        # Comparisons need two compatible modes.  Enumerating the small typed
        # ontology prevents one mistaken second mode from destroying recall.
        for first in predicates:
            if first in R.MODE_PREDICATES:
                for second in R.MODE_PREDICATES:
                    if second != first: add_atom(stores["groups"], [first, second], semantic_rank + 3)
    for first in group_heads:
        for second in R.PREDICATES:
            if second != first: add_atom(stores["groups"], [first, second], semantic_rank + 3)
    # The comparison-mode ontology is tiny and closed.  Exposing all valid
    # ordered pairs is safer than letting one weak lexical guess erase coverage.
    for first in R.MODE_PREDICATES:
        for second in R.MODE_PREDICATES:
            if second != first: add_atom(stores["groups"], [first, second], 12)
    # Always expose the finite signature grammar.  These alternatives are
    # intentionally lower priority than question-conditioned predictions.
    for op in OPS:
        if op in NO_AGG_OPS:
            add_atom(stores["signature"], (op, None), 24)
        else:
            for agg in AGGS: add_atom(stores["signature"], (op, agg), 28)
    for unit in units:
        add_atom(stores["meta"], (unit, []), semantic_rank)
    limits = {"signature": 48, "targets": 24, "times": 30, "conditions": 30,
              "selections": 24, "groups": 16, "event": 20, "meta": 10}
    return [[value for _, value in sorted(store.values(), key=lambda item: item[0])[:limits[name]]]
            for name, store in stores.items()], [
                [rank for rank, _ in sorted(store.values(), key=lambda item: item[0])[:limits[name]]]
                for name, store in stores.items()]


def compose(atom_values):
    signature, targets, times, conditions, selections, groups, event, meta = atom_values
    op, agg = signature; target, target2 = targets; time_a, time_b = times
    cond_a, cond_b = conditions; select_a, select_b = selections; unit, refs = meta
    # Start from any syntactically complete query, then replace all semantic components.
    q = Query(op=op, target=target, target2=target2, agg=agg, groups=copy.deepcopy(groups),
              event=copy.deepcopy(event), unit=unit, refs=copy.deepcopy(refs))
    q.scope.time = copy.deepcopy(time_a); q.scope.conditions = copy.deepcopy(cond_a)
    q.scope.select = copy.deepcopy(select_a)
    if time_b is not None:
        from vgpt.query_ir import Scope
        q.scope_b = Scope(copy.deepcopy(time_b), copy.deepcopy(select_b), copy.deepcopy(cond_b or []))
    return q.to_text()


def decompose(q):
    """A complete query as the same typed component tuple consumed by compose."""
    return [(q.op, q.agg), (q.target, q.target2),
            (q.scope.time, q.scope_b.time if q.scope_b else None),
            (q.scope.conditions, q.scope_b.conditions if q.scope_b else None),
            (q.scope.select, q.scope_b.select if q.scope_b else None), q.groups,
            q.event, (q.unit, q.refs)]


def local_recombinations(bases, values, ranks):
    """Preserve coherent attachments while exploring one/two component edits.

    Global Cartesian k-best search spends most of its budget mixing unrelated
    low-cost atoms.  This bounded neighbourhood instead asks which typed edit to
    an already coherent complete query could be intended.  It is gold-blind and
    identical for every question.
    """
    for base_rank, base in enumerate(bases, 1):
        try: anchor = decompose(Query.from_text(base))
        except IRError: continue
        for axis, alternatives in enumerate(values):
            for index, alternative in enumerate(alternatives):
                changed = copy.deepcopy(anchor); changed[axis] = alternative
                yield base_rank + ranks[axis][index], 1, changed
        # Pairwise edits use the best atoms on each axis.  Signatures receive a
        # wider allowance because operation/aggregation is one coupled type.
        for left in range(len(values)):
            left_limit = 20 if left == 0 else 8
            for right in range(left + 1, len(values)):
                right_limit = 20 if right == 0 else 8
                for li, lv in enumerate(values[left][:left_limit]):
                    for ri, rv in enumerate(values[right][:right_limit]):
                        changed = copy.deepcopy(anchor); changed[left] = lv; changed[right] = rv
                        yield base_rank + ranks[left][li] + ranks[right][ri], 2, changed


def kbest_combinations(lists, ranks, maximum=4000):
    if any(not values for values in lists): return []
    start = tuple(0 for _ in lists)
    def cost(indexes): return sum(ranks[axis][value] for axis, value in enumerate(indexes))
    heap = [(cost(start), start)]; seen = {start}; output = []
    while heap and len(output) < maximum:
        score, indexes = heapq.heappop(heap)
        output.append((score, [lists[axis][value] for axis, value in enumerate(indexes)]))
        for axis in range(len(lists)):
            if indexes[axis] + 1 >= len(lists[axis]): continue
            nxt = list(indexes); nxt[axis] += 1; nxt = tuple(nxt)
            if nxt not in seen: seen.add(nxt); heapq.heappush(heap, (cost(nxt), nxt))
    return output


def build(input_path, output_path, max_valid=2000, semantic_first=False):
    rows = load(DEV_PATH)
    source = [json.loads(line) for line in open(input_path, encoding="utf-8")]
    conn = sqlite3.connect(f"file:{os.path.join(ROOT, 'data', WORLD_NAMES[0])}?mode=ro", uri=True)
    profile = QueryCompiler(conn, now=DEFAULT_NOW).profile
    counts = []
    with open(output_path, "w", encoding="utf-8") as handle:
        for row, record in zip(rows, source):
            pool = Pool(profile)
            for rank, item in enumerate(record["candidates"], 1):
                for source_name in item["sources"]: pool.add(item["ir"], source_name, rank)
            # The protected symbolic parser is itself a legitimate candidate.
            # Earlier campaigns used it only as an atom donor here, which could
            # lose an already-complete correct program during recombination.
            state = nl2ir.ConvState()
            if row.get("prev_ir"):
                try: state.last = Query.from_text(row["prev_ir"])
                except IRError: pass
            parsed = nl2ir.parse(row["text"], now=DEFAULT_NOW, tz=3600,
                                 profile=profile, state=state)
            if parsed.query is not None:
                pool.add(parsed.query.to_text(), "rules", 1)
            lists, ranks = atoms(row, record["candidates"], profile)
            coherent = [item["ir"] for item in record["candidates"][:4]]
            if row.get("prev_ir"): coherent.append(row["prev_ir"])
            if parsed.query is not None: coherent.append(parsed.query.to_text())
            for score, edits, combination in local_recombinations(coherent, lists, ranks):
                try: candidate = compose(combination)
                except (IRError, ValueError, TypeError): continue
                pool.add(candidate, f"semantic_repair_{edits}", int(score))
            added = 0
            for score, combination in kbest_combinations(lists, ranks):
                try: candidate = compose(combination)
                except (IRError, ValueError, TypeError): continue
                before = len(pool._items)
                pool.add(candidate, "full_ir_bottom_up", int(score))
                added += len(pool._items) > before
                if added >= max_valid: break
            candidates = pool.ordered()
            if semantic_first:
                anchors = candidates[:10]
                anchor_ids = {id(item) for item in anchors}
                composed_sources = {"semantic_repair_1", "semantic_repair_2", "full_ir_bottom_up"}
                composed = sorted((item for item in candidates if id(item) not in anchor_ids and
                                   item.sources & composed_sources),
                                  key=lambda item: min(
                                      (0 if source == "full_ir_bottom_up" else
                                       1 if source == "semantic_repair_1" else 2,
                                       item.ranks.get(source, 10**9))
                                      for source in item.sources & composed_sources))
                composed_ids = {id(item) for item in composed}
                remainder = [item for item in candidates if id(item) not in anchor_ids and
                             id(item) not in composed_ids]
                candidates = anchors + composed + remainder
            handle.write(json.dumps({"id": row["id"], "index": record["index"],
                                     "text": row["text"],
                                     "candidates": [item.as_dict() for item in candidates]},
                                    sort_keys=True) + "\n")
            counts.append(len(candidates))
    conn.close(); return counts


def extended_recall(output_dir, pool_path):
    rows = load(DEV_PATH); pools = [json.loads(line) for line in open(pool_path, encoding="utf-8")]
    cutoffs = (1, 5, 10, 50, 100, 500, 1000, 2000)
    exact = Counter(); per_question = []
    for row, pool in zip(rows, pools):
        gold = full_key(row["ir"])
        rank = next((index + 1 for index, item in enumerate(pool["candidates"])
                     if full_key(item["ir"]) == gold), None)
        for cutoff in cutoffs: exact[cutoff] += rank is not None and rank <= cutoff
        per_question.append({"id": row["id"], "family": row["family"], "exact_rank": rank})
    result = {"n": len(rows), "pool_sha256_before_evaluation": sha256_file(pool_path),
              "exact_full_ir_candidate_recall": {str(k): exact[k] / len(rows) for k in cutoffs},
              "missing_at_2000": len(rows) - exact[2000], "per_question": per_question,
              "acceptance_or_final_opened": False}
    report_name = os.path.splitext(os.path.basename(pool_path))[0] + "_recall.json"
    with open(os.path.join(output_dir, report_name), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--campaign", required=True)
    parser.add_argument("--max-valid", type=int, default=2000)
    parser.add_argument("--name", default="candidate_pools_bottom_up")
    parser.add_argument("--semantic-first", action="store_true")
    args = parser.parse_args()
    campaign = os.path.abspath(args.campaign)
    source = os.path.join(campaign, "candidate_pools.jsonl")
    output = os.path.join(campaign, args.name + ".jsonl")
    counts = build(source, output, args.max_valid, args.semantic_first); result = extended_recall(campaign, output)
    result["pool_size"] = {"mean": sum(counts) / len(counts), "min": min(counts), "max": max(counts)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
