"""Re-rank an already validated development candidate pool without regeneration."""
import argparse
import json
import os

from .full_ir_bottom_up import extended_recall


PRIORITY = {"direct_greedy": 0, "direct_context": 1, "direct_low": 2,
            "direct_diverse": 3, "rules": 4, "lexical": 5, "embedding": 6,
            "typed_mutation": 7, "structural_projection": 8,
            "semantic_repair_1": 9, "semantic_repair_2": 10,
            "full_ir_bottom_up": 11}


def base_key(item):
    sources = item.get("sources", [])
    return (min((PRIORITY.get(source, 99) for source in sources), default=99),
            min(item.get("ranks", {}).values(), default=9999),
            -item.get("frequency", 1), item.get("ir", ""))


def composed_key(item):
    sources = set(item.get("sources", []))
    order = {"full_ir_bottom_up": 0, "semantic_repair_1": 1,
             "semantic_repair_2": 2}
    return min((order[source], item.get("ranks", {}).get(source, 10**9))
               for source in sources if source in order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    campaign = os.path.abspath(args.campaign)
    if any(word in args.input.lower() for word in ("accept", "indep_final")):
        raise SystemExit("sealed evaluation pools are not allowed")
    source_path = os.path.join(campaign, args.input)
    output_path = os.path.join(campaign, args.name + ".jsonl")
    with open(output_path, "w", encoding="utf-8") as target:
        for line in open(source_path, encoding="utf-8"):
            record = json.loads(line); candidates = sorted(record["candidates"], key=base_key)
            anchors = candidates[:10]; anchor_irs = {item["ir"] for item in anchors}
            composed = sorted((item for item in candidates if item["ir"] not in anchor_irs and
                               set(item.get("sources", [])) & {"full_ir_bottom_up", "semantic_repair_1", "semantic_repair_2"}),
                              key=composed_key)
            composed_irs = {item["ir"] for item in composed}
            remainder = [item for item in candidates if item["ir"] not in anchor_irs and
                         item["ir"] not in composed_irs]
            record["candidates"] = anchors + composed + remainder
            target.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(extended_recall(campaign, output_path), indent=2))


if __name__ == "__main__":
    main()
