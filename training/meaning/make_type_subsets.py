"""Split the round-10 training data by car type, for the comparison "one model told the car type" against "one adapter per car type".

    python training/meaning/make_type_subsets.py --data training/meaning/data/frame2_r10

Writes <data>_ice (petrol, diesel and combustion rows + `keep` of the rest) and <data>_ev (electric rows + `keep` of the rest).  The rest is kept in part so that an adapter specialised on one kind of car does not forget the
questions that do not depend on the car.  The validation and composition splits are the same as the parent's (filtered to the same car types), so that train_lora's overlap check passes.
Each subset is meant for a SHORT fine-tune from the shared adapter:
    VGPT_FRAME=2 VGPT_VEHICLE_HINT=1 python training/meaning/train_meaning.py --data <data>_ice --out training/out/meaning_frame2r10_ice --epochs 1 --bs 16 --lr 1e-4 --rank 32 --alpha 64 --seed 2 --init-adapter training/out/meaning_frame2r10/adapter.safetensors"""
import argparse
import io
import json
import os
import random

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--keep", type=float, default=0.25); ap.add_argument("--seed", type=int, default=3); a = ap.parse_args()
rnd = random.Random(a.seed); ICE = {"petrol", "diesel", "ice"}


def rows(name): return [json.loads(l) for l in io.open(os.path.join(a.data, name + ".jsonl"), encoding="utf-8")]


for tag, want in (("ice", ICE), ("ev", {"ev"})):
    out = a.data + "_" + tag; os.makedirs(out, exist_ok=True)
    for name in ("train", "val", "test_comp"):
        part = []
        for r in rows(name):
            if r.get("kind") != "frame": part.append(r) if rnd.random() < a.keep else None; continue
            if r.get("vehicle") in want or (r.get("vehicle") not in ICE | {"ev"} and tag == "ice" and False): part.append(r)
            elif rnd.random() < a.keep: part.append(r)
        with io.open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as fh:
            for r in part: fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        print(tag, name, len(part))
    io.open(os.path.join(out, "manifest.json"), "w", encoding="utf-8").write(json.dumps({"parent": a.data, "type": tag, "keep_of_the_rest": a.keep}, indent=1))

# the compute-matched control: the same number of rows as the combustion subset, drawn at random from everything
out = a.data + "_mix"; os.makedirs(out, exist_ok=True); n_ice = sum(1 for _ in io.open(os.path.join(a.data + "_ice", "train.jsonl"), encoding="utf-8"))
tr = rows("train"); rnd.shuffle(tr)
for name, part in (("train", tr[:n_ice]), ("val", rows("val")[:641]), ("test_comp", rows("test_comp")[:121])):
    with io.open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as fh:
        for r in part: fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print("mix", name, len(part))
io.open(os.path.join(out, "manifest.json"), "w", encoding="utf-8").write(json.dumps({"parent": a.data, "type": "mix", "note": "random rows, as many as the combustion subset"}, indent=1))
