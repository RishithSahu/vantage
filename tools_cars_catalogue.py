"""Writes docs/CAR_CATALOGUE.md: every car model of the test bench, its maker, powertrain, the buses and protocols it carries and how many quantities its decoder file binds - read from the packs, not typed.

    python tools_cars_catalogue.py"""
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from vgpt.vehicle.profile import Profile

PROTOCOL = {"can": "CAN", "someip": "SOME/IP", "vhal": "Android VHAL", "uds": "UDS/OBD", "log": "logs", "gnss": "NMEA GNSS", "camera": "camera", "lin": "LIN", "flexray": "FlexRay"}


def row(p):
    kinds = collections.Counter(); extra = []
    for s in p.sources.values():
        k = s.type
        if k == "can":
            if s.raw.get("canopen"): extra.append("CANopen")
            elif s.raw.get("j1939"): extra.append("J1939")
            elif s.raw.get("fd"): extra.append("CAN-FD")
            else: kinds["can"] += 1
        elif k == "uds" and s.raw.get("transport") == "doip": extra.append("DoIP")
        else: kinds[k] += 1
    names = [PROTOCOL[k] for k in PROTOCOL if kinds.get(k)] + sorted(set(extra))
    return names


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "vgpt", "vehicle", "packs", "*", "profile.json"))):
        p = Profile.load(os.path.dirname(d)); v = p.vehicle; rows.append((str(v.get("make", "")), str(v.get("model", p.id)), v.get("year", ""), v.get("powertrain", ""), p.id, row(p), len(p.concept_names())))
    rows.sort(key=lambda r: (r[0].lower(), r[1]))
    makers = sorted({r[0] for r in rows if r[0]}); protocols = collections.Counter(x for r in rows for x in r[5])
    out = ["# The cars of the test bench", "", f"{len(rows)} car models from {len(makers)} makers, generated from the decoder files (`python tools_cars_catalogue.py`). Real-vehicle models use the open CAN databases of opendbc (MIT licence); "
           "the models of a *synthetic maker* (Aster, Kestrel, Meridian, Rhino, Nova, Volta, Prestige) are written for the test bench and say so. Every car's traffic is recognised from the bytes alone (`tests/test_pipe.py`).", "",
           "Protocols and buses across the cars: " + ", ".join(f"{k} ({n})" for k, n in sorted(protocols.items(), key=lambda kv: -kv[1])) + ".", "", "| Maker | Model | Year | Powertrain | Buses and protocols | Quantities | Pack id |", "|---|---|---|---|---|---|---|"]
    for mk, md, yr, pt, pid, names, n in rows: out.append(f"| {mk} | {md} | {yr} | {pt} | {', '.join(names)} | {n} | `{pid}` |")
    open(os.path.join(ROOT, "docs", "CAR_CATALOGUE.md"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n"); print(len(rows), "cars,", len(makers), "makers")


if __name__ == "__main__":
    main()
