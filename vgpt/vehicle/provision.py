"""Provision an empty vehicle database from a profile pack: schema, signal catalogue, views, the profile and its retention - ready for the collectors.

    python -m vgpt.vehicle.provision --profile aster_ev --db data/live_aster.db [--replace]
    python -m vgpt.vehicle.provision --check packs/my_car          validate a pack and print the issues

The assistant opens the resulting database read-only; the collectors (vgpt/collect) are its only writer."""
import argparse
import os

from .. import store
from . import lake as LK
from .profile import Profile, ProfileError


def provision(profile, db, replace=False):
    p = profile if isinstance(profile, Profile) else Profile.load(profile)
    if os.path.exists(db):
        if not replace: raise FileExistsError(f"{db} exists; pass --replace to start from an empty database")
        for ext in ("", "-wal", "-shm"):
            try: os.remove(db + ext)
            except OSError: pass
    c = store.open_rw(db); sids = LK.provision(c, p); c.close(); return p, sids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile"); ap.add_argument("--db"); ap.add_argument("--replace", action="store_true"); ap.add_argument("--check")
    a = ap.parse_args()
    if a.check:
        p = Profile.load(a.check); issues = p.validate()
        for i in issues: print(i)
        print(f"{p.id}: {len(p.concept_names())} concepts, {sum(1 for i in issues if i.level == 'error')} errors, {sum(1 for i in issues if i.level == 'warning')} warnings"); raise SystemExit(1 if any(i.level == "error" for i in issues) else 0)
    if not (a.profile and a.db): ap.error("--profile and --db are required")
    try: p, sids = provision(a.profile, a.db, a.replace)
    except (ProfileError, FileExistsError) as e: raise SystemExit(str(e))
    print(f"{p.id}: {len(sids)} signals catalogued ({sum(1 for c in sids if c in LK.lake_concepts(p))} in the lake), retention {p.retention['coarse_days']} days -> {a.db}")


if __name__ == "__main__":
    main()
