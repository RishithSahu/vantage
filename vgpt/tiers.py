"""Resource tiers: one application, several resource profiles (deploy/tiers.json).

    from vgpt import tiers
    t = tiers.current()                       # VGPT_TIER=pi5 selects one; default is the reference board
    t.model["threads"], t.sqlite["cache_kb"], t.retention["coarse_days"], t.sessions

Only numbers differ between tiers - never an interface.  `store.open_ro / open_rw` take their page cache from the tier, `Runtime(tier=...)` writes the tier's retention into the database it owns
(a database moved to a bigger box keeps more history from then on), `sessions.ModelPool` starts the tier's number of model contexts and `sessions.SessionHub` admits the tier's number of sessions."""
import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(HERE, "deploy", "tiers.json")
_cache = {}


@dataclass
class Tier:
    name: str
    label: str
    ram_gb: float
    cores: int
    model: dict = field(default_factory=dict)
    sqlite: dict = field(default_factory=dict)
    sessions: int = 2
    ring_s: int = 420
    retention: dict = field(default_factory=dict)
    notes: str = ""

    def model_files(self, models_dir=None):
        d = models_dir or os.path.join(HERE, "models"); m = self.model
        return os.path.join(d, m["base"]), (os.path.join(d, m["adapter"]) if m.get("adapter") else None)

    def available(self):
        base, adapter = self.model_files(); return os.path.exists(base) and (adapter is None or os.path.exists(adapter))


def load(path=PATH):
    if path not in _cache:
        raw = json.load(open(path, encoding="utf-8")); _cache[path] = (raw.get("default", "edge2g"), {k: Tier(name=k, **{f: v for f, v in t.items() if f in Tier.__dataclass_fields__ and f != "name"}) for k, t in raw["tiers"].items()})
    return _cache[path]


def names(): return list(load()[1])


def get(name=None):
    default, tiers = load(); name = name or os.environ.get("VGPT_TIER") or default
    if name not in tiers: raise KeyError(f"unknown tier {name!r}; known: {', '.join(tiers)}")
    return tiers[name]


def current(): return get()


def sqlite_cache_kb(default=4096):
    """Page cache for a database handle: the tier's when VGPT_TIER is set, else the default every caller used before tiers existed."""
    if not os.environ.get("VGPT_TIER"): return default
    return int(get().sqlite.get("cache_kb", default))


def retention_for(profile_retention, tier=None):
    """The retention to provision with: the profile's own, raised (never lowered) to the tier's where the tier keeps more."""
    t = get(tier) if (tier or os.environ.get("VGPT_TIER")) else None
    if t is None: return dict(profile_retention)
    return {k: max(int(profile_retention.get(k, 0)), int(t.retention.get(k, 0))) for k in set(profile_retention) | set(t.retention)}
