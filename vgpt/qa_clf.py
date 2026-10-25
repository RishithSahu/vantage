"""Linear question classifier (the third opinion next to the keyword routes and the nearest-example router): tf-idf of the synonym-mapped words (1-2 grams) and of character 3-5 grams, one linear score per
route. Trained offline by tools_train_clf.py (scikit-learn LinearSVC); here only the weights are used, with numpy - no scikit-learn on the car. Deterministic, ~5 ms per question.
Used to (a) pick between two plausible everyday routes, (b) answer questions no keyword rule matched, and (c) recognise questions that are not about the car at all."""
import math, os, re
from collections import Counter
import numpy as np

_PATH = os.path.join(os.path.dirname(__file__), "data", "qa_clf.npz")
EVERYDAY = {"briefing", "explain_simple", "pump_ok", "v12", "battery_now", "best_day", "eff_day", "last_charge", "charge_cost", "charge_time", "speeding", "improve_range", "hot_battery", "unusual", "eff_trend", "battery_age", "motor_now", "overview", "reach", "advice",
            "longest_trip", "last_trip", "charging_now", "style", "style_trend", "dtc_explain", "top_speed", "soh_rate", "places_most", "commute", "leave_time", "compare_months", "compare_days", "trip_count", "regen", "signal_why", "drive_safely"}
# decision thresholds (tuned on bench/heldout_dev.tsv with a model trained without it): override a keyword route / answer when no keyword matched / give up on the question as off-topic
T = {"override": (-0.45, 0.60), "any": (-0.30, 0.90), "answer": (-0.30, 0.25), "last": (-0.5, 0.15), "gate": (-0.6, 0.05), "convo": (0.15, 0.3)}      # (minimum score of the best route, minimum lead over the keyword route / the runner-up)
GENERIC = {"events", "trend", "point", "extreme", "speedwin", "duration", "longest", "corr", "badday", "worstday", "history", "compare_periods", "conditions", "changes", "why_range2", "earliest", "day", "signals_during"}       # older broad keyword routes the model may overrule
PROTECT = {"safety", "poison", "inject", "wrong", "memory", "prove", "conf", "missing", "unknowable", "precision", "nuclear1", "nuclear3", "c1", "c2", "c3", "c4", "c5", "trace", "evidence", "alt", "cutoff", "conf_change", "used_records", "counter", "calc", "obs_inf"}


def word_features(q):
    from .qa_fuzzy import tokens
    from .qa import prep
    t = [w for w in tokens(prep(q)) if len(w) > 1]
    return t + [a + " " + b for a, b in zip(t, t[1:])]


def char_features(q):
    from .qa import prep
    out = []
    for w in prep(q).split():
        w = " " + w + " "
        for n in (3, 4, 5): out += [w[i:i + n] for i in range(len(w) - n + 1)]
    return out


class Clf:
    def __init__(self, path=None):
        d = np.load(path or _PATH, allow_pickle=False)
        self.classes = [str(c) for c in d["classes"]]; self.W = d["W"].astype(np.float32); self.b = d["b"].astype(np.float32)
        self.wv = {w: i for i, w in enumerate(d["w_vocab"].tolist())}; self.cv = {w: i for i, w in enumerate(d["c_vocab"].tolist())}
        self.widf = d["w_idf"].astype(np.float32); self.cidf = d["c_idf"].astype(np.float32); self.nw = len(self.wv)

    def _block(self, feats, vocab, idf, off, idx, val):
        c = Counter(f for f in feats if f in vocab)
        if not c: return
        v = np.array([(1 + math.log(k)) * idf[vocab[f]] for f, k in c.items()], dtype=np.float32); v /= (np.linalg.norm(v) or 1.0)
        for (f, _), x in zip(c.items(), v): idx.append(off + vocab[f]); val.append(x)

    def scores(self, q):
        idx, val = [], []
        self._block(word_features(q), self.wv, self.widf, 0, idx, val); self._block(char_features(q), self.cv, self.cidf, self.nw, idx, val)
        if not idx: return None
        return self.W[:, idx] @ np.array(val, dtype=np.float32) + self.b

    def score_of(self, s, route):
        """The raw score the model gives one particular route (about -1 for 'clearly not this', about +1 for 'clearly this')."""
        try: return float(s[self.classes.index(route)])
        except ValueError: return -1.0

    def predict(self, q, k=3):
        """-> [(route, score)] best first, plus the margin between the first two."""
        s = self.scores(q)
        if s is None: return [], 0.0
        order = np.argsort(-s)[:k]; top = [(self.classes[i], float(s[i])) for i in order]
        self.last_scores = s
        return top, (top[0][1] - top[1][1]) if len(top) > 1 else 9.0


_INST = None


def get():
    global _INST
    if _INST is None:
        try: _INST = Clf() if os.path.exists(_PATH) else False
        except Exception: _INST = False
    return _INST or None
