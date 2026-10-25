"""The single deployable VehicleGPT configuration.

The only learned component is Qwen2.5-0.5B with a LoRA adapter.  Since 2026-10-19 the DEFAULT front end is the meaning frame (vgpt/meaning, docs/semantic_method.md):
the model reads the question into a six-word frame from the ontology vocabulary and typed deterministic code type-checks it, executes it on the database and does the arithmetic.
`VGPT_MEANING=0` selects the previous front end (the four-field semantic classifier + rule layers) unchanged.  The retired autoregressive full-plan/repair pipeline is intentionally
absent from this package.
"""
import os

from .brain import Brain

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def meaning_enabled(slm=None):
    """True when the meaning-frame front end should be built.  It is the default; VGPT_MEANING=0 is the way back.  A caller that hands in its OWN classifier object (anything that is not a
    frame model) is asking for the previous front end and gets it."""
    if os.environ.get("VGPT_MEANING") == "0": return False
    return slm is None or (hasattr(slm, "glossary") and hasattr(slm, "parse"))


def make_qwen05_brain(db, slm=None, persona="driver", now=None, threads=2, meaning=None, **kw):
    """The latency-bounded on-device route used by the UI and app.py.

    DEFAULT (meaning=None, VGPT_MEANING unset): the meaning-frame front end (make_meaning_brain).  meaning=False or VGPT_MEANING=0: the previous front end below.  A missing adapter or base
    model raises FileNotFoundError (the UI shows it); there is no silent switch to the other front end.

    Qwen2.5-0.5B is used as a semantic classifier (four constrained ontology
    tokens), not as an autoregressive writer of a large JSON/tool plan.  The
    typed compiler and deterministic executors then perform schema mapping,
    time/condition parsing, validation and arithmetic.  This is the compact
    formulation that fits the Raspberry Pi latency envelope and it prevents a
    rejected plan from triggering a second multi-second generation.

    The classifier is the only learned model loaded by this factory.
    """
    if meaning if meaning is not None else meaning_enabled(slm):    # the default: the meaning-frame front end (docs/semantic_method.md)
        return make_meaning_brain(db, slm=slm, persona=persona, now=now, threads=threads, **kw)
    from .brain import Slm

    model = os.path.join(HERE, "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
    adapter = os.path.join(HERE, "models", "semantic-lora-qwen25-05b-v2.gguf")
    if not os.path.exists(model):
        raise FileNotFoundError(model)
    if not os.path.exists(adapter):
        raise FileNotFoundError(adapter)

    os.environ["VGPT_SEMANTIC_FRONTDOOR"] = "1"
    semantic = slm
    if semantic is None:
        semantic = Slm(model, threads=threads, n_ctx=896, mmap=True,
                       lora_path=adapter, n_batch=128)
        semantic.warmup()                     # pay the shared-prefix cost at startup
    return Brain(db, slm=semantic, persona=persona, now=now, slm_ir=None,
                 policy="clarify", selector=None, qc_mode="all", **kw)


def _frame_model(adapter, threads):
    """The frame model: the int8 ONNX Runtime build when the product names one (VGPT_ORT_DIR, set from models/product.json) and the runtime is installed, else llama.cpp.  VGPT_BACKEND=llama forces llama.cpp."""
    from .meaning.interface import FrameModel
    d = os.environ.get("VGPT_ORT_DIR")
    if d and os.environ.get("VGPT_BACKEND", "auto") != "llama" and os.path.exists(os.path.join(d, "model.onnx")):
        try:
            from .meaning.ort_backend import OrtFrameModel
            return OrtFrameModel(d, threads=int(os.environ.get("VGPT_THREADS", threads)))       # the scaler / the board's plan can widen the context (Pi 4: 2 cores -> 4 cores)
        except ImportError: pass                                                                          # onnxruntime is not installed: the slower llama.cpp path gives the same frames
    return FrameModel(adapter, threads=threads)


CLARIFY_BELOW = 0.0                      # OFF (owner, 2026-10-24: the model must not ask back). Set VGPT_MEANING_TAU=0.8 to bring "did you mean 1 or 2" back (7 % of questions asked, 42 % of the wrong readings caught; bench/clarify_eval.py)


def set_vehicle_of(model, db):
    """Tell a model trained with the car-type line (VGPT_VEHICLE_HINT=1) what the vehicle is - electric, petrol, diesel or hybrid - from what the database records (vgpt/vehicle/caps.py)."""
    if not hasattr(model, "set_vehicle"): return
    import sqlite3
    from .vehicle import caps
    try:
        c = sqlite3.connect("file:" + str(db).replace("\\", "/") + "?mode=ro", uri=True)
        try: model.set_vehicle(caps.vehicle_kind(c))
        finally: c.close()
    except Exception: model.set_vehicle(None)


def make_meaning_brain(db, slm=None, persona="driver", now=None, threads=2, adapter=None, tau=None, **kw):
    """The meaning-frame front end: Qwen2.5-0.5B + meaning LoRA predicts a frame from the ontology vocabulary; the typed resolver executes it.  The meaning is the model's; the code that still influences it is listed in docs/BOUNDARY.md.
    VGPT_MEANING_ADAPTER selects the adapter file; VGPT_MEANING_TAU (default 0 = off; 0.8 was the old setting) asks back below a token-probability threshold."""
    from .meaning.interface import FrameModel
    from .meaning.pipeline import MeaningBrain
    from .vehicle import lake as _lake
    _lake.adopt_path(db)                                                                                 # the decoding grammar is generated from the registered concepts: adopt the vehicle first
    best = os.path.join(HERE, "models", "meaning-lora-full_v11.gguf")                                    # the final adapter of the 2026-10-20 training (owner stopped training after it, docs/TRAINING_QUEUE.md); VGPT_MEANING_ADAPTER=models/meaning-lora-full_v8.gguf (gate-checked) or full_v6.gguf is the way back; falls back to the first one
    adapter = adapter or os.environ.get("VGPT_MEANING_ADAPTER") or (best if os.path.exists(best) else os.path.join(HERE, "models", "meaning-lora-full.gguf"))
    if slm is not None and hasattr(slm, "glossary") and hasattr(slm, "parse"): model = slm                       # a model handed in (a test double, a model the caller already built) needs no adapter file
    else:
        if not os.path.exists(adapter): raise FileNotFoundError(f"{adapter} (the meaning adapter; set VGPT_MEANING=0 to use the previous front end)")
        model = _frame_model(adapter, threads)
    set_vehicle_of(model, db)
    if hasattr(model, "refresh"): model.refresh()                                                      # a model that was built for another vehicle follows this one
    return MeaningBrain(db, model, persona=persona, now=now, tau=float(os.environ.get("VGPT_MEANING_TAU", tau if tau is not None else CLARIFY_BELOW)), **kw)


def make_brain(db, slm=None, persona="driver", now=None, **kw):
    """Compatibility name for the one supported product configuration."""
    return make_qwen05_brain(db, slm=slm, persona=persona, now=now, **kw)


def make_legacy_brain(db, slm=None, persona="driver", now=None, **kw):
    """The rule-based baseline: rule parser + clarification policy, no model on the answer path."""
    return Brain(db, slm=slm, persona=persona, now=now, slm_ir=None, policy="clarify", selector=None, qc_mode="all", **kw)
