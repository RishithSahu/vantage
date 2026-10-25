"""The meaning frame: what a question asks, as six words from the ontology's closed vocabulary.

    act concept op cond vs ent

act = kind of question (now / stat / trend / compare / events / why / spec / define / advice ...), concept = what it is about, op = operator, cond = driving condition that restricts the
data, vs = second operand of a comparison, ent = a selected trip / charge session / day.  Literal values (numbers, units, dates, clock times, durations) are NOT part of the frame:
they are parsed from the text by deterministic normalisers (vgpt/meaning/literals.py) because they are values, not meaning."""
from dataclasses import dataclass

from .ontology import fields, FRAME_FIELDS, FRAME2_FIELDS


@dataclass(frozen=True)
class Frame:
    act: str
    concept: str
    op: str = "none"
    cond: str = "none"
    vs: str = "none"
    ent: str = "none"
    c2: str = "none"                    # version 2: the second concept (correlation, two quantities side by side, A or B)
    ref: str = "none"                   # version 2: what this question takes from the previous turn (period, concept, focus, item, answer)

    def to_text(self, version=None): return " ".join(getattr(self, f) for f in fields(version))

    @classmethod
    def from_text(cls, text):
        w = str(text).split()
        if len(w) not in (len(FRAME_FIELDS), len(FRAME2_FIELDS)): raise ValueError(f"a frame has {len(FRAME_FIELDS)} words (or {len(FRAME2_FIELDS)} in version 2), got {len(w)}: {text!r}")
        return cls(*w)

    def as_dict(self, version=None): return {f: getattr(self, f) for f in fields(version)}
