"""The shape of an investigation's answer: a headline, then sections that keep what was RECORDED apart from what is only ASSOCIATED or POSSIBLE, and say what could not be determined."""
from dataclasses import dataclass, field


@dataclass
class Report:
    title: str
    template: str
    lines: list = field(default_factory=list)                 # (heading, [text...])
    facts: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)                 # what was computed, in order, for "how did you get that"
    premise: str = None
    evidence: dict = field(default_factory=dict)
    ok: bool = True
    focus_ts: int = None
    focus_event: str = None

    def add(self, heading, *items):
        items = [i for i in items if i]
        if items: self.lines.append((heading, list(items)))
        return self

    def step(self, text): self.steps.append(text)

    def text(self):
        out = [self.title]
        if self.premise: out.append(self.premise)
        for heading, items in self.lines:
            out.append("")
            if heading: out.append(heading + ":")
            for i in items: out.append(i if i.startswith(("- ", "  ")) else "- " + i)
        return "\n".join(out)
