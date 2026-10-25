"""A small matcher for the subset of GBNF that vgpt/plangrammar.py produces (literals, character classes with ranges / negation / {m,n}, rule references, groups, alternation, ? * + and {m,n}).
It exists so that tests and the representability check can ask OFFLINE "does the generation grammar accept this exact call text?" without a model. It is a test utility: the product never calls it, and it never sees a gold interpretation at inference time.
    g = Grammar(text); g.accepts('statistic(turn="new", signal="speed")]')"""
import re
import sys

sys.setrecursionlimit(10000)


class Grammar:
    def __init__(self, text):
        self.rules = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            name, body = line.split("::=", 1); self.rules[name.strip()] = _Parser(body.strip()).parse()
        self._memo = {}

    def accepts(self, s, root="root"):
        self._memo = {}
        return len(s) in self._match(("ref", root), s, 0)

    def _match(self, node, s, i):
        """-> set of end positions."""
        k = node[0]
        if k == "lit": return {i + len(node[1])} if s.startswith(node[1], i) else set()
        if k == "cls":
            _, neg, items = node
            if i >= len(s): return set()
            ch = s[i]; hit = any((a <= ch <= b) for a, b in items)
            return {i + 1} if hit != neg else set()
        if k == "ref":
            key = (node[1], i)
            if key not in self._memo: self._memo[key] = self._match(self.rules[node[1]], s, i)
            return self._memo[key]
        if k == "alt":
            out = set()
            for a in node[1]: out |= self._match(a, s, i)
            return out
        if k == "seq":
            cur = {i}
            for part in node[1]:
                nxt = set()
                for p in cur: nxt |= self._match(part, s, p)
                cur = nxt
                if not cur: return set()
            return cur
        if k == "rep":
            _, inner, lo, hi = node; cur = {i}; out = {i} if lo == 0 else set(); n = 0
            while cur and (hi is None or n < hi):
                nxt = set()
                for p in cur: nxt |= self._match(inner, s, p)
                n += 1
                if n >= lo: out |= nxt
                if n > len(s) + 1: break
                cur = nxt
            return out
        raise ValueError(k)


class _Parser:
    def __init__(self, text): self.t = text; self.i = 0

    def ws(self):
        while self.i < len(self.t) and self.t[self.i] in " \t": self.i += 1

    def parse(self):
        node = self.alt(); self.ws()
        if self.i != len(self.t): raise ValueError(f"unparsed grammar text near {self.t[self.i:self.i + 30]!r}")
        return node

    def alt(self):
        alts = [self.seq()]
        self.ws()
        while self.i < len(self.t) and self.t[self.i] == "|":
            self.i += 1; alts.append(self.seq()); self.ws()
        return alts[0] if len(alts) == 1 else ("alt", alts)

    def seq(self):
        parts = []
        while True:
            self.ws()
            if self.i >= len(self.t) or self.t[self.i] in "|)": break
            parts.append(self.quantified())
        return parts[0] if len(parts) == 1 else ("seq", parts)

    def quantified(self):
        atom = self.atom()
        while self.i < len(self.t):
            c = self.t[self.i]
            if c == "?": self.i += 1; atom = ("rep", atom, 0, 1)
            elif c == "*": self.i += 1; atom = ("rep", atom, 0, None)
            elif c == "+": self.i += 1; atom = ("rep", atom, 1, None)
            elif c == "{":
                m = re.compile(r"\{(\d+)(?:,(\d*))?\}").match(self.t, self.i)
                if not m: break
                lo = int(m.group(1)); hi = lo if m.group(2) is None else (int(m.group(2)) if m.group(2) else None); self.i = m.end(); atom = ("rep", atom, lo, hi)
            else: break
        return atom

    def atom(self):
        c = self.t[self.i]
        if c == '"': return ("lit", self.string())
        if c == "[": return self.cls()
        if c == "(":
            self.i += 1; node = self.alt(); self.ws()
            if self.t[self.i] != ")": raise ValueError("missing )")
            self.i += 1; return node
        m = re.compile(r"[A-Za-z0-9\-]+").match(self.t, self.i)
        if not m: raise ValueError(f"bad grammar text near {self.t[self.i:self.i + 30]!r}")
        self.i = m.end(); return ("ref", m.group(0))

    def esc(self):
        c = self.t[self.i]; self.i += 1
        if c == "n": return "\n"
        if c == "t": return "\t"
        if c == "r": return "\r"
        return c

    def string(self):
        self.i += 1; out = []
        while self.t[self.i] != '"':
            if self.t[self.i] == "\\": self.i += 1; out.append(self.esc())
            else: out.append(self.t[self.i]); self.i += 1
        self.i += 1; return "".join(out)

    def cls(self):
        self.i += 1; neg = False
        if self.t[self.i] == "^": neg = True; self.i += 1
        items = []
        while self.t[self.i] != "]":
            a = self.char()
            if self.t[self.i] == "-" and self.t[self.i + 1] != "]": self.i += 1; b = self.char(); items.append((a, b))
            else: items.append((a, a))
        self.i += 1; return ("cls", neg, items)

    def char(self):
        if self.t[self.i] == "\\": self.i += 1; return self.esc()
        c = self.t[self.i]; self.i += 1; return c
