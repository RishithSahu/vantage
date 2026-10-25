"""Tool-call dialects: how each frozen model's own chat template writes a tool call.

The plan language (functions, typed arguments, the grammar's value types) is the same for every model. What differs is the WRAPPER, and each model is used in its documented form:
  LFM2.x   tools declared in the system turn as JSON, the call written in Pythonic form:   <|tool_call_start|>[fn(arg="value", ...)]<|tool_call_end|>
  Granite 4  OpenAI-style function schemas inside <tools>, the call written as JSON:       <tool_call>\\n{"name": "fn", "arguments": {...}}\\n</tool_call>
A dialect supplies: the text that opens the assistant turn, how a call (function + arguments) is written for examples / conversation history, how a generated call is parsed, the tool declarations handed to the chat template, and the grammar style."""
import json
import re

from . import schema as S
from . import toolspec as T


class Dialect:
    name = ""; style = "py"; open_text = ""; grounded = True                                  # grounded: the call carries the quoted words of each optional argument (checked by plancompile.ground)

    def assistant_text(self, fn, args): raise NotImplementedError
    def inline(self, fn, args): raise NotImplementedError
    def parse(self, raw): raise NotImplementedError
    def complete(self, raw): raise NotImplementedError
    def declarations(self, registry, level="rich"): raise NotImplementedError


class Pythonic(Dialect):
    name = "lfm"; style = "py"; open_text = f"{T.CALL_OPEN}["

    def inline(self, fn, args): return T.args_to_pythonic(fn, args)
    def assistant_text(self, fn, args): return f"{T.CALL_OPEN}[{self.inline(fn, args)}]{T.CALL_CLOSE}"
    def parse(self, raw): return T.parse_call("[" + raw)
    def complete(self, raw): return raw.rstrip().endswith("]")
    def declarations(self, registry, level="rich"): return [json.dumps(f, separators=(",", ":")) for f in S.function_schemas(registry, level)]            # one JSON declaration per function in the system turn ("List of tools: [...]")


class JsonCall(Dialect):
    name = "granite"; style = "json"; open_text = "<tool_call>\n"

    def inline(self, fn, args): return json.dumps({"name": fn, "arguments": args}, ensure_ascii=False)
    def assistant_text(self, fn, args): return f"<tool_call>\n{self.inline(fn, args)}\n</tool_call>"
    def complete(self, raw): return raw.rstrip().endswith("}")

    def parse(self, raw):
        try: obj = json.loads(raw.strip())
        except ValueError as e: raise T.CallError(f"not well-formed JSON: {e}")
        if not isinstance(obj, dict) or set(obj) != {"name", "arguments"} or not isinstance(obj["arguments"], dict): raise T.CallError("not a {name, arguments} call")
        fn, args = obj["name"], obj["arguments"]
        if fn not in T.FUNCS: raise T.CallError(f"unknown function {fn!r}")
        T.check_args(fn, args)
        return fn, args

    def declarations(self, registry, level="rich"): return [{"type": "function", "function": f} for f in S.function_schemas(registry, level)]            # OpenAI-style function schemas inside <tools>


class XLam(JsonCall):
    """The native form of the function-calling specialist xLAM-2: the tools are declared in the system turn by the checkpoint's own chat template and the call is a JSON ARRAY of {name, arguments} objects (here exactly one element). Arguments are the plain values - no quoted-words companions (grounded=False): a missing period / condition / unit is simply left out and stays unspecified."""
    name = "xlam"; style = "jsonarray"; open_text = ""; grounded = False

    def assistant_text(self, fn, args): return "[" + self.inline(fn, args) + "]"
    def complete(self, raw): return raw.rstrip().endswith("]")

    def parse(self, raw):
        try: arr = json.loads(raw.strip())
        except ValueError as e: raise T.CallError(f"not well-formed JSON: {e}")
        if not isinstance(arr, list) or len(arr) != 1: raise T.CallError("expected a JSON array with exactly one call")
        return JsonCall.parse(self, json.dumps(arr[0]))

    def declarations(self, registry, level="rich"): return [{"type": "function", "function": f} for f in S.function_schemas(registry, level, grounded=False)]


class QwenCall(JsonCall):
    """Qwen2.5's documented tool-call form (<tool_call>{name, arguments}</tool_call>) with plain arguments (no quoted-words companions): the reference-model interface."""
    name = "qwen"; grounded = False

    def declarations(self, registry, level="rich"): return [{"type": "function", "function": f} for f in S.function_schemas(registry, level, grounded=False)]


class XLamReversed(XLam):
    """Diagnostic only: the same declarations in the opposite order (is the choice driven by the position of a tool in the list?)."""
    name = "xlam_rev"

    def declarations(self, registry, level="rich"): return XLam.declarations(self, registry, level)[::-1]


class Frame(Dialect):
    """Not a tool call: ONE plain line, `operation | turn="new" | field=value | ... | field=none`. Every field of the operation is written; an absent one is an explicit `none`, and in a follow-up `keep` leaves a field as it was. The model needs no tool declarations: the operations and fields are listed in the instruction (generated from the same definition)."""
    name = "frame"; style = "frame"; open_text = ""; grounded = False                          # no quotations: absence is explicit instead

    def _fields(self, fn, args):
        new = args.get("turn", "new") == "new"; dropped = set(args.get("drop") or []); out = []
        for pn, pt in T.frame_fields(fn):
            if pn in args: v = json.dumps(args[pn], ensure_ascii=False)
            elif pn == "turn": continue
            elif new or pn == "conditions_edit": v = "none"
            else: v = "none" if T.FRAME_DROP.get(pn) in dropped else "keep"
            out.append(f"{pn}={v}")
        return out

    def inline(self, fn, args):
        head = [fn] + ([f"turn={json.dumps(args.get('turn', 'new'))}"] if fn not in T.NO_TURN else [])
        return " | ".join(head + self._fields(fn, args))

    def assistant_text(self, fn, args): return self.inline(fn, args)
    def complete(self, raw): return bool(raw.strip())
    def declarations(self, registry, level="rich"): return None

    def parse(self, raw):
        text = raw.strip().split("\n")[0].strip()
        if not text: raise T.CallError("empty frame")
        parts = text.split(" | ", 1); fn = parts[0].strip()
        if fn not in T.FUNCS: raise T.CallError(f"unknown operation {fn!r}")
        rest = parts[1] if len(parts) > 1 else ""; args = {}; drop = []; dec = json.JSONDecoder(); i = 0; names = {p[0] for p in T.frame_fields(fn)} | {"turn"}
        while i < len(rest):
            m = re.compile(r"\s*([a-z_0-9]+)=").match(rest, i)
            if not m: raise T.CallError("malformed frame field")
            name = m.group(1); i = m.end()
            if name not in names: raise T.CallError(f"unknown field {name!r}")
            if rest.startswith("none", i) and (i + 4 == len(rest) or rest[i + 4:i + 7] == " | "):
                i += 4
                if args.get("turn", "new") != "new" and T.FRAME_DROP.get(name): drop.append(T.FRAME_DROP[name])
            elif rest.startswith("keep", i) and (i + 4 == len(rest) or rest[i + 4:i + 7] == " | "): i += 4
            else:
                try: v, i = dec.raw_decode(rest, i)
                except ValueError as e: raise T.CallError(f"bad value for {name}: {e}")
                args[name] = v
            if i < len(rest):
                if rest[i:i + 3] != " | ": raise T.CallError("fields must be separated by ' | '")
                i += 3
        if fn not in T.NO_TURN: args.setdefault("turn", "new")
        if drop: args["drop"] = sorted(set(drop))
        return fn, args


class Intent(Frame):
    """The INTENT surface: `result=count | turn="new" | target="HARSH_BRAKE" | period="this month" | restrictions... | field=none`. The model says what RESULT the person wants back; this dialect derives the executor (typed routing on the result and the registry kind of the target), so the model never chooses between overlapping executor names. `registry` must be set (the planner does)."""
    name = "intent"; style = "intent"; open_text = ""; grounded = False
    registry = None

    def kind_of(self, i):
        c = self.registry.by_id.get(i) if (self.registry is not None and i) else None
        return c.kind if c is not None else None

    def inline(self, fn, args):
        result, sargs = T.encode_intent(fn, args); meta = result in ("ask", "unresolved", "chat", "refuse"); new = args.get("turn", "new") == "new"; dropped = set(args.get("drop") or []); parts = [f"result={result}"]
        for sn, st in T.surface_fields(result):
            if sn == "turn": continue
            if sn == "conditions_edit" or True:
                pass
            if sn in sargs: v = json.dumps(sargs[sn], ensure_ascii=False)
            elif new or sn == "conditions_edit": v = "none"
            else: v = "none" if T.FRAME_DROP.get(sn) in dropped else "keep"
            parts.append(f"{sn}={v}")
        if not meta: parts.insert(1, f"turn={json.dumps(args.get('turn', 'new'))}")
        return " | ".join(parts)

    def parse(self, raw):
        text = raw.strip().split("\n")[0].strip(); m = re.match(r"result=([a-z_]+)", text)
        if not m or m.group(1) not in T.INTENT_ORDER: raise T.CallError("no valid result")
        result = m.group(1); rest = text[m.end():].lstrip(" |").strip(); names = {sn for sn, _ in T.surface_fields(result)} | {"turn"}; sargs = {}; drop = []; dec = json.JSONDecoder(); i = 0
        while i < len(rest):
            mm = re.compile(r"\s*([a-z_0-9]+)=").match(rest, i)
            if not mm: raise T.CallError("malformed field")
            name = mm.group(1); i = mm.end()
            if name not in names: raise T.CallError(f"unknown field {name!r}")
            if rest.startswith("none", i) and (i + 4 == len(rest) or rest[i + 4:i + 7] == " | "):
                i += 4
                if sargs.get("turn", "new") != "new" and T.FRAME_DROP.get(name): drop.append(T.FRAME_DROP[name])
            elif rest.startswith("keep", i) and (i + 4 == len(rest) or rest[i + 4:i + 7] == " | "): i += 4
            else:
                try: v, i = dec.raw_decode(rest, i)
                except ValueError as e: raise T.CallError(f"bad value for {name}: {e}")
                sargs[name] = v
            if i < len(rest):
                if rest[i:i + 3] != " | ": raise T.CallError("fields must be separated by ' | '")
                i += 3
        fn, args = T.decode_intent(result, sargs, self.kind_of)
        if fn not in T.NO_TURN: args.setdefault("turn", "new")
        if drop: args["drop"] = sorted(set(drop))
        return fn, args


def detect(template):
    """The dialect of a chat template (by its documented tool-call markers)."""
    if "tool_call_name" in template: return XLam()                                               # xLAM-2: the template documents a JSON array of {name, arguments}
    if "<tool_call>" in template: return JsonCall()                                              # Granite: calls are JSON inside <tool_call> tags
    if "<|tool_call_start|>" in template or "List of tools:" in template: return Pythonic()       # LFM2.x: tools listed in the system turn, calls written in Pythonic form between its call markers
    raise ValueError("the chat template documents no tool-call format this plan language supports")


def by_name(name):
    return {"lfm": Pythonic, "granite": JsonCall, "json": JsonCall, "xlam": XLam, "xlam_rev": XLamReversed, "qwen": QwenCall, "frame": Frame, "intent": Intent}[name]()
