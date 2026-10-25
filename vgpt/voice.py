"""Voice input: the microphone is heard on this computer only.

The page captures the microphone, turns it into 16 kHz mono 16-bit audio and sends it in short pieces to this module (loopback only).  A speech recogniser that runs on this computer writes the
words down; this module listens for the wake word and gives the page the question to put in the text box:

    "Vantage, how is my battery?"        -> the question goes into the text box (live while it is being said)
    "Vantage"  ... "how is my battery?"  -> "Vantage" alone makes it listen to the next sentence
    "Vantage stop"                       -> listening stops

What is not a command is thrown away the moment it is recognised: nothing the people in the car say to each other is kept, sent to the page or written to a log.  Audio is never written to disk.
Nothing here uses an online service (the browser's own speech recognition sends audio to a company's server and is deliberately not used).
The microphone is read by THIS program (`NativeMic`, PortAudio through the `sounddevice` package), not by a browser: no web view, Chrome or Edge permission is involved, and the page only shows the words that come back.
A page that has to capture the audio itself (no `sounddevice`) still can: it sends the pieces to `Service.audio`.

The recogniser is a separate, replaceable piece (`Recognizer`).  The one supported for the car is Vosk, a small offline recogniser (about 40 MB model, runs on two Raspberry Pi cores):
  pip install vosk      and      models/asr/vosk-model-small-en-us-0.15/   (or VGPT_VOSK_MODEL=<folder>)
Without them `status()` says exactly what is missing and the microphone button explains it; nothing is downloaded by the application.

Interim rule (debt, see docs/TRAINING_QUEUE.md): the wake word and the stop command are recognised from the written words (a fuzzy match on the first word).  A trained keyword spotter would replace this."""
import base64
import collections
import difflib
import os
import re
import threading
import time

import numpy as np

RATE = 16000
WAKE = "vantage"
ARM_S = 8.0                                   # after "Vantage" alone, the next sentence is the question
FIRST_S = 12.0                                # pressing the microphone button is itself a request: the first sentence after it is the question, with or without the wake word
IDLE_S = 20.0                                 # no audio for this long: the session ends (the page was closed)
MAX_CHUNK = 48000                             # bytes of audio accepted in one piece (1.5 s)


def words(text):
    return re.sub(r"[^a-z0-9' ]+", " ", str(text).lower()).split()


NOT_WAKE = {"advantage", "advantages", "vantages"}      # real words that sound like the wake word and mean something else ...
QSTART = {"how", "what", "when", "where", "why", "who", "which", "is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "show", "tell", "give", "list", "any", "has", "have", "am", "whats", "stop"}    # ... unless a question follows: a small recogniser often writes "Vantage, how is ..." as "advantage how is ..."


def is_wake(w):
    """A recogniser writes 'vantage' as vantage, van tage, vantage's ...: the first word is compared by its spelling, not exactly."""
    return bool(w) and w not in NOT_WAKE and (w == WAKE or difflib.SequenceMatcher(None, w, WAKE).ratio() >= 0.8)


def wake_len(ws):
    """How many written words the wake word took (1, or 2 for 'van tage'); 0 when the sentence does not begin with it."""
    if not ws: return 0
    if ws[0] in NOT_WAKE: return 1 if len(ws) >= 2 and ws[1] in QSTART else 0
    if is_wake(ws[0]): return 1
    return 2 if len(ws) >= 2 and is_wake(ws[0] + ws[1]) else 0


def is_stop(ws):
    return bool(ws) and len(ws) <= 3 and ws[0] in ("stop", "cancel", "quiet") and all(x in ("stop", "listening", "it", "please", "cancel", "quiet", "now") for x in ws)


def drop_words(text, n):
    for _ in range(n): text = re.sub(r"^\W*[\w']+\W*", "", str(text), count=1)
    return text


def parse(text, armed=False):
    """-> (kind, text): 'stop' | 'question' (text is the question, as written) | 'wake' | None (not for us: ignored)."""
    ws = words(text)
    if not ws: return None, ""
    if len(ws) >= 2 and ws[0] in ("hey", "ok", "okay", "hi") and wake_len(ws[1:]): ws = ws[1:]; text = drop_words(text, 1)
    n = wake_len(ws)
    if n:
        rest = ws[n:]
        if not rest: return "wake", ""
        if is_stop(rest): return "stop", ""
        return "question", drop_words(text, n).strip(" ,.;:-")[:600]
    if armed:
        if is_stop(ws): return "stop", ""
        return "question", str(text).strip()[:600]
    return None, ""


class Recognizer:
    """Streaming speech recogniser: `accept(pcm16)` -> (final_text or None, partial_text)."""
    name = "none"

    def accept(self, pcm):
        raise NotImplementedError

    def reset(self): pass
    def close(self): pass


class Vosk(Recognizer):
    name = "vosk"

    def __init__(self, folder):
        import vosk
        vosk.SetLogLevel(-1)
        self._vosk = vosk; self.model = vosk.Model(folder); self.rec = None; self.reset()

    def reset(self):
        self.rec = self._vosk.KaldiRecognizer(self.model, RATE)

    def accept(self, pcm):
        import json
        if self.rec.AcceptWaveform(pcm): return (json.loads(self.rec.Result()).get("text") or "").strip(), ""
        return None, (json.loads(self.rec.PartialResult()).get("partial") or "").strip()


class Scripted(Recognizer):
    """A stand-in for tests and demonstrations: each 'utterance' is the text the recogniser will report once that much audio has been heard."""
    name = "scripted"

    def __init__(self, script):
        self.script = collections.deque(script); self.heard = 0; self.cur = None

    def accept(self, pcm):
        self.heard += len(pcm)
        if self.cur is None and self.script: self.cur = self.script.popleft()
        if self.cur is None: return None, ""
        need, text = self.cur
        if self.heard >= need: self.cur = None; self.heard = 0; return text, ""
        return None, text.split(" ", 1)[0] if self.heard * 2 > need else ""


def model_folder(root):
    cands = [os.environ.get("VGPT_VOSK_MODEL") or "", os.path.join(root, "models", "asr", "vosk-model-small-en-us-0.15")]
    cands += [os.path.join(root, "models", "asr", d) for d in (os.listdir(os.path.join(root, "models", "asr")) if os.path.isdir(os.path.join(root, "models", "asr")) else []) if d.startswith("vosk-model")]
    return next((c for c in cands if c and os.path.isdir(c)), None)


def availability(root):
    """-> (ok, reason).  Says exactly what is missing."""
    try: import vosk  # noqa: F401
    except ImportError: return False, "The offline speech recogniser is not installed (pip install vosk)."
    f = model_folder(root)
    if not f: return False, "No offline speech model found. Put the small English Vosk model in models/asr/ (vosk-model-small-en-us-0.15)."
    return True, f


def native_available():
    """(ok, reason): can this program read the microphone itself?"""
    try: import sounddevice as sd
    except Exception as e: return False, f"the audio library is missing ({type(e).__name__})"
    try:
        d = sd.query_devices(kind="input")
        return (True, str(d.get("name", "microphone"))) if d else (False, "no microphone was found")
    except Exception as e: return False, f"no microphone was found ({type(e).__name__})"


class NativeMic:
    """Reads the default microphone as 16 kHz mono 16-bit audio and hands every block to `sink(bytes)`.  `stream_factory` is replaceable for tests."""
    def __init__(self, sink, stream_factory=None, block=1600):
        self.sink = sink; self.block = block; self.stream = None; self.error = ""; self.factory = stream_factory

    def start(self):
        def cb(indata, frames, t, status):
            try: self.sink(bytes(indata))
            except Exception as e: self.error = f"{type(e).__name__}: {e}"
        if self.factory: self.stream = self.factory(cb)
        else:
            import sounddevice as sd
            self.stream = sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=self.block, callback=cb)
        self.stream.start(); return self

    def stop(self):
        s, self.stream = self.stream, None
        if s is not None:
            try: s.stop(); s.close()
            except Exception: pass


class Session:
    """One listening session: audio in, events out.  Recognition runs on a worker thread so that receiving audio never waits for it."""
    def __init__(self, rec, close_rec=True, first_arm=0.0):
        self.rec = rec; self.close_rec = close_rec; self.q = collections.deque(maxlen=150); self.ev = []; self.lock = threading.Lock(); self.cv = threading.Condition(self.lock)
        self.armed_until = (time.monotonic() + first_arm) if first_arm else 0.0; self.level = 0.0; self.last_audio = time.monotonic(); self.done = False; self.dropped = 0; self.hold_until = 0.0; self.draft = ""
        if first_arm: self.ev.append({"type": "armed"})
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def feed(self, pcm):
        """Audio in (bytes of little-endian int16).  Returns nothing: events are collected with `take()`."""
        if len(pcm) % 2: pcm = pcm[:-1]
        a = np.frombuffer(pcm, dtype="<i2")
        with self.cv:
            self.last_audio = time.monotonic()
            self.level = float(min(1.0, np.sqrt(np.mean(a.astype(np.float32) ** 2)) / 6000.0)) if a.size else 0.0
            if time.monotonic() < self.hold_until: return                  # the car is speaking an answer: its own voice is not listened to
            if len(self.q) == self.q.maxlen: self.dropped += 1
            self.q.append(bytes(pcm)); self.cv.notify()

    def hold(self, seconds):
        with self.cv: self.hold_until = max(self.hold_until, time.monotonic() + seconds); self.q.clear()

    def take(self):
        with self.cv:
            out, self.ev = self.ev, []
            return out, self.level

    def _emit(self, kind, **k):
        with self.cv: self.ev.append(dict(type=kind, **k))

    def _run(self):
        while True:
            with self.cv:
                while not self.q and not self.done: self.cv.wait(1.0)
                if self.done: return
                chunk = self.q.popleft()
            try: final, partial = self.rec.accept(chunk)
            except Exception as e:
                self._emit("error", message=f"{type(e).__name__}: {e}"); self.done = True; return
            now = time.monotonic(); armed = now < self.armed_until
            if final is not None:
                self.draft = ""
                kind, text = parse(final, armed)
                if kind == "question": self.armed_until = 0.0; self._emit("question", text=text)
                elif kind == "wake": self.armed_until = now + ARM_S; self._emit("armed")
                elif kind == "stop": self._emit("stop"); self.done = True; return
                elif words(final): self._emit("ignored", words=len(words(final)))                 # speech that was not for the assistant: only THAT it was heard is reported, never what was said
            elif partial:
                kind, text = parse(partial, armed)
                if kind == "stop" and words(partial)[:1] and is_wake(words(partial)[0]): self._emit("stop"); self.done = True; return
                if kind == "question" and text and text != self.draft: self.draft = text; self._emit("draft", text=text)
                elif kind == "wake" and not armed: self.armed_until = now + ARM_S; self._emit("armed")

    def alive(self):
        return not self.done and time.monotonic() - self.last_audio < IDLE_S

    def close(self):
        with self.cv: self.done = True; self.cv.notify_all()
        if self.t.is_alive() and threading.current_thread() is not self.t: self.t.join(timeout=3.0)
        if self.close_rec:
            try: self.rec.close()
            except Exception: pass


class Service:
    """The one microphone of the application: start / audio / stop.  `factory` makes the recogniser (replaceable for tests)."""
    def __init__(self, root, factory=None):
        self.root = root; self.factory = factory; self.s = None; self.lock = threading.Lock(); self.rec = None; self.mic = None; self.mic_factory = None

    def status(self):
        nat = self.mic_factory is not None or native_available()[0]
        if self.factory: return {"available": True, "backend": "custom", "reason": "", "listening": self.listening(), "native": nat}
        ok, why = availability(self.root)
        return {"available": ok, "backend": "vosk" if ok else None, "reason": "" if ok else why, "listening": self.listening(), "native": bool(ok and nat), "microphone": native_available()[1] if nat else ""}

    def listening(self):
        return self.s is not None and self.s.alive()

    def start(self, native=False):
        """`native=True`: this program reads the microphone itself; otherwise the page sends the audio."""
        with self.lock:
            if self.s is not None: self.stop()
            if self.factory: rec = self.factory()
            else:
                ok, why = availability(self.root)
                if not ok: return {"ok": False, "reason": why}
                try:
                    if self.rec is None: self.rec = Vosk(why)                # the model is loaded once and kept for the next session
                    else: self.rec.reset()
                    rec = self.rec
                except Exception as e: return {"ok": False, "reason": f"The speech model could not be loaded: {type(e).__name__}: {e}"}
            self.s = Session(rec, close_rec=bool(self.factory), first_arm=FIRST_S)
            if native:
                if self.mic_factory is None:
                    ok, why = native_available()
                    if not ok: self.s.close(); self.s = None; return {"ok": False, "reason": "The microphone could not be opened: " + why}
                try: self.mic = NativeMic(self.s.feed, self.mic_factory).start()
                except Exception as e: self.s.close(); self.s = None; return {"ok": False, "reason": f"The microphone could not be opened: {type(e).__name__}: {e}"}
            return {"ok": True, "backend": rec.name, "native": bool(native)}

    def audio(self, b64):
        s = self.s
        if s is None: return {"ok": False, "ended": True, "events": [], "level": 0.0}
        try: pcm = base64.b64decode(b64 or "", validate=False)
        except Exception: return {"ok": False, "error": "bad audio", "events": [], "level": 0.0}
        if len(pcm) > MAX_CHUNK: pcm = pcm[:MAX_CHUNK]
        if not s.done and pcm: s.feed(pcm)
        ev, level = s.take(); ended = not s.alive()
        if ended: self.stop()                                                   # 'Vantage stop' was heard, or the recogniser failed: the events above are the last ones
        return {"ok": True, "events": ev, "level": round(level, 3), "ended": ended}

    def hold(self, seconds=3.0):
        if self.s is not None: self.s.hold(seconds)
        return {"ok": True}

    def arm(self):
        """The person pressed the microphone button while it was listening: the next sentence is the question, no wake word needed."""
        s = self.s
        if s is None or not s.alive(): return {"ok": False}
        with s.cv: s.armed_until = time.monotonic() + FIRST_S; s.ev.append({"type": "armed"})
        return {"ok": True}

    def stop(self):
        m, self.mic = self.mic, None
        if m is not None: m.stop()
        s, self.s = self.s, None
        if s is not None: s.close()
        return {"ok": True}
