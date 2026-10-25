"""Voice input: the wake word, the stop command, privacy of what is not for the assistant, and the endpoints.  The speech recogniser is replaced by a scripted one (the real one needs the offline model)."""
import base64
import json
import os
import secrets
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DB = os.path.join(ROOT, "data", "cs5_petrol_ice.db")

from vgpt import voice  # noqa: E402

voice.FIRST_S = 0.0                                   # the tests below are about the wake word; FirstSentence at the end sets the real value
CHUNK = bytes(8000)                                   # a quarter of a second of 16 kHz 16-bit audio


class Parse(unittest.TestCase):
    def test_a_question_after_the_wake_word_is_the_question(self):
        self.assertEqual(voice.parse("Vantage, how is my battery?"), ("question", "how is my battery?"))
        self.assertEqual(voice.parse("hey vantage where am I"), ("question", "where am I"))
        self.assertEqual(voice.parse("van tage what is my range"), ("question", "what is my range"))

    def test_the_wake_word_alone_makes_it_listen_to_the_next_sentence(self):
        self.assertEqual(voice.parse("vantage"), ("wake", ""))
        self.assertEqual(voice.parse("how far can I go", armed=True), ("question", "how far can I go"))

    def test_vantage_stop_stops_and_a_sentence_that_only_contains_stop_is_not_a_command(self):
        for t in ("Vantage stop", "vantage, stop.", "vantage stop listening", "VANTAGE CANCEL"): self.assertEqual(voice.parse(t)[0], "stop", t)
        self.assertEqual(voice.parse("stop at the next junction"), (None, ""))
        self.assertEqual(voice.parse("vantage stop at the next junction")[0], "question")

    def test_a_small_recogniser_that_writes_advantage_for_vantage_is_still_understood_when_a_question_follows(self):
        self.assertEqual(voice.parse("advantage how is my battery"), ("question", "how is my battery")); self.assertEqual(voice.parse("hey advantage what is my range")[0], "question")
        for t in ("the advantage of this car is the range", "advantage of electric", "advantages"): self.assertEqual(voice.parse(t), (None, ""), t)

    def test_talk_that_is_not_for_the_assistant_is_ignored(self):
        for t in ("we should turn left here", "the advantage of this car is the range", "", "   "): self.assertEqual(voice.parse(t), (None, ""), t)


def run(script, chunks, gap=0.02):
    """-> every event the page would have received while `chunks` quarter-seconds of audio were sent."""
    svc = voice.Service(ROOT, factory=lambda: voice.Scripted(script)); assert svc.start()["ok"]
    out = []
    for _ in range(chunks):
        r = svc.audio(base64.b64encode(CHUNK).decode()); out += r["events"]; time.sleep(gap)
        if r["ended"]: break
    time.sleep(0.2); r = svc.audio(base64.b64encode(CHUNK).decode()); out += r["events"]; svc.stop(); return out


class Session(unittest.TestCase):
    def test_a_spoken_question_arrives_as_text_and_what_else_is_said_does_not(self):
        ev = run([(16000, "we should turn left here"), (16000, "Vantage how is my battery")], 12)
        qs = [e for e in ev if e["type"] == "question"]; self.assertEqual([e["text"] for e in qs], ["how is my battery"])
        self.assertNotIn("turn left", json.dumps(ev), "talk that is not for the assistant never reaches the page")

    def test_vantage_then_the_question(self):
        ev = run([(12000, "vantage"), (16000, "what is my range")], 12)
        self.assertIn("armed", [e["type"] for e in ev]); self.assertEqual([e["text"] for e in ev if e["type"] == "question"], ["what is my range"])

    def test_vantage_stop_ends_the_session(self):
        svc = voice.Service(ROOT, factory=lambda: voice.Scripted([(8000, "vantage stop")])); svc.start(); ended = False; kinds = []
        for _ in range(20):
            r = svc.audio(base64.b64encode(CHUNK).decode()); kinds += [e["type"] for e in r["events"]]; time.sleep(0.03)
            if r["ended"]: ended = True; break
        self.assertTrue(ended); self.assertIn("stop", kinds); self.assertFalse(svc.listening()); self.assertTrue(svc.audio("")["ended"], "after the stop nothing more is accepted")

    def test_the_car_does_not_listen_to_its_own_answer(self):
        svc = voice.Service(ROOT, factory=lambda: voice.Scripted([(8000, "vantage how is my battery")])); svc.start(); svc.hold(2.0); out = []
        for _ in range(6): out += svc.audio(base64.b64encode(CHUNK).decode())["events"]; time.sleep(0.02)
        self.assertEqual([e for e in out if e["type"] in ("question", "draft")], []); svc.stop()

    def test_a_failing_recogniser_ends_the_session_with_a_message(self):
        class Bad(voice.Recognizer):
            def accept(self, pcm): raise RuntimeError("model missing")
        svc = voice.Service(ROOT, factory=Bad); svc.start(); out = []
        for _ in range(8):
            r = svc.audio(base64.b64encode(CHUNK).decode()); out += r["events"]; time.sleep(0.03)
            if r["ended"]: break
        self.assertIn("error", [e["type"] for e in out])

    def test_without_the_offline_model_the_service_says_exactly_what_is_missing(self):
        old = os.environ.get("VGPT_VOSK_MODEL"); os.environ["VGPT_VOSK_MODEL"] = os.path.join(tempfile.gettempdir(), "no_model_here")
        try:
            svc = voice.Service(tempfile.mkdtemp()); st = svc.status()
            if st["available"]: self.skipTest("a speech model is installed here")
            self.assertTrue(st["reason"]); r = svc.start(); self.assertFalse(r["ok"]); self.assertEqual(r["reason"], st["reason"])
        finally:
            if old is None: os.environ.pop("VGPT_VOSK_MODEL", None)
            else: os.environ["VGPT_VOSK_MODEL"] = old


def post(dispatch, path, body=None):
    r = dispatch("POST", path, {"Content-Type": "application/json"}, json.dumps(body or {})); return r.status, json.loads(r.body or b"{}")


@unittest.skipUnless(os.path.exists(DB), "the Case Study 5 petrol file is not present")
class Endpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._env = dict(os.environ)                                             # starting a server sets process-wide variables (VGPT_NO_QA_CLF ...): give them back for the tests that follow
        os.environ["VEHICLEGPT_HOME"] = tempfile.mkdtemp()
        import vehiclegpt
        cls.m, cls.app, d = vehiclegpt.start_app("0.5b", DB); cls.dispatch = staticmethod(d)

    @classmethod
    def tearDownClass(cls):
        cls.app.voice.stop()
        os.environ.clear(); os.environ.update(cls._env)

    def test_a_question_spoken_through_the_endpoints_reaches_the_page_and_the_log_holds_no_words(self):
        self.app.voice.factory = lambda: voice.Scripted([(16000, "Vantage where am I")])
        st, r = post(self.dispatch, "/api/voice/start"); self.assertEqual((st, r["ok"]), (200, True)); got = []
        for _ in range(12):
            st, r = post(self.dispatch, "/api/voice/audio", {"pcm": base64.b64encode(CHUNK).decode()}); got += r["events"]; time.sleep(0.03)
        time.sleep(0.2); st, r = post(self.dispatch, "/api/voice/audio", {"pcm": base64.b64encode(CHUNK).decode()}); got += r["events"]
        self.assertEqual([e["text"] for e in got if e["type"] == "question"], ["where am I"])
        self.assertEqual(post(self.dispatch, "/api/voice/stop")[0], 200)
        trail = json.dumps(self.app.audit.tail(50)); self.assertIn("voice_start", trail); self.assertNotIn("where am I", trail, "the audit log records that voice was used, never what was said")

    def test_status_reports_availability(self):
        r = self.dispatch("GET", "/api/voice/status"); d = json.loads(r.body)
        self.assertEqual(r.status, 200); self.assertIn("available", d)


def listen(wav_name, seconds_after=6.0):
    """-> the events the page would receive when `wav_name` (a recorded sentence + 1.5 s of silence) is spoken to the REAL offline recogniser, sent in quarter-second pieces."""
    import wave
    w = wave.open(os.path.join(ROOT, "tests", "fixtures", wav_name), "rb"); pcm = w.readframes(w.getnframes()); w.close()
    svc = voice.Service(ROOT); r = svc.start(); assert r["ok"], r
    out = []
    try:
        for i in range(0, len(pcm), 8000):
            out += svc.audio(base64.b64encode(pcm[i:i + 8000]).decode())["events"]; time.sleep(0.03)
        end = time.time() + seconds_after
        while time.time() < end and not any(e["type"] in ("question", "stop") for e in out): out += svc.audio(base64.b64encode(bytes(8000)).decode())["events"]; time.sleep(0.1)
    finally: svc.stop()
    return out


@unittest.skipUnless(voice.availability(ROOT)[0], "the offline speech recogniser (pip install vosk + models/asr) is not installed")
class RealRecogniser(unittest.TestCase):
    """The recorded sentences were spoken by this computer's own Windows voice (tests/fixtures); the recogniser is the real one."""
    def test_a_spoken_question_after_the_wake_word_reaches_the_text_box(self):
        q = [e for e in listen("vantage_how_is_my_battery.wav") if e["type"] == "question"]
        self.assertEqual(len(q), 1); self.assertIn("battery", q[0]["text"]); self.assertNotIn("vantage", q[0]["text"].lower())

    def test_vantage_stop_ends_listening(self):
        self.assertIn("stop", [e["type"] for e in listen("vantage_stop.wav")])

    def test_talk_that_is_not_for_the_car_is_reported_only_as_heard_never_as_words(self):
        ev = listen("talk_not_for_the_car.wav", seconds_after=2.0)
        self.assertTrue(all(e["type"] == "ignored" and set(e) == {"type", "words"} for e in ev), ev)


class FirstSentence(unittest.TestCase):
    """Pressing the microphone button is itself a request: the first sentence after it is the question, with or without the wake word; after that the wake word is needed again, and speech that is not for the assistant is
    reported only as having been heard (how many words), never what was said."""

    def setUp(self): self.old = voice.FIRST_S; voice.FIRST_S = 12.0
    def tearDown(self): voice.FIRST_S = self.old

    def test_the_first_sentence_is_the_question_without_the_wake_word(self):
        ev = run([(16000, "how is my battery")], 10); self.assertEqual([e["text"] for e in ev if e["type"] == "question"], ["how is my battery"]); self.assertEqual(ev[0]["type"], "armed")

    def test_after_the_first_question_the_wake_word_is_needed_again(self):
        ev = run([(16000, "how is my battery"), (16000, "we should turn left here"), (16000, "vantage what is my range")], 16)
        self.assertEqual([e["text"] for e in ev if e["type"] == "question"], ["how is my battery", "what is my range"]); self.assertNotIn("turn left", json.dumps(ev))
        ig = [e for e in ev if e["type"] == "ignored"]; self.assertEqual(len(ig), 1); self.assertEqual(ig[0], {"type": "ignored", "words": 5})

    def test_pressing_the_button_again_makes_the_next_sentence_the_question(self):
        voice.FIRST_S = 0.0
        svc = voice.Service(ROOT, factory=lambda: voice.Scripted([(16000, "we should turn left here"), (16000, "how far can I go")])); svc.start(); out = []; voice.FIRST_S = 12.0
        for i in range(3): out += svc.audio(base64.b64encode(CHUNK).decode())["events"]; time.sleep(0.15)                # the first sentence (2 chunks) is heard and ignored; the second has barely begun
        self.assertTrue(svc.arm()["ok"])
        for i in range(10): out += svc.audio(base64.b64encode(CHUNK).decode())["events"]; time.sleep(0.02)
        time.sleep(0.3); out += svc.audio(base64.b64encode(CHUNK).decode())["events"]; svc.stop()
        self.assertIn("how far can I go", [e.get("text") for e in out if e["type"] == "question"]); self.assertNotIn("turn left", json.dumps(out))
        self.assertFalse(svc.arm()["ok"], "nothing to arm once it has stopped")

    def test_the_window_closes(self):
        old = voice.FIRST_S; voice.FIRST_S = 0.05
        try:
            svc = voice.Service(ROOT, factory=lambda: voice.Scripted([(16000, "how is my battery")])); svc.start(); time.sleep(0.3); out = []
            for _ in range(8): out += svc.audio(base64.b64encode(CHUNK).decode())["events"]; time.sleep(0.02)
            time.sleep(0.2); out += svc.audio(base64.b64encode(CHUNK).decode())["events"]; svc.stop()
        finally: voice.FIRST_S = old
        self.assertEqual([e for e in out if e["type"] == "question"], [])



class NativeMicrophone(unittest.TestCase):
    """The program reads the microphone itself (no browser): blocks from the audio device go straight into the session, the page only polls for what was heard."""

    class FakeStream:
        def __init__(self, cb, pcm): self.cb, self.pcm, self.on, self.t = cb, pcm, False, None
        def start(self):
            import threading
            self.on = True; self.t = threading.Thread(target=self._pump, daemon=True); self.t.start()
        def _pump(self):
            for i in range(0, len(self.pcm), 3200):
                if not self.on: return
                self.cb(self.pcm[i:i + 3200], 1600, None, None); time.sleep(0.01)
        def stop(self): self.on = False
        def close(self): self.on = False

    def service(self, script, pcm=bytes(160000)):
        svc = voice.Service(ROOT, factory=lambda: voice.Scripted(script)); svc.mic_factory = lambda cb: self.FakeStream(cb, pcm); return svc

    def poll(self, svc, seconds=3.0):
        out = []; end = time.monotonic() + seconds
        while time.monotonic() < end:
            r = svc.audio(""); out += r["events"]
            if r["ended"]: break
            time.sleep(0.05)
        return out

    def test_the_status_says_the_program_can_listen_by_itself(self):
        svc = self.service([]); self.assertTrue(svc.status()["native"])

    def test_a_question_heard_by_the_program_reaches_the_page_through_polling(self):
        svc = self.service([(16000, "we should turn left here"), (16000, "Vantage how is my battery")]); self.assertEqual(svc.start(native=True), {"ok": True, "backend": "scripted", "native": True})
        ev = self.poll(svc); svc.stop()
        self.assertEqual([e["text"] for e in ev if e["type"] == "question"], ["how is my battery"]); self.assertNotIn("turn left", json.dumps(ev))

    def test_stopping_releases_the_microphone(self):
        svc = self.service([]); svc.start(native=True); mic = svc.mic; self.assertIsNotNone(mic.stream); svc.stop(); self.assertIsNone(mic.stream); self.assertFalse(svc.listening())

    def test_a_missing_microphone_is_said_not_hidden(self):
        svc = voice.Service(ROOT, factory=lambda: voice.Scripted([])); svc.mic_factory = None
        import unittest.mock as mock
        with mock.patch.object(voice, "native_available", return_value=(False, "no microphone was found")):
            r = svc.start(native=True)
        self.assertFalse(r["ok"]); self.assertIn("microphone", r["reason"]); self.assertFalse(svc.listening())

    @unittest.skipUnless(voice.availability(ROOT)[0] and sys.platform == "win32", "needs the offline model and the Windows speech synthesiser")
    def test_real_speech_is_recognised_offline(self):
        import subprocess, wave
        wav = os.path.join(tempfile.mkdtemp(), "q.wav")
        ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
              f"$s.SetOutputToWaveFile('{wav}',$f); $s.Speak('Vantage, how is my battery'); $s.Dispose()")
        if subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, timeout=60).returncode or not os.path.exists(wav): self.skipTest("no speech synthesiser")
        w = wave.open(wav); pcm = w.readframes(w.getnframes()) + bytes(32000); w.close()
        svc = voice.Service(ROOT); svc.mic_factory = lambda cb: self.FakeStream(cb, pcm); self.assertTrue(svc.start(native=True)["ok"])
        ev = self.poll(svc, 15); svc.stop()
        qs = [e["text"] for e in ev if e["type"] == "question"]; self.assertTrue(qs and "battery" in qs[0], ev)


def any_db():
    for d in (DB, os.path.join(ROOT, "data", "world_aster.db"), os.path.join(ROOT, "data", "world4", "aster_ev", "aster_ev_4yr.db")):
        if os.path.exists(d): return d


@unittest.skipUnless(any_db(), "no vehicle history is present")
class EndToEnd(unittest.TestCase):
    """The whole chain exactly as the application runs it: the page asks the dispatcher to start NATIVE listening, polls /api/voice/audio with no audio of its own, the program's own microphone reader feeds the REAL offline
    recogniser, and the words come back as events.  The device is replaced by a stream that plays a recorded sentence; everything else is the product."""

    @classmethod
    def setUpClass(cls):
        cls._env = dict(os.environ); os.environ["VEHICLEGPT_HOME"] = tempfile.mkdtemp()
        import vehiclegpt
        cls.m, cls.app, d = vehiclegpt.start_app("0.5b", any_db()); cls.dispatch = staticmethod(d)

    @classmethod
    def tearDownClass(cls):
        cls.app.voice.stop(); os.environ.clear(); os.environ.update(cls._env)

    class RealTimeStream(NativeMicrophone.FakeStream):
        def _pump(self):                                                                                  # a microphone delivers 100 ms of audio every 100 ms, not the whole sentence at once
            for i in range(0, len(self.pcm), 3200):
                if not self.on: return
                self.cb(self.pcm[i:i + 3200], 1600, None, None); time.sleep(0.1)

    def speak(self, wav, seconds=12.0, first_arm=0.0):
        import wave
        w = wave.open(os.path.join(ROOT, "tests", "fixtures", wav), "rb"); pcm = w.readframes(w.getnframes()) + bytes(48000); w.close()
        old = voice.FIRST_S; voice.FIRST_S = first_arm; self.app.voice.mic_factory = lambda cb: self.RealTimeStream(cb, pcm)
        self.assertTrue(json.loads(self.dispatch("GET", "/api/voice/status").body)["native"])
        st, r = post(self.dispatch, "/api/voice/start", {"native": True}); self.assertEqual((st, r.get("ok")), (200, True), r); got = []; t0 = time.time()
        try:
            while time.time() - t0 < seconds:
                st, r = post(self.dispatch, "/api/voice/audio", {"pcm": ""}); self.assertEqual(st, 200); got += r["events"]                 # the page polls with no audio of its own
                if r.get("ended") or any(e["type"] in ("question", "stop") for e in r["events"]): break
                time.sleep(0.15)
        finally:
            post(self.dispatch, "/api/voice/stop"); voice.FIRST_S = old
        return got

    def test_vantage_how_is_my_battery_comes_back_as_a_question(self):
        ev = self.speak("vantage_how_is_my_battery.wav"); qs = [e["text"] for e in ev if e["type"] == "question"]
        self.assertEqual(len(qs), 1, ev); self.assertIn("battery", qs[0])

    def test_vantage_stop_ends_it(self):
        ev = self.speak("vantage_stop.wav"); self.assertIn("stop", [e["type"] for e in ev], ev)

    def test_the_sentence_after_the_button_works_without_the_wake_word(self):
        ev = self.speak("vantage_how_is_my_battery.wav", first_arm=12.0); self.assertTrue([e for e in ev if e["type"] == "question"], ev)


if __name__ == "__main__":
    unittest.main()
