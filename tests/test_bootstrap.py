"""One start-up for every door (vgpt/bootstrap.py): `vehiclegpt.py run`, `python ui/server.py` and `app.py` harden the process, adopt the product's model from models/product.json, keep the memory budget and check the install - the same
way.  Importing the module changes nothing; the interface module cannot be started without it; the resolved model is what /api/health reports."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vgpt import bootstrap as B  # noqa: E402

PRODUCT = os.path.join(ROOT, "models", "product.json")


class Importing(unittest.TestCase):
    def test_changes_no_environment_variable(self):
        code = "import os, sys; sys.path.insert(0, %r); b = dict(os.environ); import vgpt.bootstrap; print('SAME' if dict(os.environ) == b else 'CHANGED')" % ROOT
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=ROOT); self.assertIn("SAME", p.stdout, p.stdout + p.stderr)


class Adopting(unittest.TestCase):
    def fake_root(self, cfg, adapter=True, ort=True):
        d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "models")); json.dump(cfg, open(os.path.join(d, "models", "product.json"), "w"))
        if adapter and cfg.get("adapter"): open(os.path.join(d, "models", cfg["adapter"]), "wb").write(b"adapter-bytes")
        if ort and cfg.get("ort"): os.makedirs(os.path.join(d, "models", cfg["ort"])); open(os.path.join(d, "models", cfg["ort"], "model.onnx"), "wb").write(b"onnx-bytes")
        return d

    def clean_env(self): return mock.patch.dict(os.environ, {}, clear=False)

    def test_the_manifest_sets_the_frame_the_adapter_and_the_onnx_build_and_the_choice_is_recorded_with_hashes(self):
        import hashlib
        root = self.fake_root({"adapter": "a.gguf", "frame": 2, "ort": "ortdir", "vehicle_hint": True})
        with mock.patch.dict(os.environ):
            for k in ("VGPT_FRAME", "VGPT_MEANING_ADAPTER", "VGPT_ORT_DIR", "VGPT_VEHICLE_HINT"): os.environ.pop(k, None)
            r = B.adopt_product_model(root)
            self.assertEqual(os.environ["VGPT_FRAME"], "2"); self.assertEqual(os.environ["VGPT_MEANING_ADAPTER"], os.path.join(root, "models", "a.gguf")); self.assertEqual(os.environ["VGPT_ORT_DIR"], os.path.join(root, "models", "ortdir")); self.assertEqual(os.environ["VGPT_VEHICLE_HINT"], "1")
            self.assertEqual(r["source"], "models/product.json"); self.assertEqual((r["adapter"], r["frame"], r["ort_dir"]), ("a.gguf", 2, "ortdir")); self.assertEqual(r["adapter_sha256"], hashlib.sha256(b"adapter-bytes").hexdigest())
            self.assertEqual(B.resolved()["adapter"], "a.gguf"); self.assertTrue(r["product_json"])

    def test_an_explicit_setting_of_the_person_wins(self):
        root = self.fake_root({"adapter": "a.gguf", "frame": 2})
        with mock.patch.dict(os.environ, {"VGPT_FRAME": "1"}):
            os.environ.pop("VGPT_MEANING_ADAPTER", None); r = B.adopt_product_model(root); self.assertEqual(os.environ["VGPT_FRAME"], "1"); self.assertNotIn("VGPT_MEANING_ADAPTER", os.environ); self.assertEqual(r["source"], "environment")

    def test_a_manifest_that_names_a_file_that_is_not_there_changes_nothing(self):
        root = self.fake_root({"adapter": "missing.gguf", "frame": 2}, adapter=False)
        with mock.patch.dict(os.environ):
            for k in ("VGPT_FRAME", "VGPT_MEANING_ADAPTER"): os.environ.pop(k, None)
            B.adopt_product_model(root); self.assertNotIn("VGPT_FRAME", os.environ)

    def test_the_launcher_and_the_interface_module_use_this_same_adoption(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("vehiclegpt_bs", os.path.join(ROOT, "vehiclegpt.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        root = self.fake_root({"adapter": "a.gguf", "frame": 2}); m.ROOT = root
        with mock.patch.dict(os.environ):
            os.environ.pop("VGPT_FRAME", None); os.environ.pop("VGPT_MEANING_ADAPTER", None); m._adopt_product_model(); self.assertEqual(os.environ.get("VGPT_FRAME"), "2")


class Starting(unittest.TestCase):
    def setUp(self): self._done = list(B._DONE); B._DONE.clear()
    def tearDown(self): B._DONE[:] = self._done

    def test_startup_hardens_adopts_checks_and_guards_once_per_process(self):
        calls = []
        with mock.patch("vgpt.hardening.harden", side_effect=lambda: calls.append("harden")), mock.patch.object(B, "adopt_product_model", side_effect=lambda r=None: calls.append("adopt")), \
             mock.patch.object(B, "preflight", side_effect=lambda r=None, m="0.5b": (calls.append("doctor") or True, "", [])):
            guard = mock.Mock()
            ok, rep = B.startup(ROOT, "0.5b", 1500, guard=guard); self.assertTrue(ok); self.assertEqual(calls, ["harden", "adopt", "doctor"]); guard.assert_called_once_with(1500)
            ok, rep = B.startup(ROOT, "0.5b", 1500, guard=guard); self.assertTrue(ok); self.assertEqual(len(calls), 3, "the second door of the same process does not start again"); guard.assert_called_once()

    def test_a_failed_check_stops_the_start_with_the_plain_report(self):
        with mock.patch("vgpt.hardening.harden"), mock.patch.object(B, "adopt_product_model"), mock.patch.object(B, "preflight", return_value=(False, "Python: too old", [])):
            ok, rep = B.startup(ROOT, "0.5b", 0); self.assertFalse(ok); self.assertIn("too old", rep); self.assertFalse(B._DONE)

    def test_the_memory_limit_comes_from_the_argument_then_the_environment(self):
        with mock.patch.dict(os.environ, {"VGPT_MEMORY_LIMIT_MB": "1800"}): self.assertEqual(B.memory_limit_from(None), 1800); self.assertEqual(B.memory_limit_from(1500), 1500)
        with mock.patch.dict(os.environ, {"VGPT_MEMORY_LIMIT_MB": "junk"}): self.assertEqual(B.memory_limit_from(None), 0)

    def test_the_interface_module_starts_through_the_bootstrap_and_refuses_when_it_says_no(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("uiserver_bs", os.path.join(ROOT, "ui", "server.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        with mock.patch.object(B, "startup", return_value=(False, "doctor says no")) as st, mock.patch.object(m, "make_app", side_effect=AssertionError("must not build the app")), mock.patch("builtins.print"):
            self.assertEqual(m.main(["--memory-limit-mb", "1900"]), 2); st.assert_called_once(); self.assertEqual(st.call_args[0][2], 1900)

    def test_the_terminal_chat_starts_through_the_bootstrap_too(self):
        self.assertIn("bootstrap.startup", open(os.path.join(ROOT, "app.py"), encoding="utf-8").read())


class Hashing(unittest.TestCase):
    def test_a_hash_is_remembered_across_starts_and_recomputed_when_the_file_changes(self):
        import hashlib
        home = tempfile.mkdtemp(); f = os.path.join(home, "big.bin"); open(f, "wb").write(b"abc" * 1000)
        with mock.patch.dict(os.environ, {"VEHICLEGPT_HOME": home}):
            B._HASH_CACHE.clear(); h1 = B.file_sha256(f); self.assertEqual(h1, hashlib.sha256(b"abc" * 1000).hexdigest()); self.assertTrue(os.path.exists(os.path.join(home, "hashes.json")))
            B._HASH_CACHE.clear()
            with mock.patch.object(B.hashlib, "sha256", side_effect=AssertionError("the file must not be read again")): self.assertEqual(B.file_sha256(f), h1)
            open(f, "ab").write(b"!"); os.utime(f, (os.stat(f).st_atime, os.stat(f).st_mtime + 10)); B._HASH_CACHE.clear(); self.assertNotEqual(B.file_sha256(f), h1)

    def test_the_onnx_hash_is_not_computed_at_start_up_but_in_the_background_when_asked(self):
        d = tempfile.mkdtemp(); open(os.path.join(d, "model.onnx"), "wb").write(b"onnx" * 100); B._RESOLVED["ort_sha256"] = None; del B._HASHING[:]
        with mock.patch.dict(os.environ, {"VGPT_ORT_DIR": d, "VEHICLEGPT_HOME": tempfile.mkdtemp()}):
            B.adopt_product_model(tempfile.mkdtemp()); self.assertIsNone(B.resolved()["ort_sha256"], "adopting the product model does not read the ONNX file")
            B.ort_sha256()
            for _ in range(100):
                if B.ort_sha256(): break
                import time; time.sleep(0.05)
            import hashlib; self.assertEqual(B.ort_sha256(), hashlib.sha256(b"onnx" * 100).hexdigest())
        del B._HASHING[:]; B._RESOLVED["ort_sha256"] = None


class Describing(unittest.TestCase):
    def test_a_loaded_model_is_described_by_what_it_really_is(self):
        class OrtFrameModel: version = 2; decoder = "mask"; provider = "CPUExecutionProvider"; vehicle_hint = True; vehicle = "petrol"; llm = object(); onto = type("O", (), {"version": 2})()
        d = B.describe_model(OrtFrameModel()); self.assertEqual((d["backend"], d["frame"], d["decoder"], d["ontology"]), ("onnxruntime-int8", 2, "mask", 2))
        class FrameModel: version = 2; decoder = "mask"; llm = object()
        self.assertEqual(B.describe_model(FrameModel())["backend"], "llama.cpp"); self.assertFalse(B.describe_model(None)["loaded"])

    def test_the_product_manifest_of_this_install_resolves_to_files_that_exist(self):
        if not os.path.exists(PRODUCT): self.skipTest("no models/product.json")
        j = json.load(open(PRODUCT, encoding="utf-8")); self.assertTrue(os.path.exists(os.path.join(ROOT, "models", j["adapter"])), "the adapter the manifest names")
        if j.get("ort"): self.assertTrue(os.path.exists(os.path.join(ROOT, "models", j["ort"], "model.onnx")), "the ONNX build the manifest names")


if __name__ == "__main__": unittest.main()
