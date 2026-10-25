"""Release provenance (tools_release_manifest.py, vgpt/releasecheck.py): a manifest records the source, the models, the vocabulary, the test data, the runtime, the version control state and the installer; a tree is verified against it; the
doctor refuses an installed copy whose files differ from the release it names; /api/health reports it."""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import tools_release_manifest as T  # noqa: E402
from vgpt import releasecheck as RC  # noqa: E402


def tree():
    d = tempfile.mkdtemp()
    def w(rel, data=b"x"):
        p = os.path.join(d, rel); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data)
    w("vgpt/a.py", b"A = 1\n"); w("vgpt/meaning/ontology_data.json", b"{}"); w("ui/app.html", b"<html>"); w("ui/research.py", b"study"); w("vehiclegpt.py", b"main"); w("app.py", b"chat")
    w("models/product.json", json.dumps({"adapter": "ad.gguf", "frame": 2, "ort": "ort"}).encode()); w("models/ad.gguf", b"adapter"); w("models/qwen2.5-0.5b-instruct-q4_k_m.gguf", b"base"); w("models/ort/model.onnx", b"onnx"); w("models/ort/draft.json", b"{}")
    return d


class Manifest(unittest.TestCase):
    def test_it_records_source_models_vocabulary_runtime_vcs_and_installer(self):
        d = tree(); inst = os.path.join(d, "Setup.exe"); open(inst, "wb").write(b"setup"); m = T.make(d, installer=inst, version="9.9.9")
        self.assertEqual(m["version"], "9.9.9"); self.assertEqual(len(m["source"]["digest"]), 64); self.assertIn("vgpt/a.py", m["source"]["files"]); self.assertNotIn("ui/research.py", m["source"]["files"], "the study is not part of a release")
        for rel in ("models/product.json", "models/ad.gguf", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "models/ort/model.onnx"): self.assertEqual(len(m["models"][rel]["sha256"]), 64, rel)
        self.assertIn("vgpt/meaning/ontology_data.json", m["vocabulary"]["files"]); self.assertIn("python", m["runtime"]); self.assertIn("platform", m["runtime"]); self.assertEqual(m["installer"]["file"], "Setup.exe"); self.assertEqual(len(m["installer"]["sha256"]), 64)
        self.assertEqual(m["vcs"]["kind"], "none"); self.assertFalse(m["vcs"]["releasable_from_tag"], "a tree that is not a git worktree is not releasable from a tag - the manifest says so")
        for k in ("test_data",): self.assertIn(k, m)

    def test_the_same_tree_gives_the_same_digest_and_a_changed_file_gives_another(self):
        d = tree(); a = T.make(d)["source"]["digest"]; self.assertEqual(T.make(d)["source"]["digest"], a)
        open(os.path.join(d, "vgpt", "a.py"), "ab").write(b"B = 2\n"); self.assertNotEqual(T.make(d)["source"]["digest"], a)

    def test_verify_finds_a_changed_a_missing_and_an_extra_file_and_a_changed_model(self):
        d = tree(); m = T.make(d); self.assertEqual(RC.verify(d, m, models=True), [])
        open(os.path.join(d, "vgpt", "a.py"), "ab").write(b"#\n"); os.remove(os.path.join(d, "ui", "app.html")); open(os.path.join(d, "vgpt", "extra.py"), "w").write("1\n"); open(os.path.join(d, "models", "ad.gguf"), "ab").write(b"!")
        bad = " | ".join(RC.verify(d, m, models=True)); self.assertIn("source file changed: vgpt/a.py", bad); self.assertIn("missing source file ui/app.html", bad); self.assertIn("not in the release: vgpt/extra.py", bad); self.assertIn("model file changed: models/ad.gguf", bad)

    def test_the_cheap_check_notices_a_model_of_another_size_without_hashing_it(self):
        d = tree(); m = T.make(d); open(os.path.join(d, "models", "ort", "model.onnx"), "ab").write(b"more"); self.assertTrue(any("another size" in x for x in RC.verify(d, m, models=False)))

    def test_info_says_whether_the_model_the_process_chose_is_the_one_the_release_names(self):
        d = tree(); m = T.make(d); json.dump(m, open(os.path.join(d, "RELEASE.json"), "w")); sha = m["models"]["models/ad.gguf"]["sha256"]
        i = RC.info(d, {"adapter": "ad.gguf", "adapter_sha256": sha}); self.assertTrue(i["present"]); self.assertTrue(i["models_match"]); self.assertEqual(i["version"], m["version"])
        self.assertFalse(RC.info(d, {"adapter": "ad.gguf", "adapter_sha256": "0" * 64})["models_match"]); self.assertEqual(RC.info(tempfile.mkdtemp()), {"present": False})

    def test_the_doctor_refuses_an_installed_copy_that_differs_from_its_release(self):
        from vgpt import doctor
        d = tree(); json.dump(T.make(d), open(os.path.join(d, "RELEASE.json"), "w"))
        chk = lambda: [c for c in doctor.run_checks(d, None, "none") if c["name"] == "Release manifest"][0]
        self.assertEqual(chk()["status"], doctor.OK); open(os.path.join(d, "vgpt", "a.py"), "ab").write(b"# edited\n"); c = chk(); self.assertEqual(c["status"], doctor.FAIL); self.assertIn("a.py", c["detail"]); self.assertIn("reinstall", c["fix"])
        os.remove(os.path.join(d, "RELEASE.json")); self.assertEqual(chk()["status"], doctor.OK); self.assertIn("development tree", chk()["detail"])

    def test_this_tree_verifies_against_a_manifest_made_from_it(self):
        out = os.path.join(tempfile.mkdtemp(), "RELEASE.json"); self.assertEqual(T.main(["make", "--root", ROOT, "--out", out]), 0); self.assertEqual(T.main(["verify", "--root", ROOT, "--manifest", out]), 0)


if __name__ == "__main__": unittest.main()
