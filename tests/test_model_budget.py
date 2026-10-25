import time
import unittest
from unittest import mock

from ui import server
from vgpt import brain


class _TinySlm:
    def __init__(self, path, **kw): self.closed = False; self.load_s = 0.0
    def warmup(self): pass
    def parse(self, text): return "NONE", {"metric": None, "event": None}, {"first": False}
    def close(self): self.closed = True


class ModelBudget(unittest.TestCase):
    def test_cli_exposes_only_the_half_billion_parameter_model(self):
        parser = server.build_parser()
        self.assertEqual(parser.parse_args([]).model, "0.5b")
        with self.assertRaises(SystemExit): parser.parse_args(["--model", "1.5b"])
        with self.assertRaises(SystemExit): parser.parse_args(["--planner", "granite"])

    def test_lazy_model_releases_itself_while_idle(self):
        with mock.patch.object(brain, "Slm", _TinySlm):
            lazy = brain.LazySlm("unused.gguf", idle_s=0.03)
            lazy.parse("hello")
            self.assertTrue(lazy.loaded)
            limit = time.monotonic() + 0.5
            while lazy.loaded and time.monotonic() < limit: time.sleep(0.01)
            self.assertFalse(lazy.loaded)


if __name__ == "__main__": unittest.main()
