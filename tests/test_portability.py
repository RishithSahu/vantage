"""The product must not depend on one operating system: the scan reads every module and reports a call, module or path that exists on one system only and is not guarded."""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt import portability  # noqa: E402


def scan_text(text):
    d = tempfile.mkdtemp(); p = os.path.join(d, "m.py")
    with open(p, "w", encoding="utf-8") as fh: fh.write(text)
    return portability.scan_file(p)


class TheProductIsPortable(unittest.TestCase):
    def test_no_module_uses_a_one_system_feature_unguarded(self):
        found = portability.scan(); self.assertEqual(found, {}, "\n".join(f"{m}:{l} {k} {w}" for m, hits in found.items() for l, k, w in hits))


class TheScanCanSeeAProblem(unittest.TestCase):
    """A scan that finds nothing proves nothing unless it is also shown to find something."""
    def test_it_flags_a_windows_only_call(self): self.assertEqual([k for _l, k, _w in scan_text("import os\nos.startfile('x')\n")], ["windows"])
    def test_it_flags_a_posix_only_module_import(self): self.assertTrue(any(w == "fcntl" for _l, _k, w in scan_text("import fcntl\n")))
    def test_it_flags_a_linux_kernel_only_facility_that_qnx_and_android_lack(self): self.assertEqual([k for _l, k, _w in scan_text("import select\nep = select.epoll()\n")], ["linux"])
    def test_it_flags_a_hard_coded_system_path(self): self.assertEqual([k for _l, k, _w in scan_text("x = open('/proc/meminfo')\n")], ["path"])
    def test_it_flags_a_hard_coded_windows_path(self): self.assertEqual([k for _l, k, _w in scan_text("x = 'C:\\\\Windows\\\\x'\n")], ["path"])

    def test_a_platform_test_around_the_use_makes_it_acceptable(self):
        self.assertEqual(scan_text("import os\nif os.name == 'nt':\n    os.startfile('x')\n"), [])
        self.assertEqual(scan_text("import os\nx = os.getuid() if hasattr(os, 'getuid') else 0\n"), [])
        self.assertEqual(scan_text("import sys\nif sys.platform.startswith('linux'):\n    x = open('/proc/meminfo')\n"), [])

    def test_an_import_error_handler_makes_it_acceptable(self):
        self.assertEqual(scan_text("try:\n    import fcntl\nexcept ImportError:\n    fcntl = None\n"), [])
        self.assertEqual(scan_text("try:\n    x = open('/proc/meminfo').read()\nexcept OSError:\n    x = ''\n"), [])

    def test_a_variable_that_is_only_named_like_a_module_is_not_the_module(self):
        self.assertEqual(scan_text("resource = 'a'\ngrp = [1]\nprint(resource, grp)\n"), [])


class TheOneSystemModulesAreNamedAndReasoned(unittest.TestCase):
    def test_every_exception_exists_and_says_why(self):
        for rel, why in portability.ONE_SYSTEM.items(): self.assertTrue(os.path.exists(os.path.join(ROOT, rel)), rel); self.assertGreater(len(why), 30, rel)

    def test_the_product_modules_do_not_import_the_windows_only_test_harness(self):
        for base in ("vgpt", "ui"):
            for d, _dirs, files in os.walk(os.path.join(ROOT, base)):
                for f in files:
                    if f.endswith(".py") and f != "governor.py":
                        with open(os.path.join(d, f), encoding="utf-8") as fh: text = fh.read()
                        self.assertNotIn("import governor", text, f); self.assertNotIn("from vgpt.governor", text, f)


if __name__ == "__main__":
    unittest.main()
