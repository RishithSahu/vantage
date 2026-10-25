"""One version: vgpt/version.py is the source of truth; the installer script and the top of the changelog say the same, so `VantageCli version`, /api/health, the installer's file name and the release manifest cannot disagree."""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vgpt.version import VERSION


class One(unittest.TestCase):
    def test_the_installer_script_and_the_changelog_say_the_version_of_the_code(self):
        iss = open(os.path.join(ROOT, "deploy", "windows", "Vantage.iss"), encoding="utf-8").read(); self.assertEqual(re.search(r'#define AppVersion "([^"]+)"', iss).group(1), VERSION)
        top = re.search(r"^## (\d+\.\d+\.\d+)", open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8").read(), re.M).group(1); self.assertEqual(top, VERSION)

    def test_the_command_line_prints_it(self):
        import subprocess
        p = subprocess.run([sys.executable, os.path.join(ROOT, "vehiclegpt.py"), "version"], capture_output=True, text=True, timeout=120, cwd=ROOT); self.assertEqual(p.stdout.strip().splitlines()[-1], VERSION)


if __name__ == "__main__": unittest.main()
