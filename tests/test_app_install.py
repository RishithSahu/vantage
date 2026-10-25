"""Vantage is an APP: `run` starts the app window (fully offline, no browser, no server), `install-app` puts an icon on the desktop and in the menu, and the icon file is a real icon."""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vgpt import appinstall  # noqa: E402


def launcher():
    spec = importlib.util.spec_from_file_location("vehiclegpt_cli", os.path.join(ROOT, "vehiclegpt.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Icon(unittest.TestCase):
    def test_the_icon_is_a_png_and_a_windows_icon_that_wraps_the_same_png(self):
        d = tempfile.mkdtemp(); png, ico = os.path.join(d, "a.png"), os.path.join(d, "a.ico"); appinstall.make_icon(png, ico)
        data = open(png, "rb").read(); self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n"); raw = open(ico, "rb").read()
        self.assertEqual(raw[:6], b"\x00\x00\x01\x00\x01\x00"); size, offset = int.from_bytes(raw[14:18], "little"), int.from_bytes(raw[18:22], "little"); self.assertEqual(offset, 22); self.assertEqual(raw[22:22 + size], data)
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED); self.assertEqual(img.shape, (256, 256, 4)); self.assertEqual(int(img[0, 0, 3]), 0); self.assertEqual(int(img[128, 20, 3]), 255)       # transparent corner, solid body
        shutil.rmtree(d, ignore_errors=True)

    def test_the_icon_shipped_with_the_application_exists(self):
        for f in ("vantage.png", "vantage.ico"): self.assertTrue(os.path.exists(os.path.join(ROOT, "ui", "modern", f)), f)


class WhatTheIconStarts(unittest.TestCase):
    def test_the_command_of_the_icon(self):
        prog, args = appinstall._command(False); self.assertTrue(args[0].endswith("vehiclegpt.py")); self.assertEqual(args[1:], ["run"])
        prog, args = appinstall._command(True); self.assertEqual(args[1:], ["run", "--kiosk"])
        if os.name == "nt": self.assertTrue(os.path.basename(prog).lower() in ("pythonw.exe", "python.exe"))

    def test_the_linux_menu_entry(self):
        e = appinstall.desktop_entry("Vantage", "Ask your car", "/usr/bin/python3", ["/opt/my app/vehiclegpt.py", "run"], "/opt/my app/vantage.png")
        self.assertIn("Type=Application", e); self.assertIn("Terminal=false", e); self.assertIn('Exec=/usr/bin/python3 "/opt/my app/vehiclegpt.py" run', e); self.assertIn("Icon=/opt/my app/vantage.png", e); self.assertIn("Name=Vantage", e)


@unittest.skipUnless(os.name == "nt", "the shortcut files are Windows ones; the Linux entry is tested above")
class WindowsShortcuts(unittest.TestCase):
    def test_install_makes_two_shortcuts_that_open_the_app_and_remove_takes_them_away(self):
        home = tempfile.mkdtemp(); said = []
        paths = appinstall.install("Vantage Test", home=home, say=said.append); self.assertEqual(len(paths), 2); self.assertTrue(all(os.path.exists(p) and p.endswith(".lnk") for p in paths))
        ps = "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK); Write-Output $s.TargetPath; Write-Output $s.Arguments; Write-Output $s.IconLocation; Write-Output $s.WindowStyle"
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=dict(os.environ, LNK=paths[0]), capture_output=True, text=True, timeout=60).stdout.split("\n"); target, args, icon, style = [x.strip() for x in out[:4]]
        self.assertTrue(os.path.basename(target).lower() in ("pythonw.exe", "python.exe")); self.assertIn("vehiclegpt.py", args); self.assertTrue(args.rstrip().endswith("run")); self.assertNotIn("--vantage", args); self.assertTrue(icon.lower().startswith(os.path.join(ROOT, "ui", "modern", "vantage.ico").lower()))
        gone = appinstall.install("Vantage Test", home=home, remove=True, say=said.append); self.assertEqual(sorted(gone), sorted(paths)); self.assertFalse(any(os.path.exists(p) for p in paths))
        shutil.rmtree(home, ignore_errors=True)


class MacBundle(unittest.TestCase):
    """The layout of a macOS app bundle, written and checked here (a Mac was not available to run it)."""
    def test_the_bundle_has_a_valid_plist_an_icns_and_an_executable_launcher(self):
        import plistlib
        d = tempfile.mkdtemp(); b = appinstall.macos_bundle(d, "Vantage Test", "/usr/bin/python3", ["/opt/my app/vehiclegpt.py", "open", "--vantage", "10.0.0.5:8765"])
        self.assertTrue(b.endswith("Vantage Test.app")); plist = plistlib.load(open(os.path.join(b, "Contents", "Info.plist"), "rb")); self.assertEqual(plist["CFBundleExecutable"], "Vantage Test"); self.assertEqual(plist["CFBundlePackageType"], "APPL"); self.assertEqual(plist["CFBundleIconFile"], "Vantage Test")
        launcher = os.path.join(b, "Contents", "MacOS", "Vantage Test"); text = open(launcher, encoding="utf-8", newline="").read(); self.assertTrue(text.startswith("#!/bin/sh\n")); self.assertIn("exec /usr/bin/python3 '/opt/my app/vehiclegpt.py' open --vantage 10.0.0.5:8765", text); self.assertNotIn("\r", text)
        if os.name != "nt": self.assertTrue(os.access(launcher, os.X_OK))
        icns = open(os.path.join(b, "Contents", "Resources", "Vantage Test.icns"), "rb").read(); self.assertEqual(icns[:4], b"icns"); self.assertEqual(int.from_bytes(icns[4:8], "big"), len(icns)); self.assertEqual(icns[8:12], b"ic08"); self.assertEqual(icns[16:24], b"\x89PNG\r\n\x1a\n")
        shutil.rmtree(d, ignore_errors=True)

    def test_on_a_mac_the_icon_goes_to_the_applications_folder_of_the_user(self):
        with mock.patch.object(appinstall.sys, "platform", "darwin"):
            kind, paths = appinstall.locations("Vantage", home="/Users/x"); self.assertEqual(kind, "macos"); self.assertEqual(paths, [os.path.join("/Users/x", "Applications", "Vantage.app")])


class RunIsTheOfflineApp(unittest.TestCase):
    def test_run_hands_the_persons_choices_to_the_window(self):
        m = launcher(); seen = []; fake = SimpleNamespace(main=lambda argv: (seen.append(argv), 0)[1])
        a = SimpleNamespace(model="0.5b", camera="clip", shell="auto", db="gen_courier.db", kiosk=True, memory_limit_mb=0)
        with mock.patch.object(m, "_server_module", return_value=fake), mock.patch.object(m.doctor, "run_checks", return_value=[]): rc = m.cmd_run(a)
        self.assertEqual(rc, 0); self.assertEqual(seen, [["--model", "0.5b", "--camera", "clip", "--shell", "auto", "--db", os.path.join("data", "gen_courier.db"), "--kiosk"]])

    def test_a_window_that_cannot_be_opened_is_told_to_the_person_who_clicked_the_icon(self):
        m = launcher(); fake = SimpleNamespace(main=lambda argv: 2); a = SimpleNamespace(model="0.5b", camera="clip", shell="auto", db=None, kiosk=False, memory_limit_mb=0)
        with mock.patch.object(m, "_server_module", return_value=fake), mock.patch.object(m.doctor, "run_checks", return_value=[]), mock.patch.object(m, "_tell") as told: rc = m.cmd_run(a)
        self.assertEqual(rc, 2); self.assertTrue(told.called); self.assertIn("doctor", told.call_args.args[0])

    def test_a_failed_check_stops_the_start_with_the_report(self):
        m = launcher(); a = SimpleNamespace(model="0.5b", camera="clip", shell="auto", db=None, kiosk=False, memory_limit_mb=0)
        with mock.patch.object(m.doctor, "run_checks", return_value=[{"name": "Window", "status": "fail", "detail": "none", "fix": "install one"}]), mock.patch.object(m, "_server_module") as srv, mock.patch.object(m, "_tell") as told:
            self.assertEqual(m.cmd_run(a), 2)
        self.assertFalse(srv.called); self.assertIn("Window", told.call_args.args[0])

    def test_the_commands_that_needed_a_network_are_gone(self):
        m = launcher()
        for cmd in ("vantage", "open", "link-key"):                                                          # "car" is back, but as a local simulator that writes to a pipe (tests/test_pipe.py), not a network command
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit): m.main([cmd])
        for flag in ("--port", "--no-browser", "--bind"):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit): m.main(["run", flag, "1"])

    def test_the_launcher_imports_no_network_facility_and_names_no_address(self):
        import ast
        tree = ast.parse(open(os.path.join(ROOT, "vehiclegpt.py"), encoding="utf-8").read()); mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import): mods.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module: mods.add(n.module)
        self.assertEqual(sorted(m for m in mods if m.split(".")[0] in ("socket", "ssl", "http", "urllib", "webbrowser", "socketserver", "ftplib", "smtplib", "xmlrpc")), [])
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        for s in strings: self.assertNotRegex(s, r"https?://|127\.0\.0\.1|localhost"); 


if __name__ == "__main__":
    unittest.main()
