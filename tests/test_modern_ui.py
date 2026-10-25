"""The modern interface (ui/modern): answered at `/` by the in-process dispatcher (no server), the earlier one kept at `/classic`, static files answered safely, and every API path its script calls exists."""
import importlib.util
import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from tests.stub_planner import StubPlanner  # noqa: E402

MODERN = os.path.join(ROOT, "ui", "modern")
HAVE = os.path.exists(os.path.join(ROOT, "data", "ref", "camera_clip.hevc")) and os.path.exists(os.path.join(ROOT, "data", "gen_courier.db"))


class StaticFiles(unittest.TestCase):
    """No server needed: what the page loads and what its script depends on."""

    def test_the_page_loads_only_its_own_files(self):
        html = open(os.path.join(MODERN, "index.html"), encoding="utf-8").read()
        self.assertIn("<title>VehicleGPT</title>", html); self.assertIn('href="/modern/app.css"', html); self.assertIn('src="/modern/app.js"', html)
        self.assertEqual(re.findall(r'(?:src|href)="(https?:)?//', html), [], "the page asks for nothing from another address")
        for f in ("app.css", "app.js"): self.assertTrue(os.path.getsize(os.path.join(MODERN, f)) > 1000, f)

    def test_the_fonts_are_bundled_and_the_stylesheet_asks_only_for_them(self):
        """The look rests on two open-licence faces that ship in the folder (nothing is fetched); every url() in the stylesheet must be one of those files."""
        css = open(os.path.join(MODERN, "app.css"), encoding="utf-8").read(); urls = re.findall(r"url\(([^)]*)\)", css)
        self.assertEqual(len(urls), 5); self.assertEqual(re.findall(r"https?:", css), [], "no external requests")
        for u in urls:
            self.assertTrue(u.startswith("/modern/") and u.endswith(".woff2"), u); path = os.path.join(MODERN, u[len("/modern/"):]); self.assertTrue(os.path.getsize(path) > 4000, u)
        self.assertEqual({"Barlow Semi Condensed", "Oxanium", "Michroma"}, set(re.findall(r'font-family:"([^"]+)";font-weight', css)), "three faces, three jobs: reading, HUD labels, digital numerals")
        lic = open(os.path.join(MODERN, "LICENSES.txt"), encoding="utf-8").read(); self.assertIn("SIL OPEN FONT LICENSE", lic.upper()); self.assertIn("Phosphor", lic); self.assertIn("MIT", lic)

    def test_every_icon_the_page_uses_is_in_the_bundled_sprite(self):
        """Icons are Phosphor symbols from icons.svg, never hand-drawn paths: each name used by the markup or the script must exist as a symbol."""
        sprite = open(os.path.join(MODERN, "icons.svg"), encoding="utf-8").read(); have = set(re.findall(r'<symbol id="i-([a-z0-9-]+)"', sprite)); self.assertGreater(len(have), 40)
        html = open(os.path.join(MODERN, "index.html"), encoding="utf-8").read(); js = open(os.path.join(MODERN, "app.js"), encoding="utf-8").read()
        used = set(re.findall(r"icons\.svg#i-([a-z0-9-]+)", html)) | set(re.findall(r"\bic\('([a-z0-9-]+)'", js)) | set(re.findall(r"\bicon: '([a-z0-9-]+)'", js)) | {"sun", "moon"}
        used |= set(re.findall(r"(?:now|at|stat|extreme|trend|compare|events|context|why|spec|define|howto|advice|health|estimate|social|offtopic|unsupported|control|unsafe|ambiguous): \['([a-z0-9-]+)'", js))
        self.assertGreater(len(used), 25, used); self.assertEqual(sorted(used - have), [], "icons used but missing from icons.svg")
        bar = html.split("<script src=")[0].split("<body>")[1].split("<main")[0]; bar = bar[bar.index("</svg>") + 6:] if bar.lstrip().startswith("<svg width=\"0\"") else bar; self.assertNotIn("<path", bar, "no inline hand-drawn icon paths in the header markup (the hidden defs hold the car and its gradients only)")

    def test_the_copy_and_styles_keep_the_design_rules(self):
        """Mechanical guards from the design review: no em or en dash anywhere on the page, no stripe backgrounds, no transition on everything, no magic z-index, no removed focus ring."""
        for f in ("index.html", "app.css", "app.js"):
            t = open(os.path.join(MODERN, f), encoding="utf-8").read(); self.assertNotIn("—", t, f); self.assertNotIn("–", t, f); self.assertNotIn("&mdash;", t, f)
        css = open(os.path.join(MODERN, "app.css"), encoding="utf-8").read()
        for bad in ("repeating-linear-gradient", "transition:all", "transition: all", "background-clip:text", "scale(0)", "outline:none", "outline:0"): self.assertNotIn(bad, css, bad)
        self.assertEqual(re.findall(r"z-index:\s*(\d+)", css), [], "layers come from the --z- tokens")
        self.assertIn(":focus-visible", css); self.assertIn("prefers-reduced-motion", css); self.assertIn("hover:hover", css)

    def test_the_script_has_no_dynamic_code_and_escapes_what_it_inserts(self):
        js = open(os.path.join(MODERN, "app.js"), encoding="utf-8").read()
        for bad in ("eval(", "new Function(", "document.write(", "http://", "https://"): self.assertNotIn(bad, js, bad)
        self.assertIn("const esc =", js)

    def test_every_api_path_the_script_calls_exists_on_the_server(self):
        js = open(os.path.join(MODERN, "app.js"), encoding="utf-8").read(); server = open(os.path.join(ROOT, "ui", "server.py"), encoding="utf-8").read()
        used = set(re.findall(r"""api\(\s*['"`](/api/[a-z/]+)""", js)) | {"/api/frame"}                       # the camera picture is fetched one frame at a time through the bridge
        self.assertTrue({"/api/state", "/api/ask", "/api/track", "/api/units", "/api/feedback", "/api/datasets", "/api/dataset"} <= used, used)
        self.assertEqual(sorted(p for p in used if p in ("/api/playback", "/api/fault", "/api/camera") or p.startswith("/api/live")), [], "the car drives by itself: no scenario, speed, fault or live-control calls from the page")
        for path in used: self.assertIn(f'"{path}"', server, path)


@unittest.skipUnless(HAVE, "needs the camera clip and data/gen_courier.db")
class Served(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["VEHICLEGPT_HOME"] = tempfile.mkdtemp()
        spec = importlib.util.spec_from_file_location("uiserver_modern", os.path.join(ROOT, "ui", "server.py")); cls.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)
        app = cls.m.App(SimpleNamespace(db="data/gen_courier.db", model="none", camera="clip", webcam_index=0, port=0), planner_factory=lambda conn: StubPlanner(conn, {}))
        cls.dispatch = staticmethod(cls.m.make_dispatcher(app))

    def get(self, path, headers=None):
        r = self.dispatch("GET", path, headers or {}); return r.status, dict(r.headers), r.body

    def test_root_is_the_modern_interface_and_the_earlier_ones_are_kept(self):
        st, h, body = self.get("/"); self.assertEqual(st, 200); self.assertIn(b"/modern/app.js", body); self.assertIn("text/html", h["Content-Type"])
        for path, marker in (("/classic", b"<title>VehicleGPT</title>"), ("/app", b"<title>VehicleGPT</title>"), ("/headunit", b"Head Unit")):
            st, _, body = self.get(path); self.assertEqual(st, 200, path); self.assertIn(marker, body, path); self.assertNotIn(b"/modern/app.js", body, path)

    def test_static_files_have_the_right_type_and_revalidate_by_etag(self):
        st, h, body = self.get("/modern/app.js"); self.assertEqual(st, 200); self.assertIn("javascript", h["Content-Type"]); self.assertEqual(h["X-Content-Type-Options"], "nosniff")
        st, h2, _ = self.get("/modern/app.css"); self.assertEqual(st, 200); self.assertIn("text/css", h2["Content-Type"])
        st, h3, font = self.get("/modern/oxanium.woff2"); self.assertEqual(st, 200); self.assertEqual(h3["Content-Type"], "font/woff2"); self.assertEqual(font[:4], b"wOF2")
        st, h4, icons = self.get("/modern/icons.svg"); self.assertEqual(st, 200); self.assertEqual(h4["Content-Type"], "image/svg+xml"); self.assertIn(b"<symbol", icons)
        st, _, empty = self.get("/modern/app.js", {"If-None-Match": h["ETag"]}); self.assertEqual((st, empty), (304, b""))

    def test_the_static_route_cannot_leave_its_folder_or_serve_other_types(self):
        for path in ("/modern/FONTS-LICENSE.txt", "/modern/..%2Fserver.py", "/modern/../server.py", "/modern/%2e%2e/server.py", "/modern/sub/app.js", "/modern/app.py", "/modern/missing.js", "/modern/"):
            st, _, _ = self.get(path); self.assertEqual(st, 404, path)

    def test_the_apis_the_page_needs_answer(self):
        for path in ("/api/state", "/api/datasets", "/api/track?s=1800", "/api/health"):
            st, _, body = self.get(path); self.assertEqual(st, 200, path); self.assertTrue(body.startswith((b"{", b"[")), path)


if __name__ == "__main__":
    unittest.main()
