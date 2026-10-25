"""A window through the operating system's own embedded web view (the `pywebview` package: WebView2 on Windows, WKWebView on macOS, WebKitGTK or Qt WebEngine on Linux).

The page is handed over as text (`html=`), so no file server is started; the page's calls arrive as `Api.post(text)` (a JavaScript-to-Python call inside the same process) and the replies go back with `evaluate_js`.  No
socket, no port, no address.  The package is an optional dependency: without it `vgpt.shell.run` says so and tries the next shell."""


def available():
    import importlib.util
    try: return importlib.util.find_spec("webview") is not None
    except (ImportError, ValueError): return False


def run(bridge, html, title="Vantage", kiosk=False):
    import webview

    class _Api:
        """What the page may call.  Only `post`: the web view exposes every public attribute of this object to the page, so the bridge is kept private."""
        def __init__(self, b): self._b = b
        def post(self, text): self._b.post(text)

    window = webview.create_window(title, html=html, js_api=_Api(bridge), fullscreen=bool(kiosk), width=1280, height=780, min_size=(800, 480), text_select=True)
    bridge.attach(lambda text: window.evaluate_js("window.__vgpt_reply(" + text + ")"))
    webview.start(debug=False, private_mode=True)
    return 0
