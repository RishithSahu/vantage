"""The packed page: the whole interface as ONE self-contained document.

A page that is fetched by address needs a server, and this product has none.  So the interface (`ui/modern`: index.html, app.css, app.js, voice.js, the icon sprite, the fonts, the audio worklet) is packed into a single
HTML text that a window can show without loading anything: styles, scripts and the worklet's source are inline, fonts are data URIs, the icon sprite is inline, and the bridge (`bridge.js`) comes first.  A content-security
policy forbids every connection (`connect-src 'none'`), every frame and every object, whatever a script tries.

`build()` makes the text, `load()` returns it (from `ui/modern/dist/vantage.html` when that is newer than the sources, else it builds it)."""
import base64
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODERN = os.path.join(ROOT, "ui", "modern")
DIST = os.path.join(MODERN, "dist", "vantage.html")
SOURCES = ("index.html", "app.css", "app.js", "voice.js", "voice-worklet.js", "bridge.js", "icons.svg")
CSP = ("default-src 'none'; script-src 'unsafe-inline' blob:; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; media-src blob:; worker-src blob:; "
       "connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'")


def _text(modern, name):
    with open(os.path.join(modern, name), encoding="utf-8", newline="") as f: return f.read().replace("\r\n", "\n")


def _inline(js): return "<script>" + js.replace("</script", "<\\/script") + "</script>"


def build(modern=MODERN):
    """-> the packed page as text.  Raises ValueError if anything in it would still be loaded by address."""
    html = _text(modern, "index.html"); css = _text(modern, "app.css")

    def font(m):
        with open(os.path.join(modern, m.group(1)), "rb") as f: return "url(data:font/woff2;base64," + base64.b64encode(f.read()).decode("ascii") + ")"
    css = re.sub(r"url\(/modern/([A-Za-z0-9_.-]+\.woff2)\)", font, css)
    sprite = _text(modern, "icons.svg"); sprite = re.sub(r"<\?xml[^>]*\?>\s*", "", sprite).strip(); sprite = sprite.replace("<svg", '<svg style="display:none" aria-hidden="true"', 1)
    app = _text(modern, "app.js").replace("/modern/icons.svg#", "#"); voice = _text(modern, "voice.js").replace("'/modern/voice-worklet.js'", "''"); worklet = _text(modern, "voice-worklet.js"); bridge = _text(modern, "bridge.js")
    html = html.replace("/modern/icons.svg#", "#")
    for old, new in (('<link rel="stylesheet" href="/modern/app.css">', "<style>" + css + "</style>"), ('<script src="/modern/app.js" defer></script>', _inline(app)),
                     ('<script src="/modern/voice.js" defer></script>', _inline("window.__VGPT_WORKLET = " + json.dumps(worklet).replace("</", "<\\/") + ";") + _inline(voice))):
        if html.count(old) != 1: raise ValueError("the page no longer contains: " + old)
        html = html.replace(old, new)
    m = re.search(r'<meta charset="[^"]*">', html)
    if not m: raise ValueError("the page has no charset declaration")
    html = html[:m.end()] + '\n<meta http-equiv="Content-Security-Policy" content="' + CSP + '">\n' + _inline(bridge) + html[m.end():]
    html = html.replace("<body>", "<body>\n" + sprite, 1)
    left = re.findall(r"""(?:src|href|url\()\s*=?\s*["']?(/[A-Za-z][^"')\s>]*)""", html)
    if left: raise ValueError("these would still be loaded by address: " + ", ".join(sorted(set(left))[:5]))
    return html


def load():
    """The packed page: the prebuilt file when it is at least as new as every source, else built now (and kept, when the folder can be written)."""
    try:
        newest = max(os.path.getmtime(os.path.join(MODERN, n)) for n in SOURCES)
        if os.path.exists(DIST) and os.path.getmtime(DIST) >= newest:
            with open(DIST, encoding="utf-8", newline="") as f: return f.read()
    except OSError: pass
    html = build()
    try:
        os.makedirs(os.path.dirname(DIST), exist_ok=True)
        with open(DIST, "w", encoding="utf-8", newline="") as f: f.write(html)
    except OSError: pass
    return html
