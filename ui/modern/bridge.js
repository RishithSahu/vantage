/* The page's only way to reach the program: an in-process bridge.  There is no network in this application.
 *
 * The page asks for things the way it always did (`fetch('/api/ask', {...})`); this script answers those calls by passing the request to the program that shows the page (a native web view, a Chromium window driven over pipes,
 * an Android WebView) and turning its reply back into a `Response`.  Nothing here opens a connection: `fetch` of any address that is not one of the program's own routes is refused, and so are XMLHttpRequest, WebSocket,
 * EventSource and sendBeacon to anywhere else.  The packed page also carries a content-security policy that forbids every connection, which holds even if a script tried.
 *
 * Messages (JSON text) page -> program:  {"id": 7, "m": "POST", "p": "/api/ask", "h": {"Content-Type": "application/json", "Cookie": "vgpt_s=..."}, "b": "{\"q\":\"...\"}"}
 *                          program -> page: window.__vgpt_reply({"id": 7, "s": 200, "h": [["Content-Type", "application/json"], ...], "b": "<base64 of the body>"})
 * The program provides ONE of: window.pywebview.api.post (a native web view), window.__vgpt_post (Chromium binding), window.VantageNative.post (Android). */
(() => {
'use strict';
if (window.__vgpt_bridge) return; window.__vgpt_bridge = true;
const pending = new Map(); let seq = 0, session = '';
const OWN = /^\/(api\/|camera|favicon)/;                                                                  // the program's own routes; nothing else is an address this page may use
const offline = what => { const e = new TypeError('This application is fully offline: ' + what + ' is not available.'); e.name = 'OfflineError'; return e; };

function transport() {
  if (window.pywebview && window.pywebview.api && typeof window.pywebview.api.post === 'function') return s => window.pywebview.api.post(s);
  if (typeof window.__vgpt_post === 'function') return s => window.__vgpt_post(s);
  if (window.VantageNative && typeof window.VantageNative.post === 'function') return s => window.VantageNative.post(s);
  return null;
}
let readyP = null;
function ready() {
  if (readyP) return readyP;
  return readyP = new Promise((resolve, reject) => {
    const t0 = Date.now(), tick = () => { const t = transport(); if (t) return resolve(t); if (Date.now() - t0 > 20000) { readyP = null; return reject(new Error('The program behind this page is not answering.')); } setTimeout(tick, 25); };
    window.addEventListener('pywebviewready', tick, { once: true }); tick();
  });
}

const b64ToBytes = b => { const bin = atob(b), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };

function call(method, path, headers, body) {
  return ready().then(send => new Promise(resolve => {
    const id = ++seq, h = Object.assign({}, headers); if (session) h.Cookie = 'vgpt_s=' + session;
    pending.set(id, resolve); send(JSON.stringify({ id, m: method, p: path, h, b: body == null ? '' : String(body) }));
  }));
}
window.__vgpt_reply = r => {
  const done = pending.get(r.id); if (!done) return; pending.delete(r.id);
  for (const [k, v] of (r.h || [])) if (k.toLowerCase() === 'set-cookie') { const m = /^vgpt_s=([^;]*)/.exec(v); if (m) session = /Max-Age=0/i.test(v) ? '' : m[1]; }      // the role's session lives in this page's memory only
  done(r);
};

window.fetch = async (input, init = {}) => {
  const url = typeof input === 'string' ? input : (input && input.url) || '';
  if (!OWN.test(url)) throw offline('reaching ' + (url.slice(0, 60) || 'an address'));
  const headers = {}; if (init.headers) new Headers(init.headers).forEach((v, k) => { headers[k] = v; });
  const r = await call((init.method || 'GET').toUpperCase(), url, headers, init.body);
  const hd = new Headers(); for (const [k, v] of (r.h || [])) if (k.toLowerCase() !== 'set-cookie') hd.append(k, v);
  return new Response([204, 205, 304].includes(r.s) ? null : b64ToBytes(r.b || ''), { status: r.s, headers: hd });
};
if (navigator.sendBeacon) navigator.sendBeacon = (url, data) => { if (!OWN.test(String(url))) return false; call('POST', String(url), { 'Content-Type': 'application/json' }, typeof data === 'string' ? data : ''); return true; };
for (const name of ['XMLHttpRequest', 'WebSocket', 'EventSource', 'RTCPeerConnection', 'WebTransport']) if (name in window) Object.defineProperty(window, name, { configurable: false, get() { throw offline(name); } });
})();
