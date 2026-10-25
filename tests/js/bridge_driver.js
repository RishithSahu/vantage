// Runs ui/modern/bridge.js in a sandbox with a fake "native side" and prints one JSON line of what happened.  Used by tests/test_shell.py (needs node).
const vm = require('vm'), fs = require('fs'), path = require('path');
const code = fs.readFileSync(path.join(__dirname, '..', '..', 'ui', 'modern', 'bridge.js'), 'utf8');
const sent = []; let replies = [];
class FakeWS { constructor() { this.made = true; } } class FakeXHR { constructor() { this.made = true; } } class FakeES { constructor() { this.made = true; } }
const ctx = { console, setTimeout, clearTimeout, atob: globalThis.atob, btoa: globalThis.btoa, Headers, Response, URL, JSON, Promise, Map, Date, Error, TypeError, Uint8Array, Object, String, Array,
  navigator: { sendBeacon: () => true }, addEventListener() {}, WebSocket: FakeWS, XMLHttpRequest: FakeXHR, EventSource: FakeES };
ctx.window = ctx; vm.createContext(ctx);
const b64 = o => Buffer.from(typeof o === 'string' ? o : JSON.stringify(o)).toString('base64');
ctx.__vgpt_post = text => {                                                         // the "program": answers after a moment, like the real one
  const m = JSON.parse(text); sent.push(m);
  let r = { id: m.id, s: 200, h: [['Content-Type', 'application/json']], b: b64({ ok: true, echo: m.p, method: m.m }) };
  if (m.p === '/api/login') r.h.push(['Set-Cookie', 'vgpt_s=tok123; HttpOnly; SameSite=Strict; Path=/']);
  if (m.p === '/api/logout') r.h.push(['Set-Cookie', 'vgpt_s=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0']);
  if (m.p === '/api/nothing') r = { id: m.id, s: 204, h: [], b: '' };
  if (m.p.startsWith('/api/frame')) r = { id: m.id, s: 200, h: [['Content-Type', 'image/jpeg']], b: Buffer.from([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3]).toString('base64') };
  if (m.p === '/api/bad') r = { id: m.id, s: 403, h: [['Content-Type', 'application/json']], b: b64({ ok: false, error: 'no' }) };
  setTimeout(() => ctx.__vgpt_reply(r), 3);
};
vm.runInContext(code, ctx);
(async () => {
  const out = {};
  const f = ctx.fetch;
  let r = await f('/api/health', { cache: 'no-store' }); out.get = { status: r.status, ok: r.ok, body: await r.json(), sent: sent[0] };
  r = await f('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ q: 'hi' }) }); out.post = { status: r.status, sent: sent[1] };
  await f('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }); await f('/api/role'); out.cookie_after_login = sent[3].h.Cookie;
  await f('/api/logout', { method: 'POST', body: '{}' }); await f('/api/role'); out.cookie_after_logout = sent[5].h.Cookie === undefined ? null : sent[5].h.Cookie;
  r = await f('/api/nothing'); out.no_content = { status: r.status, body_is_null: r.body === null };
  r = await f('/api/frame?1'); const buf = new Uint8Array(await r.arrayBuffer()); out.binary = { status: r.status, type: r.headers.get('content-type'), first: [...buf.slice(0, 2)], length: buf.length };
  r = await f('/api/bad'); out.refused = { status: r.status, ok: r.ok };
  out.rejected = {};
  for (const url of ['https://example.com/', 'http://127.0.0.1:8765/api/health', '//evil.example/api/x', '/other', 'api/health', '']) { try { await f(url); out.rejected[url] = 'ALLOWED'; } catch (e) { out.rejected[url] = e.name; } }
  out.no_extra_traffic_for_rejected = sent.length;
  out.classes = {}; for (const n of ['WebSocket', 'XMLHttpRequest', 'EventSource']) { try { new ctx[n](); out.classes[n] = 'ALLOWED'; } catch (e) { out.classes[n] = e.name; } }
  out.beacon = { own: ctx.navigator.sendBeacon('/api/voice/stop', '{}'), other: ctx.navigator.sendBeacon('https://example.com/x', '{}') };
  await new Promise(res => setTimeout(res, 20)); out.beacon_sent = sent.filter(m => m.p === '/api/voice/stop').length;
  out.again = ctx.__vgpt_bridge;
  console.log('RESULT ' + JSON.stringify(out));
})().catch(e => { console.log('RESULT ' + JSON.stringify({ crashed: String(e && e.stack || e) })); });
