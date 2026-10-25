/* Voice input.  The microphone is captured here and sent, as small pieces of audio, to this computer's own assistant (loopback only), where an offline recogniser writes the words down.
   "Vantage, how is my battery?" puts the question in the text box as it is being said; "Vantage" alone listens to the next sentence; "Vantage stop" (or the Stop button) ends listening.
   Nothing is sent to an online service and nothing else that is said is kept.  The browser's own speech recognition is deliberately not used: it sends the audio to a company's server.
   Uses from app.js (window.VGPT): api, toast, showPage, page, inp, grow, ask, locked, prefs. */
(() => {
  'use strict';
  const V = window.VGPT; if (!V) return;
  const $ = s => document.querySelector(s);
  const mic = $('#bMic'), pill = $('#vpill'), txt = $('#vtxt'), stopB = $('#bVStop'), bars = pill ? [...pill.querySelectorAll('.vlv i')] : [];
  if (!mic || !pill) return;
  let poll = null, ctx = null, stream = null, node = null, src = null, on = false, starting = false, inflight = 0, muted = false, status = null, lastDraft = '';

  const say = (msg, s) => { txt.textContent = msg; pill.dataset.s = s || 'listening'; pill.hidden = false; };
  const level = v => bars.forEach((b, i) => { const h = 4 + Math.round(18 * Math.min(1, v * 1.6) * (0.55 + 0.45 * Math.sin(i * 1.7 + performance.now() / 140))); b.style.height = h + 'px'; b.style.opacity = String(0.35 + 0.65 * Math.min(1, v * 2)); });
  const b64 = buf => { const u = new Uint8Array(buf); let s = ''; for (let i = 0; i < u.length; i += 4096) s += String.fromCharCode.apply(null, u.subarray(i, i + 4096)); return btoa(s); };

  async function check() {
    try { status = await (await fetch('/api/voice/status')).json(); } catch { status = null; }
    const ok = !!(status && status.available && (status.native || (navigator.mediaDevices && navigator.mediaDevices.getUserMedia)));
    mic.dataset.na = String(!ok); mic.title = ok ? 'Listen for “Vantage”' : 'Voice input is not set up';
    return ok;
  }

  function fill(text, final) {
    if (V.page() !== 'ask') V.showPage('ask');
    V.inp.value = text; V.grow();
    if (!final) { say('“' + text + '”', 'hearing'); return; }
    if (V.locked() || V.prefs.voiceAsk !== false) { say('Asking: “' + text + '”', 'heard'); V.ask(text); }                    // moving: nobody can press Enter, so what was said is asked
    else { say('Heard: “' + text + '”  -  press Enter to ask', 'heard'); V.inp.focus(); }
  }

  function onReply(r) {
    if (!r || !on) return;
    level(r.level || 0); watchSilence(r.level || 0);
    for (const e of r.events || []) {
      if (e.type === 'draft') { lastDraft = e.text; fill(e.text, false); }
      else if (e.type === 'question') { lastDraft = ''; fill(e.text, true); }
      else if (e.type === 'armed') say('Yes? Say your question…', 'armed');
      else if (e.type === 'ignored') say('I heard talking, but no question for me. Say “Vantage, …” and then your question.', 'listening');
      else if (e.type === 'stop') { stop('Stopped listening'); return; }
      else if (e.type === 'error') { stop('Voice input stopped: ' + e.message); return; }
    }
    if (r.ended) stop('Listening ended');
    else if (!(r.events || []).length && pill.dataset.s !== 'hearing' && pill.dataset.s !== 'heard' && pill.dataset.s !== 'armed') say('Listening - say “Vantage …”', 'listening');
  }

  /* A quiet room and a blocked microphone look the same (many microphones gate their noise to zero), so nothing is blamed: after a few seconds without any sound above the background the pill says so and what to check. */
  let loudest = 0, since = 0, hinted = false;
  function watchSilence(lv) {
    const t = performance.now(); if (!since) since = t; loudest = Math.max(loudest, lv);
    if (loudest >= 0.02) { hinted = false; since = t; loudest = 0; return; }
    if (t - since > 7000 && !hinted) { hinted = true; say('Nothing heard yet - speak closer, or check the microphone (Windows Settings > Privacy > Microphone)', 'listening'); }
  }

  function send(buf) {
    if (!on || muted || inflight >= 3) return;                         // while the car speaks an answer it does not listen to itself; a slow recogniser drops audio rather than lagging
    inflight++; V.api('/api/voice/audio', { pcm: b64(buf) }).then(onReply).catch(() => {}).finally(() => { inflight--; });
  }

  async function start() {
    if (on || starting) return; starting = true; say('Starting…', 'starting');
    try {
      if (!(await check())) { pill.hidden = true; V.toast(status && status.reason ? status.reason : 'Voice input is not available on this device.', { key: 'voice', ms: 6000 }); return; }
      const r = await V.api('/api/voice/start', { native: !!status.native }); if (!r || !r.ok) { pill.hidden = true; V.toast((r && r.reason) || 'Voice input could not start.', { key: 'voice', ms: 6000 }); return; }
      if (r.native) {                                                  // the program itself listens to the microphone (no browser involved); the page only asks what was heard
        on = true; loudest = 0; since = 0; hinted = false; mic.setAttribute('aria-pressed', 'true'); say('Listening - ask your question', 'armed');
        poll = setInterval(() => { if (on && inflight < 1) { inflight++; V.api('/api/voice/audio', { pcm: '' }).then(onReply).catch(() => {}).finally(() => { inflight--; }); } }, 150); return;
      }
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const wurl = window.__VGPT_WORKLET ? URL.createObjectURL(new Blob([window.__VGPT_WORKLET], { type: 'text/javascript' })) : '/modern/voice-worklet.js';           // the packed page carries the worklet's source: nothing is loaded by address
      try { await ctx.audioWorklet.addModule(wurl); } finally { if (wurl.startsWith('blob:')) URL.revokeObjectURL(wurl); }
      src = ctx.createMediaStreamSource(stream); node = new AudioWorkletNode(ctx, 'pcm-down'); node.port.onmessage = e => send(e.data); src.connect(node);
      on = true; mic.setAttribute('aria-pressed', 'true'); say('Listening - say “Vantage …”', 'listening');
    } catch (e) {
      await release(); pill.hidden = true; V.api('/api/voice/stop', {}).catch(() => {});
      V.toast(e && e.name === 'NotAllowedError' ? 'The microphone is blocked for this app. Allow it and try again.' : e && e.name === 'NotFoundError' ? 'No microphone was found.' : 'Voice input could not start: ' + (e && e.message || e), { key: 'voice', ms: 6000 });
    } finally { starting = false; }
  }

  async function release() {
    try { if (node) { node.port.onmessage = null; node.disconnect(); } if (src) src.disconnect(); if (stream) stream.getTracks().forEach(t => t.stop()); if (ctx && ctx.state !== 'closed') await ctx.close(); } catch { /* already closed */ }
    node = src = stream = ctx = null;
  }

  async function stop(message) {
    if (poll) { clearInterval(poll); poll = null; }
    const was = on; on = false; mic.setAttribute('aria-pressed', 'false'); pill.hidden = true; level(0); await release();
    if (was) { V.api('/api/voice/stop', {}).catch(() => {}); if (message) V.toast(message, { key: 'voice', ms: 2500 }); }
  }

  mic.addEventListener('click', () => { if (on) { say('Yes? Say your question…', 'armed'); V.api('/api/voice/arm', {}).catch(() => {}); } else start(); });          // pressed while listening: the next sentence is the question (the Stop button ends listening)
  stopB.addEventListener('click', () => stop('Stopped listening'));
  window.addEventListener('pagehide', () => { if (on) navigator.sendBeacon && navigator.sendBeacon('/api/voice/stop', new Blob(['{}'], { type: 'application/json' })); });
  window.VoiceMute = m => { muted = !!m; };                         // app.js: the car is speaking an answer
  window.Voice = { start, stop, check, get on() { return on; } };
  setTimeout(check, 1500);
})();
