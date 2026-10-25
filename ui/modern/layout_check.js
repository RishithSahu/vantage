/* Layout self-check (a developer tool, not loaded by the page).  In the browser console:
     const s = document.createElement('script'); s.src = '/modern/layout_check.js'; document.head.append(s)
     __audit('now')                      one report for the current screen size and page
     await __sweep([['world_aster.db', 'EV']], ['ask', 'car', 'energy', 'map', 'cam'])      every page of every vehicle type listed
   It reports: the whole page scrolling, horizontal overflow, scroll areas inside scroll areas ("double scrolling"), text cut off by its box, things outside the screen, header controls that overlap. */
window.__audit = function (label) {
  const out = { label, vw: innerWidth, vh: innerHeight, issues: [] }; const se = document.scrollingElement;
  const name = el => { let s = el.tagName.toLowerCase(); if (el.id) s += '#' + el.id; else if (el.className && typeof el.className === 'string') s += '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.'); return s; };
  if (se.scrollWidth > innerWidth + 1) out.issues.push('HORIZONTAL OVERFLOW of the page: ' + se.scrollWidth + ' > ' + innerWidth);
  if (se.scrollHeight > innerHeight + 1) out.issues.push('THE WHOLE PAGE SCROLLS: ' + se.scrollHeight + ' > ' + innerHeight);
  const vis = el => { const cs = getComputedStyle(el); if (cs.display === 'none' || cs.visibility === 'hidden' || +cs.opacity === 0) return false; const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const skip = el => !!el.closest('dialog:not([open])') || !!el.closest('.skip') || !!el.closest('.toasts');
  const scr = [];
  for (const el of document.querySelectorAll('body *')) {
    if (skip(el) || !vis(el)) continue; const cs = getComputedStyle(el);
    if (['auto', 'scroll'].includes(cs.overflowY) && el.scrollHeight > el.clientHeight + 2) scr.push({ el, axis: 'y' });
    if (['auto', 'scroll'].includes(cs.overflowX) && el.scrollWidth > el.clientWidth + 2) scr.push({ el, axis: 'x' });
  }
  const set = new Set(scr.map(s => s.el)); const inScroller = el => { for (let p = el.parentElement; p; p = p.parentElement) if (set.has(p)) return true; return false; };
  for (const s of scr) { let p = s.el.parentElement; while (p && p !== document.body) { if (set.has(p)) { out.issues.push('DOUBLE SCROLL: ' + name(s.el) + ' (scrolls ' + s.axis + ') inside ' + name(p)); break; } p = p.parentElement; } }
  out.scrollers = [...new Set(scr.map(s => name(s.el) + ':' + s.axis))];
  let clipped = 0;
  for (const el of document.querySelectorAll('body *')) {
    if (skip(el) || !vis(el) || el.children.length || inScroller(el)) continue; if (!el.textContent.trim()) continue; const r = el.getBoundingClientRect(); let p = el.parentElement;
    while (p && p !== document.documentElement) { const cs = getComputedStyle(p); if (cs.overflow !== 'visible') { const pr = p.getBoundingClientRect(); if (r.right > pr.right + 3 || r.bottom > pr.bottom + 3 || r.left < pr.left - 3 || r.top < pr.top - 3) { if (clipped++ < 6) out.issues.push('CLIPPED: "' + el.textContent.trim().slice(0, 28) + '" (' + name(el) + ') cut by ' + name(p)); } break; } p = p.parentElement; }
  }
  let off = 0;
  for (const el of document.querySelectorAll('body *')) {
    if (skip(el) || !vis(el) || el.children.length > 3 || inScroller(el)) continue; const r = el.getBoundingClientRect();
    if (r.right > innerWidth + 2 || r.left < -2 || r.bottom > innerHeight + 2) { if (off++ < 5) out.issues.push('OFF SCREEN: ' + name(el) + ' [' + Math.round(r.left) + ',' + Math.round(r.top) + ',' + Math.round(r.right) + ',' + Math.round(r.bottom) + ']'); }
  }
  const hb = [...document.querySelectorAll('.bar .brand, .bar .hud, .bar .ctl > *')].filter(vis).map(e => [name(e), e.getBoundingClientRect()]);
  for (let i = 0; i < hb.length; i++) for (let j = i + 1; j < hb.length; j++) { const a = hb[i][1], b = hb[j][1]; if (a.left < b.right - 1 && b.left < a.right - 1 && a.top < b.bottom - 1 && b.top < a.bottom - 1) out.issues.push('OVERLAP in the header: ' + hb[i][0] + ' / ' + hb[j][0]); }
  for (const el of document.querySelectorAll('[hidden]')) { if (getComputedStyle(el).display !== 'none' && !el.closest('dialog:not([open])')) out.issues.push('MARKED HIDDEN BUT STILL DRAWN: ' + name(el)); }      // the page said "hide this" and a style rule overrode it
  return out;
};
window.__sweep = async function (list, pages, dialogs) {
  const sleep = ms => new Promise(r => setTimeout(r, ms)); const res = [];
  for (const [ds, tag] of list) {
    await fetch('/api/dataset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: ds }) }); await sleep(4200);
    for (const p of pages) { location.hash = '#' + p; await sleep(1300); const a = window.__audit(tag + ' ' + p); res.push(a.issues.length || a.scrollers.length > 1 ? { l: a.label, vw: a.vw + 'x' + a.vh, i: a.issues, s: a.scrollers } : { l: a.label, ok: true, s: a.scrollers }); }
    if (dialogs) {                                                                                      // the settings sheet and the question browser must also fit
      for (const t of ['look', 'units', 'sim']) { document.querySelector('#bSet').click(); await sleep(300); document.querySelector('.tab[data-t=' + t + ']').click(); await sleep(400); document.getAnimations().forEach(a => { try { a.finish(); } catch (e) {} }); await sleep(80); const sh = document.querySelector('#sheet'), r = sh.getBoundingClientRect(), b = document.querySelector('#sbody'), i = [];
        if (r.bottom > innerHeight + 1 || r.right > innerWidth + 1) i.push('settings sheet outside the screen'); if (b.scrollHeight > b.clientHeight + 2 && sh.scrollHeight > sh.clientHeight + 2) i.push('double scroll in the settings sheet');
        res.push(i.length ? { l: tag + ' settings:' + t, vw: innerWidth + 'x' + innerHeight, i } : { l: tag + ' settings:' + t, ok: true }); document.querySelector('#bSetX').click(); await sleep(250); }
      document.querySelector('#bPal').click(); await sleep(300); document.getAnimations().forEach(a => { try { a.finish(); } catch (e) {} }); await sleep(80); const pal = document.querySelector('#palette'), pr = pal.getBoundingClientRect(), list = document.querySelector('#palList'), pi = [];
      if (pr.bottom > innerHeight + 1 || pr.right > innerWidth + 1) pi.push('question browser outside the screen'); if (list.scrollHeight > list.clientHeight + 2 && pal.scrollHeight > pal.clientHeight + 2) pi.push('double scroll in the question browser');
      res.push(pi.length ? { l: tag + ' palette', vw: innerWidth + 'x' + innerHeight, i: pi } : { l: tag + ' palette', ok: true }); pal.close(); await sleep(200);
    }
  }
  return res;
};
