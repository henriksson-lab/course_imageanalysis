#!/usr/bin/env python3
"""Check that every interactive demo of the rendered site fits on the screen.

Serves docs/ on a free local port and opens each page that contains demos in headless Chrome
(DevTools protocol, the Browser class of console.py). After the demos have mounted, it measures
every .qb-demo: its box, its main figure (the largest canvas/svg over 150x150 px) and its
control panel (.qb-panel, else the element holding the most inputs and buttons that does not
contain a figure).

  pages  (*/exercise.html, demos.html, overview.html, topics/*.html) at 1366x768 and 1920x1080
  slides (*/lec.html) at 1280x720, each slide holding a demo shown with Reveal.slide(h, v);
         sizes are in slide units (the slide is 1280x720, reveal scales it to the window)

A demo FAILS when
  - it is taller than the viewport (slide) height minus 60 px            "too tall"
  - (pages) its control panel starts below the bottom of its main figure
    (the controls wrapped underneath; a single toolbar row, at most 100 px high and spanning
    most of the figure's width, is allowed)                                "controls below figure"
  - (pages) it sticks out of the viewport on the right                   "wider than viewport"
  - (slides) it sticks out of the slide                                  "overflows slide"

Usage: python3 tools/check/fit.py [--out PREFIX] [--jobs N] [--site docs] [PAGE ...]
       python3 tools/check/fit.py --rejudge OLD.json --out PREFIX   (apply the rules again)
PAGE is relative to the site, e.g. asm/exercise.html (default: every page with demos).
Writes PREFIX.json and PREFIX.txt (default /tmp/fit); exit 1 if any demo fails.
"""
import argparse
import functools
import glob
import http.server
import json
import os
import queue
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from console import Browser  # noqa: E402

PAGE_VIEWPORTS = [(1366, 768), (1920, 1080)]
SLIDE_VIEWPORT = (1280, 720)
MARGIN = 60          # a demo may use the viewport height minus this
FIG_MIN = 150        # canvas/svg at least this large (both sides) count as figures
TOL = 4              # px of slack for the geometric tests

# Shared by pages and slides: measure every .qb-demo inside `root`. `toLocal` maps a client
# rect to the coordinates reported (page or slide units).
MEASURE_JS = r"""
function qbMeasure(root, toLocal) {
  const CTL = 'input,button,select,textarea';
  const vis = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== 'hidden'; };
  const out = [];
  for (const d of root.querySelectorAll('.qb-demo')) {
    if (!vis(d)) { out.push({ name: d.dataset.demo, hidden: true }); continue; }
    const box = toLocal(d.getBoundingClientRect());
    const figs = [...d.querySelectorAll('canvas,svg')].filter(e => {
      if (e.parentElement && e.parentElement.closest('svg')) return false;   // nested svg
      const r = e.getBoundingClientRect();
      return r.width >= FIG_MIN && r.height >= FIG_MIN && vis(e);
    });
    let main = null, area = 0;
    for (const f of figs) { const r = f.getBoundingClientRect(); if (r.width * r.height > area) { area = r.width * r.height; main = f; } }
    let panel = [...d.querySelectorAll('.qb-panel')].filter(vis)
      .sort((a, b) => b.querySelectorAll(CTL).length - a.querySelectorAll(CTL).length)[0] || null;
    if (!panel) {
      let best = 0;
      for (const e of d.querySelectorAll('div,fieldset,form,section,aside')) {
        if (!vis(e) || figs.some(f => e.contains(f))) continue;
        const n = e.querySelectorAll(CTL).length;
        if (n > best) { best = n; panel = e; }   // first (outermost) wins ties
      }
    }
    out.push({
      name: d.dataset.demo, box,
      fig: main ? toLocal(main.getBoundingClientRect()) : null,
      panel: panel ? toLocal(panel.getBoundingClientRect()) : null,
      // the panel scrolls (taller than allowed) / its content is cut off on the right
      panelScrolls: panel ? panel.scrollHeight > panel.clientHeight + 2 : false,
      panelClipX: panel ? Math.max(0, panel.scrollWidth - panel.clientWidth) : 0,
      controls: d.querySelectorAll(CTL).length,
    });
  }
  return out;
}
""".replace('FIG_MIN', str(FIG_MIN))

READY_JS = r"""(() => {
  const hosts = [...document.querySelectorAll('[data-demo]')];
  return hosts.every(h => h.classList.contains('qb-demo') && h.childElementCount > 0)
    && document.readyState === 'complete';
})()"""

PAGE_JS = MEASURE_JS + r"""
(() => {
  const sx = scrollX, sy = scrollY;
  const loc = r => ({ x: Math.round(r.left + sx), y: Math.round(r.top + sy), w: Math.round(r.width), h: Math.round(r.height) });
  return { vw: document.documentElement.clientWidth, vh: innerHeight, demos: qbMeasure(document, loc) };
})()"""

SLIDES_JS = r"""(() => {
  if (!window.Reveal || !Reveal.isReady()) return null;
  Reveal.configure({ transition: 'none', backgroundTransition: 'none' });
  const res = [];
  for (const s of document.querySelectorAll('.reveal .slides section')) {
    if (s.querySelector('section')) continue;               // vertical stack wrapper
    if (!s.querySelector('[data-demo]')) continue;
    const i = Reveal.getIndices(s);
    res.push({ h: i.h, v: i.v || 0, id: s.id || '' });
  }
  return res;
})()"""

SLIDE_MEASURE_JS = MEASURE_JS + r"""
(() => {
  const s = Reveal.getCurrentSlide();
  const sl = document.querySelector('.reveal .slides').getBoundingClientRect();
  const cfg = Reveal.getConfig(), k = sl.width / cfg.width;
  const loc = r => ({ x: Math.round((r.left - sl.left) / k), y: Math.round((r.top - sl.top) / k), w: Math.round(r.width / k), h: Math.round(r.height / k) });
  return { vw: cfg.width, vh: cfg.height, scale: +k.toFixed(3), id: s.id || '', demos: qbMeasure(s, loc) };
})()"""


class Fit(Browser):
    def eval(self, expr, timeout=120):
        r = self.wait_id(self.send('Runtime.evaluate', expression=expr, returnByValue=True,
                                   awaitPromise=True), time.time() + timeout)
        if r is None:
            raise RuntimeError('evaluate timed out')
        res = r.get('result', {})
        if 'exceptionDetails' in res:
            raise RuntimeError('page script: ' + str(res['exceptionDetails'].get('exception', {}).get('description')))
        return res.get('result', {}).get('value')

    def load(self, url, w, h, settle):
        dl = time.time() + 30
        self.wait_id(self.send('Page.enable'), dl)
        self.wait_id(self.send('Emulation.setDeviceMetricsOverride', width=w, height=h,
                               deviceScaleFactor=1, mobile=False), dl)
        self.wait_id(self.send('Page.navigate', url='about:blank'), dl)
        self.send('Page.navigate', url=url)
        self.ws.settimeout(0.5)
        t0 = time.time()
        while time.time() - t0 < 40:
            try:
                r = json.loads(self.ws.recv())
            except Exception:
                continue
            if r.get('method') == 'Page.loadEventFired':
                break
        self.ws.settimeout(60)
        t0 = time.time()
        while time.time() - t0 < 30 and not self.eval(READY_JS):
            time.sleep(0.3)
        time.sleep(settle)


def judge(m, kind):
    """Failure reasons of one measured demo."""
    if m.get('hidden'):
        return []
    b, vh, vw = m['box'], m['vh'], m['vw']
    why = []
    if b['h'] > vh - MARGIN:
        why.append(f"too tall ({b['h']} > {vh - MARGIN})")
    if kind == 'page':
        f, p = m['fig'], m['panel']
        # A one-row toolbar spanning the figure under it is a design, not a wrap.
        if f and p and p['y'] >= f['y'] + f['h'] - TOL and (p['h'] > 100 or p['w'] < 0.6 * f['w']):
            why.append(f"controls below figure (panel top {p['y'] - b['y']} >= figure bottom {f['y'] + f['h'] - b['y']})")
        if b['x'] + b['w'] > vw + TOL or b['x'] < -TOL:
            why.append(f"wider than viewport (x {b['x']}..{b['x'] + b['w']} of {vw})")
    else:
        if b['y'] + b['h'] > vh + TOL or b['x'] + b['w'] > vw + TOL or b['x'] < -TOL or b['y'] < -TOL:
            why.append(f"overflows slide (x {b['x']}..{b['x'] + b['w']}, y {b['y']}..{b['y'] + b['h']} of {vw}x{vh})")
    return why


def measure_page(br, base, page, settle):
    rows = []
    for w, h in PAGE_VIEWPORTS:
        br.load(f'{base}/{page}', w, h, settle)
        r = br.eval(PAGE_JS)
        for m in r['demos']:
            m.update(page=page, viewport=f'{w}x{h}', vw=r['vw'], vh=r['vh'])
            m['fail'] = judge(m, 'page')
            rows.append(m)
    return rows


def measure_slides(br, base, page, settle):
    w, h = SLIDE_VIEWPORT
    br.load(f'{base}/{page}', w, h, settle)
    t0 = time.time()
    slides = None
    while time.time() - t0 < 15:
        slides = br.eval(SLIDES_JS)
        if slides is not None:
            break
        time.sleep(0.3)
    rows = []
    for s in slides or []:
        br.eval(f"Reveal.slide({s['h']}, {s['v']})")
        time.sleep(max(0.8, settle / 2))
        r = br.eval(SLIDE_MEASURE_JS)
        for m in r['demos']:
            m.update(page=page, slide=s['id'] or f"{s['h']}/{s['v']}", viewport=f'{w}x{h}',
                     vw=r['vw'], vh=r['vh'], scale=r['scale'])
            m['fail'] = judge(m, 'slide')
            rows.append(m)
    return rows


def default_pages(site):
    pats = ['*/exercise.html', '*/lec.html', 'demos.html', 'overview.html', 'topics/*.html']
    pages = []
    for p in pats:
        for f in sorted(glob.glob(os.path.join(site, p))):
            rel = os.path.relpath(f, site)
            if rel.startswith(('site_libs', 'demos/')):
                continue
            with open(f, encoding='utf-8', errors='replace') as fh:
                if 'data-demo=' in fh.read():
                    pages.append(rel)
    return pages


def serve(site):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    handler = functools.partial(Quiet, directory=site)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f'http://127.0.0.1:{srv.server_address[1]}'


def table(rows):
    fails = [r for r in rows if r['fail']]
    lines = [f"{'page':34} {'slide':22} {'demo':16} {'viewport':10} {'w':>5} {'h':>5}  reason"]
    for r in fails:
        b = r['box']
        lines.append(f"{r['page']:34} {r.get('slide', ''):22.22} {r['name'] or '?':16} {r['viewport']:10} "
                     f"{b['w']:5} {b['h']:5}  {'; '.join(r['fail'])}")
    names = sorted({r['name'] for r in fails if r['name']})
    lines.append('')
    lines.append(f"fit: {len(rows)} demo measurements, {len(fails)} fail "
                 f"({sum(1 for r in fails if r['viewport'] == '1366x768')} at 1366x768, "
                 f"{sum(1 for r in fails if r['viewport'] == '1920x1080')} at 1920x1080, "
                 f"{sum(1 for r in fails if 'slide' in r)} on slides); "
                 f"{len(names)} distinct demos fail: {' '.join(names)}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pages', nargs='*')
    ap.add_argument('--site', default='docs')
    ap.add_argument('--out', default='/tmp/fit')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--settle', type=float, default=1.5, help='seconds to wait after the demos mounted')
    ap.add_argument('--rejudge', help='re-apply the rules to the measurements in this JSON file')
    a = ap.parse_args()
    if a.rejudge:
        with open(a.rejudge) as f:
            d = json.load(f)
        for m in d['measurements']:
            m['fail'] = judge(m, 'slide' if 'slide' in m else 'page')
        return write(a.out, d['pages'], d['errors'], d['measurements'])
    pages = a.pages or default_pages(a.site)
    chrome = os.environ.get('CHROME') or shutil.which('google-chrome') or shutil.which('chromium')
    srv, base = serve(a.site)
    q = queue.Queue()
    for p in pages:
        q.put(p)
    rows, errors = [], []
    lock = threading.Lock()

    def worker():
        br = None
        while True:
            try:
                p = q.get_nowait()
            except queue.Empty:
                break
            res = None
            for _ in range(2):
                try:
                    if br is None:
                        br = Fit(chrome)
                    settle = a.settle * (3 if p == 'demos.html' else 1)
                    res = (measure_slides if p.endswith('lec.html') else measure_page)(br, base, p, settle)
                    break
                except Exception as e:  # restart a stuck browser once
                    err = f'{p}: {type(e).__name__}: {e}'
                    if br:
                        br.close()
                    br = None
            with lock:
                if res is None:
                    errors.append(err)
                    print('ERR ', err, flush=True)
                else:
                    rows.extend(res)
                    nf = sum(1 for r in res if r['fail'])
                    print(f"{'FAIL' if nf else 'ok  '} {p}: {len(res)} measurements, {nf} fail", flush=True)
        if br:
            br.close()

    threads = [threading.Thread(target=worker) for _ in range(max(1, a.jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    srv.shutdown()
    return write(a.out, pages, errors, rows)


def write(out, pages, errors, rows):
    rows.sort(key=lambda r: (r['page'], r.get('slide', ''), r['viewport'], r['box']['y'] if 'box' in r else 0))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + '.json', 'w') as f:
        json.dump({'pages': pages, 'errors': errors, 'measurements': rows}, f, indent=1)
    txt = table(rows) + ''.join(f'\nchecker error: {e}' for e in errors)
    with open(out + '.txt', 'w') as f:
        f.write(txt + '\n')
    print(txt)
    return 1 if any(r['fail'] for r in rows) or errors else 0


if __name__ == '__main__':
    sys.exit(main())
