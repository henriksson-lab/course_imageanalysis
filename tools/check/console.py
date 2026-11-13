#!/usr/bin/env python3
"""Open pages of the rendered site in headless Chrome and report browser console errors.

Each page is loaded in real time over the DevTools protocol and watched for a few seconds
after its load event. Counted as errors: uncaught exceptions (page and Web Workers),
console.error calls, and browser log entries at level "error" (failed resource loads,
blocked scripts). Warnings are not counted. Model demos that are still downloading or running
their network when the time is up are fine: only errors count.

Usage (normally through tools/check/console.sh, which starts the local server):
  python3 tools/check/console.py BASE_URL [--jobs N] [--wait S] [--list FILE | PAGE ...]
PAGE is a path relative to the site root, e.g. canny/lec.html. Exit 1 if any page has errors.
Needs google-chrome (or $CHROME) and the Python package websocket-client.
"""
import argparse
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

import websocket

# Messages that are not problems of the site.
IGNORE = ('favicon.ico', 'GPU stall due to ReadPixels', 'ResizeObserver loop')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Browser:
    def __init__(self, chrome):
        self.port = free_port()
        self.prof = tempfile.mkdtemp(prefix='console-check-')
        self.proc = subprocess.Popen(
            [chrome, '--headless=new', '--no-sandbox', f'--remote-debugging-port={self.port}',
             f'--user-data-dir={self.prof}', '--window-size=1300,900', '--mute-audio',
             '--no-first-run', 'about:blank'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tabs = None
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f'http://127.0.0.1:{self.port}/json', timeout=2))
                break
            except Exception:
                time.sleep(0.2)
        if tabs is None:
            raise RuntimeError('Chrome did not start')
        page = [t for t in tabs if t['type'] == 'page'][0]
        self.ws = websocket.create_connection(page['webSocketDebuggerUrl'], timeout=60)
        self.n = 0
        self.errors = []

    def send(self, method, session=None, **params):
        self.n += 1
        msg = {'id': self.n, 'method': method, 'params': params}
        if session:
            msg['sessionId'] = session
        self.ws.send(json.dumps(msg))
        return self.n

    def handle(self, r):
        m = r.get('method')
        p = r.get('params', {})
        txt = None
        if m == 'Runtime.exceptionThrown':
            d = p['exceptionDetails']
            txt = 'exception: ' + str(d.get('exception', {}).get('description') or d.get('text'))
            if d.get('url'):
                txt += f" ({d['url']}:{d.get('lineNumber')})"
        elif m == 'Runtime.consoleAPICalled' and p.get('type') == 'error':
            txt = 'console.error: ' + ' '.join(
                str(a.get('value', a.get('description', ''))) for a in p.get('args', []))
        elif m == 'Log.entryAdded' and p['entry']['level'] == 'error':
            e = p['entry']
            txt = f"log: {e['text']} {e.get('url', '')}".strip()
        elif m == 'Target.attachedToTarget':
            sid = p['sessionId']
            for meth in ('Runtime.enable', 'Log.enable', 'Runtime.runIfWaitingForDebugger'):
                self.send(meth, session=sid)
        if txt and not any(s in txt for s in IGNORE):
            self.errors.append(txt[:400])

    def wait_id(self, i, deadline):
        while time.time() < deadline:
            try:
                r = json.loads(self.ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            self.handle(r)
            if r.get('id') == i:
                return r
        return None

    def pump(self, until):
        self.ws.settimeout(0.25)
        while time.time() < until:
            try:
                self.handle(json.loads(self.ws.recv()))
            except websocket.WebSocketTimeoutException:
                pass
        self.ws.settimeout(60)

    def check(self, url, wait):
        self.errors = []
        dl = time.time() + 30
        self.wait_id(self.send('Page.navigate', url='about:blank'), dl)
        for meth in ('Runtime.enable', 'Log.enable', 'Page.enable'):
            self.wait_id(self.send(meth), dl)
        self.wait_id(self.send('Target.setAutoAttach', autoAttach=True,
                               waitForDebuggerOnStart=False, flatten=True), dl)
        self.errors = []
        self.send('Page.navigate', url=url)
        # Wait for the load event (at most 30 s), then watch for `wait` seconds more.
        self.ws.settimeout(0.5)
        t0 = time.time()
        while time.time() - t0 < 30:
            try:
                r = json.loads(self.ws.recv())
            except websocket.WebSocketTimeoutException:
                continue
            self.handle(r)
            if r.get('method') == 'Page.loadEventFired':
                break
        self.pump(time.time() + wait)
        return list(dict.fromkeys(self.errors))

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        shutil.rmtree(self.prof, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('pages', nargs='*')
    ap.add_argument('--list')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--wait', type=float, default=6.0)
    a = ap.parse_args()
    pages = list(a.pages)
    if a.list:
        with open(a.list) as f:
            pages += [ln.strip() for ln in f if ln.strip()]
    chrome = os.environ.get('CHROME') or shutil.which('google-chrome') or shutil.which('chromium')
    q = queue.Queue()
    for p in pages:
        q.put(p)
    results = {}
    lock = threading.Lock()

    def worker():
        b = None
        while True:
            try:
                p = q.get_nowait()
            except queue.Empty:
                break
            for attempt in range(2):
                try:
                    if b is None:
                        b = Browser(chrome)
                    errs = b.check(a.base.rstrip('/') + '/' + p, a.wait)
                    break
                except Exception as e:  # a crashed browser: restart it once
                    errs = [f'checker: {type(e).__name__}: {e}']
                    if b:
                        b.close()
                    b = None
            with lock:
                results[p] = errs
                print(f"{'FAIL' if errs else 'ok  '} {p}" + ''.join(f'\n     {e}' for e in errs), flush=True)
        if b:
            b.close()

    threads = [threading.Thread(target=worker) for _ in range(max(1, a.jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = {p: e for p, e in results.items() if e}
    print(f'console: {len(results)} pages, {len(bad)} with errors, '
          f'{sum(len(e) for e in bad.values())} errors')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
