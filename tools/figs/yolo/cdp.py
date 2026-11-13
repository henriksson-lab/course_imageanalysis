"""Minimal headless-Chrome driver over the DevTools protocol, for the yolo/ figures and the
measured Explore numbers (tools/figs/yolo/demo_shots.py, measure_yolonet.py).

Serves a temporary folder holding harness.html and a link to the repository's demos/ (with the
built demos, the Wasm modules and the models from `make wasm` / `make models`), opens one page
per demo and lets the caller evaluate JavaScript, wait for conditions and take screenshots of
elements. Needs google-chrome and the python `websocket-client` module."""
import base64, json, os, shutil, subprocess, sys, tempfile, time, urllib.request

import websocket

ROOT = os.environ.get("REPO_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))


class Browser:
    def __init__(self, port=9634, cdp_port=9688, width=1240, height=1000, scale=2):
        self.tmp = tempfile.mkdtemp(prefix="yolofigs-")
        www = os.path.join(self.tmp, "www")
        os.makedirs(www)
        os.symlink(os.path.join(ROOT, "demos"), os.path.join(www, "demos"))
        shutil.copy(os.path.join(HERE, "harness.html"), os.path.join(www, "h.html"))
        self.port = port
        self.srv = subprocess.Popen([sys.executable, "-m", "http.server", "--directory", www, str(port)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.chrome = subprocess.Popen(["google-chrome", "--headless=new", "--no-sandbox", f"--remote-debugging-port={cdp_port}",
                                        f"--user-data-dir={self.tmp}/prof", f"--window-size={width},{height}",
                                        f"--force-device-scale-factor={scale}", "--hide-scrollbars", "about:blank"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json"))
                break
            except Exception:
                time.sleep(0.2)
        self.ws = websocket.create_connection([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"], timeout=600)
        self.n = 0
        self.errors = []
        for m in ("Runtime.enable", "Log.enable", "Page.enable"):
            self.call(m)
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=scale, mobile=False)

    def _handle(self, r):
        m = r.get("method")
        if m == "Runtime.consoleAPICalled" and r["params"]["type"] == "error":
            self.errors.append(" ".join(str(a.get("value", a.get("description"))) for a in r["params"]["args"]))
        elif m == "Runtime.exceptionThrown":
            d = r["params"]["exceptionDetails"]
            self.errors.append(d.get("exception", {}).get("description", d.get("text")))
        elif m == "Log.entryAdded" and r["params"]["entry"]["level"] == "error":
            self.errors.append(r["params"]["entry"]["text"])

    def call(self, method, **params):
        self.n += 1
        i = self.n
        self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        while True:
            r = json.loads(self.ws.recv())
            self._handle(r)
            if r.get("id") == i:
                return r.get("result", {})

    def ev(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def until(self, expr, timeout=120):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.ev(expr):
                return time.time() - t0
            time.sleep(0.2)
        raise TimeoutError(expr)

    def open(self, demo, preset="", extra=""):
        self.call("Page.navigate", url=f"http://localhost:{self.port}/h.html?d={demo}&p={preset}{extra}")
        time.sleep(0.5)
        self.until("document.readyState === 'complete' && !!document.querySelector('.qb-demo')")

    def click_preset(self, label):
        ok = self.ev("(()=>{const b=[...document.querySelectorAll('.qb-presets button')].find(b=>b.textContent===%s);"
                     "if(!b) return false; b.click(); return true})()" % json.dumps(label))
        if not ok:
            raise KeyError(label)

    def shot(self, path, selector=".qb-demo", pad=4, clip=None):
        if clip is None:
            clip = self.ev("(()=>{const r=document.querySelector(%s).getBoundingClientRect();"
                           "return {x:r.left,y:r.top+scrollY,width:r.width,height:r.height}})()" % json.dumps(selector))
            clip = {"x": max(0, clip["x"] - pad), "y": max(0, clip["y"] - pad), "width": clip["width"] + 2 * pad, "height": clip["height"] + 2 * pad}
        clip["scale"] = 1
        r = self.call("Page.captureScreenshot", format="png", clip=clip, captureBeyondViewport=True)
        open(path, "wb").write(base64.b64decode(r["data"]))

    def close(self):
        try:
            self.ws.close()
        finally:
            self.chrome.terminate(); self.chrome.wait()
            self.srv.terminate(); self.srv.wait()
            shutil.rmtree(self.tmp, ignore_errors=True)
