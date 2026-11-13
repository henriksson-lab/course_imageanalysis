"""Screenshots of course pages or demo states in headless Chrome over the DevTools protocol.

Used for the stardist/ slide figures (demo states of starrays and of the real-model demo
stardistnet, which needs `make wasm` + `make models` and a server) and for console checks.

  python3 -I tools/figs/stardist/shot.py URL OUT.png [--wait JS] [--do JS] [--size 1220x900]
      [--scale 2] [--clip SELECTOR] [--timeout 60] [--port 9631]

--wait: a JS expression polled until truthy (e.g. a readout containing "after NMS");
--do:   JS run after the page has loaded and before waiting (clicks, slider changes; may be
        given several times, run in order, each followed by --wait if given);
--clip: CSS selector of the element to capture (else the viewport).
Prints console messages and exceptions; exit code 1 if there were errors.
"""
import argparse, base64, json, shutil, subprocess, sys, tempfile, time, urllib.request
import websocket

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("out")
ap.add_argument("--wait", default=""); ap.add_argument("--do", action="append", default=[])
ap.add_argument("--size", default="1220x900"); ap.add_argument("--scale", type=float, default=2)
ap.add_argument("--clip", default=""); ap.add_argument("--timeout", type=float, default=60)
ap.add_argument("--port", type=int, default=9679); ap.add_argument("--settle", type=float, default=0.6)
a = ap.parse_args()
W, H = map(int, a.size.split("x"))
prof = tempfile.mkdtemp(prefix="qbshot-")
p = subprocess.Popen(["google-chrome", "--headless=new", "--no-sandbox", f"--remote-debugging-port={a.port}",
                      f"--user-data-dir={prof}", "--hide-scrollbars", "about:blank"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
errors = []
try:
    for _ in range(100):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/json")); break
        except Exception:
            time.sleep(0.2)
    ws = websocket.create_connection([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"], timeout=a.timeout + 30)
    n = [0]

    def handle(r):
        m = r.get("method")
        if m == "Runtime.consoleAPICalled":
            t = r["params"]["type"]
            txt = " ".join(str(x.get("value", x.get("description"))) for x in r["params"]["args"])
            print(f"console.{t}: {txt}")
            if t == "error": errors.append(txt)
        elif m == "Runtime.exceptionThrown":
            d = r["params"]["exceptionDetails"]
            txt = d.get("exception", {}).get("description", d.get("text"))
            print("exception:", txt); errors.append(txt)
        elif m == "Log.entryAdded":
            e = r["params"]["entry"]
            print(f"log.{e['level']}: {e['text']} {e.get('url', '')}")
            if e["level"] == "error": errors.append(e["text"])

    def call(meth, **params):
        n[0] += 1; i = n[0]
        ws.send(json.dumps({"id": i, "method": meth, "params": params}))
        while True:
            r = json.loads(ws.recv()); handle(r)
            if r.get("id") == i: return r

    def ev(expr):
        r = call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("result", {}).get("value")

    def wait_for(expr):
        t0 = time.time()
        while time.time() - t0 < a.timeout:
            if ev(f"(()=>{{try{{return !!({expr})}}catch(e){{return false}}}})()"): return True
            time.sleep(0.3)
        print("TIMEOUT waiting for", expr); return False

    call("Runtime.enable"); call("Log.enable"); call("Page.enable")
    call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=a.scale, mobile=False)
    call("Page.navigate", url=a.url)
    wait_for("document.readyState === 'complete'")
    time.sleep(a.settle)
    if a.wait: wait_for(a.wait)
    for js in a.do:
        ev(js); time.sleep(a.settle)
        if a.wait: wait_for(a.wait)
    time.sleep(a.settle)
    params = {"format": "png"}
    if a.clip:
        b = ev(f"(()=>{{const e=document.querySelector({json.dumps(a.clip)}); e.scrollIntoView({{block:'start'}}); const r=e.getBoundingClientRect(); return [r.x,r.y,r.width,r.height]}})()")
        time.sleep(0.3)
        b = ev(f"(()=>{{const r=document.querySelector({json.dumps(a.clip)}).getBoundingClientRect(); return [r.x,r.y,r.width,r.height]}})()")
        params["clip"] = {"x": b[0], "y": b[1], "width": b[2], "height": b[3], "scale": 1}
    r = call("Page.captureScreenshot", **params)
    open(a.out, "wb").write(base64.b64decode(r["result"]["data"]))
    print(f"wrote {a.out}; errors: {len(errors)}")
finally:
    p.terminate(); p.wait()
    shutil.rmtree(prof, ignore_errors=True)
sys.exit(1 if errors else 0)
