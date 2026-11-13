# Screenshot of a page in headless Chrome over CDP (python websocket-client), with scripted clicks.
# usage: cdp_shot.py URL OUT.png WIDTH HEIGHT [--js EXPR | --js WAIT:EXPR | --js SLEEP:S]... [--timeout S] [--tmp DIR]
# Loads a page in headless Chrome via CDP, prints console errors/exceptions, takes a screenshot.
import json, subprocess, sys, time, urllib.request, base64, websocket, shutil, argparse, os
ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("out"); ap.add_argument("w", type=int); ap.add_argument("h", type=int)
ap.add_argument("--wait", default=""); ap.add_argument("--js", action="append", default=[]); ap.add_argument("--timeout", type=float, default=20)
ap.add_argument("--settle", type=float, default=2); ap.add_argument("--scale", type=float, default=1); ap.add_argument("--port", type=int, default=9641)
ap.add_argument("--eval", default=""); ap.add_argument("--tmp", default="/tmp")
a = ap.parse_args()
prof = os.path.join(a.tmp, f"chrome-{a.port}")
shutil.rmtree(prof, ignore_errors=True)
p = subprocess.Popen(["google-chrome", "--headless=new", "--no-sandbox", f"--remote-debugging-port={a.port}", "--hide-scrollbars",
                      f"--user-data-dir={prof}", f"--window-size={a.w},{a.h}", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
errors = []
try:
    for _ in range(100):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/json")); break
        except Exception: time.sleep(0.2)
    ws = websocket.create_connection([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"], timeout=600)
    n = [0]
    def send(m, **params):
        n[0] += 1; ws.send(json.dumps({"id": n[0], "method": m, "params": params})); return n[0]
    def handle(r):
        m = r.get("method")
        if m == "Runtime.consoleAPICalled":
            t = r["params"]["type"]; txt = " ".join(str(x.get("value", x.get("description"))) for x in r["params"]["args"])
            if t in ("error", "warning"): print(f"console.{t}: {txt}")
            if t == "error": errors.append(txt)
        elif m == "Runtime.exceptionThrown":
            d = r["params"]["exceptionDetails"]; txt = d.get("exception", {}).get("description", d.get("text"))
            print("exception:", txt); errors.append(txt)
        elif m == "Log.entryAdded":
            e = r["params"]["entry"]
            if e["level"] in ("error",): print(f"log.{e['level']}: {e['text']} {e.get('url','')}"); errors.append(e["text"])
    def wait(i):
        while True:
            r = json.loads(ws.recv()); handle(r)
            if r.get("id") == i: return r
    def ev(expr):
        r = wait(send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True))
        return r["result"].get("result", {}).get("value")
    wait(send("Runtime.enable")); wait(send("Log.enable")); wait(send("Page.enable"))
    wait(send("Emulation.setDeviceMetricsOverride", width=a.w, height=a.h, deviceScaleFactor=a.scale, mobile=False))
    wait(send("Page.navigate", url=a.url))
    time.sleep(1.5)
    for js in a.js:
        if js.startswith("WAIT:"):
            t0 = time.time()
            while time.time() - t0 < a.timeout and not ev(js[5:]): time.sleep(0.5)
            print(f"waited {time.time()-t0:.1f}s for {js[5:][:60]}")
        elif js.startswith("SLEEP:"):
            time.sleep(float(js[6:]))
        else:
            ev(js); time.sleep(0.3)
    if a.wait:
        t0 = time.time()
        while time.time() - t0 < a.timeout and not ev(a.wait): time.sleep(0.5)
    time.sleep(a.settle)
    if a.eval:
        print("eval:", ev(a.eval))
    r = wait(send("Page.captureScreenshot", format="png"))
    open(a.out, "wb").write(base64.b64decode(r["result"]["data"]))
    print(f"errors={len(errors)}")
finally:
    p.terminate(); p.wait()
