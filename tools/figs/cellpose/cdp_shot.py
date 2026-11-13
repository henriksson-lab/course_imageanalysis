"""Screenshots and console check of a page in headless Chrome over the DevTools protocol.
usage: python3 -I cdp_shot.py <url> <steps.json | -> [--size 1220x900] [--scale 2]
steps: a JSON list of {"js": expr} (run), {"wait": expr, "timeout": s} (until truthy),
{"sleep": s}, {"shot": path, "selector": css (clip to the element, optional), "pad": px},
{"print": expr}. Console errors and exceptions are printed; exit code 1 if any.
Needs google-chrome and the websocket-client module."""
import base64, json, os, socket, subprocess, sys, tempfile, time, urllib.request
import websocket

url, steps_arg = sys.argv[1], sys.argv[2]
size, scale = (1220, 900), 2
a = sys.argv[3:]
for i in range(0, len(a), 2):
    if a[i] == "--size": size = tuple(int(v) for v in a[i + 1].split("x"))
    if a[i] == "--scale": scale = float(a[i + 1])
steps = json.load(sys.stdin) if steps_arg == "-" else json.load(open(steps_arg))
prof = tempfile.mkdtemp(prefix="cdp-chrome-")
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
p = subprocess.Popen(["google-chrome", "--headless=new", "--no-sandbox", "--hide-scrollbars", f"--remote-debugging-port={port}",
                      f"--user-data-dir={prof}", f"--window-size={size[0]},{size[1]}", "about:blank"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
errors = []
try:
    for _ in range(100):
        try: tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json")); break
        except Exception: time.sleep(0.2)
    ws = websocket.create_connection([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"], timeout=900)
    n = [0]
    def send(m, **params):
        n[0] += 1; ws.send(json.dumps({"id": n[0], "method": m, "params": params})); return n[0]
    def handle(r):
        m = r.get("method")
        if m == "Runtime.consoleAPICalled" and r["params"]["type"] in ("error", "warning", "assert"):
            txt = " ".join(str(x.get("value", x.get("description"))) for x in r["params"]["args"])
            print(f"console.{r['params']['type']}: {txt}"); errors.append(txt)
        elif m == "Runtime.exceptionThrown":
            d = r["params"]["exceptionDetails"]; txt = d.get("exception", {}).get("description", d.get("text"))
            print("exception:", txt); errors.append(txt)
        elif m == "Log.entryAdded" and r["params"]["entry"]["level"] == "error":
            e = r["params"]["entry"]; print(f"log.error: {e['text']} {e.get('url', '')}"); errors.append(e["text"])
    def wait(i):
        while True:
            r = json.loads(ws.recv()); handle(r)
            if r.get("id") == i: return r
    def ev(expr):
        r = wait(send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True))
        res = r.get("result", {})
        if "exceptionDetails" in res: print("eval error:", res["exceptionDetails"].get("text"), expr[:80])
        return res.get("result", {}).get("value")
    wait(send("Runtime.enable")); wait(send("Log.enable")); wait(send("Page.enable"))
    wait(send("Emulation.setDeviceMetricsOverride", width=size[0], height=size[1], deviceScaleFactor=scale, mobile=False))
    wait(send("Page.navigate", url=url))
    time.sleep(1.0)
    for st in steps:
        if "js" in st: ev(st["js"])
        if "sleep" in st: time.sleep(st["sleep"])
        if "wait" in st:
            t0 = time.time()
            while not ev(st["wait"]):
                if time.time() - t0 > st.get("timeout", 120): print("TIMEOUT", st["wait"][:80]); break
                time.sleep(0.3)
        if "print" in st: print(ev(st["print"]))
        if "shot" in st:
            params = {"format": "png"}
            if st.get("selector"):
                r = ev(f"(()=>{{const e=document.querySelector({json.dumps(st['selector'])});e.scrollIntoView();const b=e.getBoundingClientRect();return [b.x+window.scrollX,b.y+window.scrollY,b.width,b.height]}})()")
                pad = st.get("pad", 4)
                params["clip"] = {"x": max(0, r[0] - pad), "y": max(0, r[1] - pad), "width": r[2] + 2 * pad, "height": r[3] + 2 * pad, "scale": 1}
            r = wait(send("Page.captureScreenshot", **params))
            os.makedirs(os.path.dirname(os.path.abspath(st["shot"])), exist_ok=True)
            open(st["shot"], "wb").write(base64.b64decode(r["result"]["data"]))
            print("wrote", st["shot"])
    # drain pending events
    ws.settimeout(0.5)
    try:
        while True: handle(json.loads(ws.recv()))
    except Exception: pass
finally:
    p.kill()
print(f"{len(errors)} console errors")
sys.exit(1 if errors else 0)
