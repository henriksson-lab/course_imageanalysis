"""Measured numbers of the yolonet demo for the yolo/ exercise (Explore solutions).
Needs `make wasm`, `make models` and the built demo (demos/build.sh yolonet). Bundles
measure.ts, runs it in headless Chrome (cdp.py) and writes measured_yolonet.json next to this file.
Run from the repository root: python3 -I tools/figs/yolo/measure_yolonet.py"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp import Browser, ROOT, HERE

js = os.path.join(ROOT, "demos", "dist", "_yolomeasure.js")
subprocess.run([os.path.join(ROOT, "node_modules/.bin/esbuild"), os.path.join(HERE, "measure.ts"), "--bundle", "--format=iife",
                "--target=es2020", "--log-level=warning", f"--outfile={js}"], check=True)
b = Browser(scale=1)
try:
    b.call("Page.navigate", url=f"http://localhost:{b.port}/h.html?d=_yolomeasure")
    b.until("!!window.__out || !!window.__err", 900)
    err = b.ev("window.__err || ''")
    if err:
        raise SystemExit("error: " + err)
    out = b.ev("window.__out")
    json.dump(out, open(os.path.join(HERE, "measured_yolonet.json"), "w"), indent=1)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, "cands", v["cands"], v["perStride"], "kept", v["kept"], "max", v["maxScore"], v["maxCls"], "net", v["netMs"], "ms")
            for e in v["list"][:12]:
                print("   ", e)
        else:
            print(k, v)
    print("errors", b.errors)
finally:
    b.close()
    os.remove(js)
