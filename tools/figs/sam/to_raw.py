"""Convert the sample micrographs used by sam/'s Explore solutions to raw arrays for
measure_samnet.ts (node has no PNG decoder for 16-bit labels).
Run: python3 -I tools/figs/sam/to_raw.py <outdir>   (after `make models`)
Writes <id>.rgb (uint8 RGB, row-major), <id>.lab (int32 labels, if any), <id>.json (w, h)."""
import json, pathlib, sys
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[3]
IMG = ROOT / "demos/dist/models/images"
SAMPLES = {"nuclei": "stardist-img2d", "u2os": "bbbc039-A02_s1", "cells": "cellpose-img02",
           "histology": "dsb-histology", "brightfield": "dsb-brightfield"}
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for sid, stem in SAMPLES.items():
    im = Image.open(IMG / f"{stem}.png").convert("RGB")
    a = np.asarray(im, dtype=np.uint8)
    h, w = a.shape[:2]
    (out / f"{sid}.rgb").write_bytes(a.tobytes())
    meta = {"w": w, "h": h, "file": f"images/{stem}.png", "labels": False}
    lab = IMG / f"{stem}-labels.png"
    if lab.exists():
        L = np.asarray(Image.open(lab), dtype=np.int32)
        (out / f"{sid}.lab").write_bytes(L.astype("<i4").tobytes())
        meta["labels"] = True
    (out / f"{sid}.json").write_text(json.dumps(meta))
    print(sid, w, h, meta["labels"])
