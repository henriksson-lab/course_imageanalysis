"""Inputs for measure.ts (the Explore numbers of stardist/exercise.qmd, measured with the real
network): the stardistnet sample images as the demo sees them (grey = mean of R, G, B; scaled
copies with a smooth resize like the demo's canvas) and their ground-truth labels, as raw
little-endian arrays plus an index.json.
  python3 -I tools/figs/stardist/measure_prep.py OUTDIR   (after `make models`)
"""
import json, sys, os
import numpy as np
from PIL import Image

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
IMG = "demos/dist/models/images/"
SAMPLES = {"nuclei": "stardist-img2d", "sparse": "dsb-fluo-small", "dim": "dsb-fluo-medium", "dense": "dsb-fluo-dense",
           "u2os": "bbbc039-A02_s1", "histology": "dsb-histology", "brightfield": "dsb-brightfield"}
SCALES = {"nuclei": [0.5, 1, 2], "dense": [0.5, 1, 2], "u2os": [0.5, 1, 2]}
index = []
for sid, f in SAMPLES.items():
    im = Image.open(IMG + f + ".png").convert("RGB")
    a = np.asarray(im, dtype=np.float32).mean(axis=2)
    lab = None
    if os.path.exists(IMG + f + "-labels.png"):
        lab = np.asarray(Image.open(IMG + f + "-labels.png"), dtype=np.int32)
    for s in SCALES.get(sid, [1]):
        h, w = a.shape
        f = min(s, 1024 / max(w, h))   # the demo caps the network input at 1024 px per side
        if f != 1:
            W, H = max(16, round(w * f)), max(16, round(h * f))
            b = np.asarray(Image.fromarray(a).resize((W, H), Image.BILINEAR if s > 1 else Image.BOX), dtype=np.float32)
            b = np.clip(np.round(b), 0, 255).astype(np.float32)   # the demo reads the canvas back as bytes
        else:
            b = a
        name = f"{sid}@{s}"
        b.astype("<f4").tofile(f"{OUT}/{name}.f32")
        e = {"name": name, "id": sid, "scale": s, "factor": round(f, 4), "w": b.shape[1], "h": b.shape[0]}
        if lab is not None:
            lb = lab if f == 1 else np.asarray(Image.fromarray(lab).resize((b.shape[1], b.shape[0]), Image.NEAREST), dtype=np.int32)
            lb.astype("<i4").tofile(f"{OUT}/{name}.lab"); e["labels"] = int(lb.max())
        index.append(e)
    if sid == "nuclei":   # the same image turned by 90 degrees and mirrored (exercise E6)
        for tag, b in (("rot90", np.rot90(a)), ("mirror", a[:, ::-1])):
            b = np.ascontiguousarray(b); name = f"{sid}@{tag}"
            b.astype("<f4").tofile(f"{OUT}/{name}.f32")
            index.append({"name": name, "id": sid, "scale": 1, "w": b.shape[1], "h": b.shape[0]})
json.dump(index, open(f"{OUT}/index.json", "w"), indent=1)
print(len(index), "inputs")
