"""Write the course micrograph (demos/src/fourier2d/cell256.png, 256 × 256 grey) as a JSON list of
grey values 0..255 for data.ts (node has no PNG decoder). Usage: python3 -I tools/figs/geodesic/cell.py <out.json>
Image: transmission electron micrograph of a mesenchymal stem cell, Robert M. Hunt, CC BY 3.0, via Wikimedia Commons."""
import os, sys, json
import numpy as np
from PIL import Image

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
a = np.asarray(Image.open(os.path.join(REPO, "demos", "src", "fourier2d", "cell256.png")).convert("L"), dtype=float)
json.dump([float(v) for v in a.ravel()], open(sys.argv[1], "w"))
