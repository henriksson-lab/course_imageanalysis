"""Demo screenshots for the yolo/ slides and exercise (real network: yolonet; analytic: yolodecode).
Needs `make wasm`, `make models` and the built demos (demos/build.sh yolonet yolodecode).
Run from the repository root: python3 -I tools/figs/yolo/demo_shots.py  → yolo/fig-*.png"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdp import Browser, ROOT

OUT = os.environ.get("YOLO_FIGS") or os.path.join(ROOT, "yolo")
RO = "(document.querySelector('.qb-readout')||{}).textContent||''"
DONE = f"({RO}).includes('decoding') && !document.querySelector('.yn-busy').textContent"

b = Browser(scale=2)
try:
    # real network on the street photo: kept boxes coloured by stride, score maps per stride
    b.open("yolonet", "default")
    b.until(DONE, 300); time.sleep(0.8)
    b.shot(os.path.join(OUT, "fig-strides.png"), ".yn-left")
    b.shot(os.path.join(OUT, "fig-yolonet.png"))
    b.click_preset("photo: raw candidates"); time.sleep(0.8); b.until(DONE); time.sleep(0.5)
    b.shot(os.path.join(OUT, "fig-raw.png"), ".yn-stage")
    b.click_preset("nuclei: threshold 0.01")
    b.until(DONE + " && document.querySelector('.yn-credit').textContent.includes('StarDist')", 300); time.sleep(0.8)
    b.shot(os.path.join(OUT, "fig-nuclei.png"), ".yn-left")
    # the analytic demo
    for p, name in [("b1", "fig-yolodecode.png")]:
        b.open("yolodecode", p); time.sleep(0.6)
        b.shot(os.path.join(OUT, name))
    print("console errors:", b.errors)
finally:
    b.close()

# keep the repository small: 256-colour PNGs, the wide demo views at 3/4 size
from PIL import Image
for f in ["fig-yolonet.png", "fig-strides.png", "fig-raw.png", "fig-nuclei.png", "fig-yolodecode.png"]:
    p = os.path.join(OUT, f)
    im = Image.open(p).convert("RGB")
    if f in ("fig-yolonet.png", "fig-yolodecode.png"):
        im = im.resize((im.width * 3 // 4, im.height * 3 // 4), Image.LANCZOS)
    im.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(p, optimize=True)
