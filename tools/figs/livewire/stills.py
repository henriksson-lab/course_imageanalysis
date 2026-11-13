"""Demo stills for livewire/lec.qmd: screenshots of the livewire demo in a preset
state (headless Chrome, device scale 2), cropped to the image panel and put side by side.
Usage (repo root; docs/ rendered so docs/demos/dist exists):
   python3 -I tools/figs/livewire/stills.py <scratch dir> <port>
It writes small HTML pages into <scratch dir>/site (demos -> docs/demos), serves them on
<port>, and screenshots each preset."""
import os, sys, subprocess, time
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUTDIR = os.path.join(REPO, "livewire")
scratch, port = sys.argv[1], sys.argv[2]
site = os.path.join(scratch, "site"); os.makedirs(site, exist_ok=True)
if not os.path.exists(os.path.join(site, "demos")):
    os.symlink(os.path.join(REPO, "docs", "demos"), os.path.join(site, "demos"))

def page(name, demo, attrs):
    with open(os.path.join(site, name + ".html"), "w") as f:
        f.write(f'<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="demos/dist/demo.css">'
                f'<style>body{{background:#fff;margin:10px;font-size:17px;font-family:sans-serif}}</style></head><body>'
                f'<div data-demo="{demo}" {attrs}></div><script src="demos/dist/{demo}.js"></script></body></html>')

def shoot(name):
    out = os.path.join(scratch, name + ".png")
    subprocess.run(["timeout", "60", "google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu",
                    f"--user-data-dir={scratch}/chrome", "--window-size=1400,1000", "--force-device-scale-factor=2",
                    "--timeout=8000", f"--screenshot={out}", f"http://localhost:{port}/{name}.html"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Image.open(out).convert("RGB")

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 56)

def side_by_side(panels, labels, out, gap=24, top=80):
    w = sum(p.width for p in panels) + gap * (len(panels) - 1); h = max(p.height for p in panels) + top
    W = Image.new("RGB", (w, h), "white"); d = ImageDraw.Draw(W); x = 0
    for p, lab in zip(panels, labels):
        W.paste(p, (x, top)); tw = d.textlength(lab, font=FONT)
        d.text((x + (p.width - tw) / 2, 8), lab, fill="black", font=FONT); x += p.width + gap
    W.save(out)

if __name__ == "__main__":
    srv = subprocess.Popen([sys.executable, "-m", "http.server", port], cwd=site,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1)
        box = (20, 124, 980, 1084)  # the image panel (CSS px 10..490 at scale 2)
        shots = {}
        for name, attrs in (("lw-nucleus", 'data-preset="nucleus"'), ("lw-cost", 'data-preset="nucleus" data-view="cost"'),
                            ("lw-weak", 'data-preset="weak"'), ("lw-neighbour", 'data-preset="neighbour"'),
                            ("lw-dist", 'data-preset="weak" data-view="distance"')):
            page(name, "livewire", attrs)
            shots[name] = shoot(name).crop(box)
        side_by_side([shots["lw-nucleus"], shots["lw-cost"]], ["traced, 4 anchors", "cost (dark = cheap)"],
                     os.path.join(OUTDIR, "fig-lwcost.png"))
        side_by_side([shots["lw-weak"], shots["lw-neighbour"]], ["one seed: short-cut", "the neighbour's edge"],
                     os.path.join(OUTDIR, "fig-behaviour.png"))
        shots["lw-dist"].save(os.path.join(OUTDIR, "fig-distance.png"))
    finally:
        srv.terminate()
