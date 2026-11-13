#!/bin/bash
# Static fallback figure of the cellcompare demo (cellnets-compare/fig-demo.png): the "dense"
# preset after running StarDist, Cellpose and SAM automatic in headless Chrome (needs
# `make wasm`, `make models` and python's websocket-client). Serves a temporary folder (the
# instances/ screenshot harness + a link to demos/) on port $PORT (default 9637).
# usage: tools/figs/cellnets-compare/demo_shot.sh <tmpdir>
set -e
OUT=$1; ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
WWW=$OUT/www; mkdir -p "$WWW"; ln -sfn "$ROOT/demos" "$WWW/demos"; cp "$ROOT/tools/figs/instances/demo_shot.html" "$WWW/h.html"
python3 -m http.server --directory "$WWW" ${PORT:-9637} >/dev/null 2>&1 & PID=$!
trap 'kill $PID' EXIT
sleep 1
CLICK='(t)=>[...document.querySelectorAll("button")].find(b=>b.textContent.startsWith(t)).click()'
IDLE='![...document.querySelectorAll("button")].some(b=>b.disabled)'
python3 -I "$ROOT/tools/figs/cellnets-compare/cdp_shot.py" "http://localhost:${PORT:-9637}/h.html?d=cellcompare&p=dense" "$OUT/cc_dense.png" 1220 860 \
  --tmp "$OUT" --timeout 400 --scale 2 --js "SLEEP:3" \
  --js "($CLICK)('Run StarDist')" --js "SLEEP:1" --js "WAIT:$IDLE" \
  --js "($CLICK)('Run Cellpose')" --js "SLEEP:1" --js "WAIT:$IDLE" \
  --js "($CLICK)('Run SAM')" --js "SLEEP:1" --js "WAIT:$IDLE"
python3 -I -c "
from PIL import Image; im = Image.open('$OUT/cc_dense.png'); im.crop((0, 0, 2440, 1720)).convert('RGB').resize((1464, 1032), Image.LANCZOS).save('$ROOT/cellnets-compare/fig-demo.png')"
