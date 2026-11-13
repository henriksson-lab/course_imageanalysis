#!/bin/bash
# Demo still for score-matching/lec.qmd ("Where learned denoisers fail"): the scorematch1d demo
# in its "under" preset, at slide label size (host inside a .reveal div), scale factor 2,
# cropped to the plots. Run from the repository root after demos/build.sh scorematch1d.
# usage: tools/figs/score-matching/still.sh <scratch dir>
set -e
T=$1; W=$T/www; mkdir -p "$W"; ln -sfn "$PWD/demos" "$W/demos"
sed 's|<div id="host"></div>|<div class="reveal"><div id="host"></div></div>|' tools/figs/instances/demo_shot.html > "$W/r.html"
python3 -m http.server --directory "$W" 9418 >/dev/null 2>&1 & PID=$!
sleep 1
timeout 60 google-chrome --headless=new --no-sandbox --disable-gpu --user-data-dir="$T/chrome" --window-size=1220,900 \
  --timeout=9000 --hide-scrollbars --force-device-scale-factor=2 --screenshot="$T/under.png" "http://localhost:9418/r.html?d=scorematch1d&p=under" 2>/dev/null || true
kill $PID
python3 -I -c "
from PIL import Image; import sys
im = Image.open(sys.argv[1]).convert('RGB').crop((0, 30, 1405, 1015))
im.save('score-matching/fig-undertrained.png', optimize=True)" "$T/under.png"
