#!/bin/bash
# Screenshots of the Hough day's demos in fixed states (presets, plus slider tweaks via still.html),
# used as stills by tools/figs/{hough-lines,hough-circles,ght,ransac}/figs.py.
# Usage: capture.sh OUTDIR   (needs the built demos in docs/demos/dist and google-chrome)
set -e
OUT=$(realpath -m "$1")
PORT=$(python3 -I -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
TMP=$(mktemp -d); mkdir -p "$TMP/stills" "$OUT"
ln -s "$REPO/docs/demos" "$TMP/demos"; cp "$HERE/still.html" "$TMP/stills/"
(cd "$TMP" && exec python3 -m http.server "$PORT" >/dev/null 2>&1) & SRV=$!
trap 'kill $SRV 2>/dev/null; rm -rf "$TMP"' EXIT
sleep 1
shot() { timeout 60 google-chrome --headless=new --no-sandbox --disable-gpu --user-data-dir="$TMP/chrome" \
  --window-size=1280,1000 --timeout=6000 --screenshot="$OUT/$1.png" \
  "http://localhost:$PORT/stills/still.html?$2&z=$RANDOM" 2>/dev/null; }
for p in clean nonms fine coarse clutter grad segments deskew; do shot hl_$p "demo=houghlines&preset=$p"; done
shot hl_noise18 "demo=houghlines&s=noise=18"
for p in plate false ellipse; do shot hc_$p "demo=houghcircle&preset=$p"; done
for p in circle rotmiss occluded similar; do shot ght_$p "demo=ght&preset=$p"; done
for p in epsbig twolines; do shot rs_$p "demo=ransac&preset=$p"; done
ls "$OUT"
