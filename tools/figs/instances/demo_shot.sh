#!/bin/bash
# Screenshots of demos in a preset state, at device scale 2, for the slide figures of the
# modern deep learning day (instances/, detection/, attention/, diffusion/).
# usage: demo_shot.sh <outdir> <demo> <preset> <name> [height]
# Serves a temporary folder (this harness + a link to demos/) on port 9123.
set -e
OUT=$1; D=$2; P=$3; NAME=$4; H=${5:-1000}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
WWW=$OUT/www; mkdir -p "$WWW"; ln -sfn "$ROOT/demos" "$WWW/demos"; cp "$(dirname "$0")/demo_shot.html" "$WWW/h.html"
python3 -m http.server --directory "$WWW" 9123 >/dev/null 2>&1 & PID=$!
sleep 1
timeout 60 google-chrome --headless=new --no-sandbox --disable-gpu --user-data-dir="$OUT/chrome" --window-size=1220,$H \
  --timeout=8000 --hide-scrollbars --force-device-scale-factor=2 --screenshot="$OUT/$NAME.png" "http://localhost:9123/h.html?d=$D&p=$P" 2>/dev/null || true
kill $PID
