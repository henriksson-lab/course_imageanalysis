#!/bin/bash
# Serve a temporary folder (demo_shot.html + a link to the repository's demos/) on port 9124
# and run cdp_shot.py with a steps file. Needs `make wasm` and `make models` for cellposenet.
# usage: demo_shot.sh <workdir> <query, e.g. d=flowtrack&preset=bend> <steps.json> [--size WxH]
set -e
WORK=$1; Q=$2; STEPS=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
WWW=$WORK/www; mkdir -p "$WWW"; ln -sfn "$ROOT/demos" "$WWW/demos"; cp "$HERE/demo_shot.html" "$WWW/h.html"
python3 -m http.server --directory "$WWW" 9124 >/dev/null 2>&1 & PID=$!
trap 'kill $PID' EXIT
sleep 1
python3 -I "$HERE/cdp_shot.py" "http://localhost:9124/h.html?$Q" "$STEPS" "$@"
