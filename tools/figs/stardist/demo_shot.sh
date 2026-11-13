#!/bin/bash
# Screenshots of demo states for the stardist/ slide figures (device scale 2).
# usage: demo_shot.sh <outdir> <name> "<query>" [extra shot.py args...]
#   e.g. demo_shot.sh SHOTS sr_c "d=starrays&preset=cshape"
#        demo_shot.sh SHOTS sd_cand "d=stardistnet&image=nuclei&view=cand" --wait "..."
# Serves a temporary folder (demo.html + a link to demos/) on port $PORT (default 9677; set PORT to change). The real-model demo
# needs `make wasm` and `make models` first (demos/dist/wasm/, demos/dist/models/).
set -e
PORT=${PORT:-9677}
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then echo "port $PORT is in use" >&2; exit 1; fi
OUT=$1; NAME=$2; Q=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
WWW=$OUT/www; mkdir -p "$WWW"; ln -sfn "$ROOT/demos" "$WWW/demos"; cp "$HERE/demo.html" "$WWW/h.html"
python3 -m http.server --directory "$WWW" $PORT >/dev/null 2>&1 & PID=$!
trap 'kill $PID' EXIT
sleep 1
python3 -I "$HERE/shot.py" "http://localhost:$PORT/h.html?$Q" "$OUT/$NAME.png" "$@"
