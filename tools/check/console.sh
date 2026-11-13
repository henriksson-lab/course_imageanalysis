#!/bin/bash
# Browser console check of the rendered site: serves docs/ on a free local port and opens every
# page (or the pages given as arguments, relative to docs/, e.g. canny/lec.html) in headless
# Chrome, reporting uncaught exceptions, console.error and failed loads (tools/check/console.py).
# Model demos need demos/dist/{wasm,models} in docs/ (make wasm models, then render or copy);
# without them they show their loading message, which is not an error.
# Environment: JOBS (parallel browsers, default 6), WAIT (seconds per page after load, default 6).
# Usage: tools/check/console.sh [PAGE ...]      (make check runs it; exit 1 on errors)
set -u
cd "$(dirname "$0")/../.."
[ -d docs ] || { echo "docs/ does not exist: render the site first (make render)"; exit 1; }
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
python3 -m http.server "$port" --bind 127.0.0.1 --directory docs >/dev/null 2>&1 &
server=$!
trap 'kill $server 2>/dev/null' EXIT
for _ in $(seq 50); do curl -s -o /dev/null "http://127.0.0.1:$port/" && break; sleep 0.2; done
list=$(mktemp)
if [ $# -gt 0 ]; then
  printf '%s\n' "$@" > "$list"
else
  (cd docs && find . -name '*.html' -not -path './site_libs/*' -not -path './demos/*' | sed 's|^\./||' | sort) > "$list"
fi
python3 tools/check/console.py "http://127.0.0.1:$port" --list "$list" --jobs "${JOBS:-6}" --wait "${WAIT:-6}"
status=$?
rm -f "$list"
exit $status
