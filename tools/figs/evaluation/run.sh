#!/bin/sh
# Bundle a figure script with esbuild and run it with node. Usage: tools/figs/evaluation/run.sh tools/figs/<unit>/figs.ts
set -e
cd "$(dirname "$0")/../../.."
out="${FIGTMP:-${TMPDIR:-/tmp}}/fig-$(basename "$(dirname "$1")").cjs"
node_modules/.bin/esbuild "$1" --bundle --platform=node --format=cjs --log-level=warning --outfile="$out"
node "$out"
