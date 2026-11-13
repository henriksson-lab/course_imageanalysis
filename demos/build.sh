#!/bin/bash
# Build interactive demos: demos/src/<name>/main.ts -> demos/dist/<name>.js (IIFE,
# so it also works from file://). With no arguments, builds every demo.
# Images are inlined as data URLs: canvas pixels of a file:// image are unreadable.
# demos/src/lib/ is shared code, imported by the demos, not a demo itself.
# WebAssembly (demos/README.md): a demo imports the wasm-bindgen glue of a crate built by
# `make wasm` (demos/dist/wasm/<crate>/<crate>.js); a `.wasm` import is inlined as bytes
# (binary loader). Crates a demo imports are built first when their output is missing or
# older than the crate's sources.
set -e
cd "$(dirname "$0")/.."
mkdir -p demos/dist
cp demos/demo.css demos/dist/demo.css
names=("$@")
[ ${#names[@]} -eq 0 ] && names=($(ls demos/src | grep -v '^lib$'))

# Wasm crates used by these demos (imports of dist/wasm/<crate>/ in their sources).
crates=()
for n in "${names[@]}"; do
  for c in $(grep -rhoE 'dist/wasm/[A-Za-z0-9_-]+/' "demos/src/$n" 2>/dev/null | cut -d/ -f3 | sort -u); do
    out="demos/dist/wasm/$c/${c}_bg.wasm"
    if [ ! -f "$out" ] || [ -n "$(find "demos/wasm/$c/src" "demos/wasm/$c/Cargo.toml" -newer "$out" 2>/dev/null | head -1)" ]; then
      crates+=("$c")
    fi
  done
done
[ ${#crates[@]} -gt 0 ] && demos/wasm/build.sh $(printf '%s\n' "${crates[@]}" | sort -u)

for n in "${names[@]}"; do
  node_modules/.bin/esbuild "demos/src/$n/main.ts" --bundle --format=iife --target=es2020 \
    --minify --log-level=warning --outfile="demos/dist/$n.js" \
    --loader:.jpg=dataurl --loader:.png=dataurl \
    --loader:.wasm=binary --define:import.meta.url=undefined
done
echo "built: ${names[*]}"
