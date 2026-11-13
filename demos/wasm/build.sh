#!/bin/bash
# Build the Rust -> WebAssembly crates: demos/wasm/<name>/ (Cargo crate, cdylib,
# wasm-bindgen) -> demos/dist/wasm/<name>/{<name>.js, <name>.d.ts, <name>_bg.wasm}.
# With no arguments, builds every crate (every demos/wasm/*/Cargo.toml).
#   make wasm                  # all crates
#   demos/wasm/build.sh hello  # one crate
# Also bundles an optional browser example, demos/wasm/<name>/example/{index.html,main.ts},
# into demos/dist/wasm/<name>/example/ (a test page, not linked from the course).
# Writes demos/dist/wasm/index.json (bytes and sha256 per .wasm: the cache key of lib/wasm.ts)
# and demos/dist/wasm/package.json ({"type":"module"}, so node can import the glue in tests).
# demos/dist/wasm/ is git-ignored: CI builds it for the deployed site.
# Toolchain (once): rustup target add wasm32-unknown-unknown
#                   cargo install wasm-bindgen-cli --version <the crates' pinned version> --locked
set -e
cd "$(dirname "$0")/../.."
export CARGO_TARGET_DIR="$PWD/demos/wasm/target"
OUT=demos/dist/wasm
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  for d in demos/wasm/*/Cargo.toml; do [ -e "$d" ] && names+=("$(basename "$(dirname "$d")")"); done
fi
[ ${#names[@]} -eq 0 ] && { echo "wasm: no crates in demos/wasm/"; exit 0; }

command -v cargo >/dev/null || { echo "wasm: cargo not found (install Rust: https://rustup.rs)"; exit 1; }
command -v wasm-bindgen >/dev/null || {
  echo "wasm: wasm-bindgen not found: cargo install wasm-bindgen-cli --version <version> --locked"; exit 1; }
cli=$(wasm-bindgen --version | awk '{print $2}')
mkdir -p "$OUT"

for n in "${names[@]}"; do
  dir=demos/wasm/$n
  [ -f "$dir/Cargo.toml" ] || { echo "wasm: $dir/Cargo.toml missing"; exit 1; }
  # The crate's wasm-bindgen must equal the CLI version exactly, or the glue is wrong.
  want=$(grep -E '^wasm-bindgen *=' "$dir/Cargo.toml" | head -1 | sed -E 's/.*"=?([0-9.]+)".*/\1/')
  if [ -n "$want" ] && [ "$want" != "$cli" ]; then
    echo "wasm: $n pins wasm-bindgen $want but the CLI is $cli:"
    echo "      cargo install wasm-bindgen-cli --version $want --locked"; exit 1
  fi
  cargo build --quiet --release --target wasm32-unknown-unknown --manifest-path "$dir/Cargo.toml"
  lib=$(echo "$n" | tr '-' '_')
  wasm-bindgen --target web --out-dir "$OUT/$n" --out-name "$n" \
    "$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/$lib.wasm"
  # Optional size pass when binaryen is installed (not required).
  if command -v wasm-opt >/dev/null; then
    wasm-opt -Os --enable-bulk-memory --enable-nontrapping-float-to-int \
      "$OUT/$n/${n}_bg.wasm" -o "$OUT/$n/${n}_bg.wasm"
  fi
  # Browser example page (optional).
  if [ -f "$dir/example/main.ts" ]; then
    mkdir -p "$OUT/$n/example"
    node_modules/.bin/tsc --noEmit --strict --target es2020 --lib es2020,dom --module esnext \
      --moduleResolution bundler "$dir/example/main.ts"
    node_modules/.bin/esbuild "$dir/example/main.ts" --bundle --format=iife --target=es2020 \
      --minify --log-level=warning --loader:.wasm=binary --define:import.meta.url=undefined \
      --outfile="$OUT/$n/example/main.js"
    [ -f "$dir/example/index.html" ] && cp "$dir/example/index.html" "$OUT/$n/example/"
  fi
  echo "wasm: $n -> $OUT/$n/ ($(stat -c %s "$OUT/$n/${n}_bg.wasm") bytes)"
done

echo '{"type": "module"}' > "$OUT/package.json"
# index.json over every built module (also ones not rebuilt this time).
python3 - "$OUT" <<'EOF'
import hashlib, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
index = {}
for w in sorted(out.glob("*/*_bg.wasm")):
    b = w.read_bytes()
    index[w.parent.name] = {"file": f"{w.parent.name}/{w.name}", "bytes": len(b),
                            "sha256": hashlib.sha256(b).hexdigest()}
(out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
EOF
