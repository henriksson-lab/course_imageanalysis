# Patched copy of sam2-rs (for wasm32)

The `sam` crate builds SAM 2.1 from the henriksson-lab crate **sam2-rs** on **Candle**. The
published sam2-rs does not run on `wasm32-unknown-unknown`, so the crate depends on this copy
instead (a path dependency in `../Cargo.toml`). The upstream repository is not modified.

| Copy | Source | Patch |
|---|---|---|
| `sam2-rs/` | github.com/henriksson-lab/sam2-rs, rev `edb4cb76cfd85bdc69915893a9addea7ebe3bd62` (the commit of the converter in `tools/models/manifest.json`, and still the latest `main` on 2026-11-13), `src/` without `src/bin/`, and `manifests/` | `std::time::Instant::now()` panics on wasm32 ("time not implemented"), and the mask decoder and prompt encoder call it unconditionally (for optional profiling). Every use now goes through `src/clock.rs`: std's `Instant` natively, a stand-in with zero elapsed time on wasm32. Mechanical replacement in 10 files. Manifest without the CUDA build script and the optional features (download, image/video decoding, CUDA, MKL), and with Candle at rev `2512982e1f879939d25b60797913761876c49465` (the revision `../Cargo.toml` uses) instead of upstream's `0998c8c`, which does not compile for wasm32. |

Candle itself is no longer vendored: until rev `0998c8c` its `candle-nn` called
`std::is_x86_feature_detected!` inside `cfg!(target_arch = "x86_64") && …`, which does not compile
on wasm32. The fork's rev `2512982` (branch `integration/stardist-clearmap-ops`, "Guard CPU
attention AVX detection on x86-64") puts it behind `#[cfg]`, so Candle is used as published.

To update: copy the same directories from the new sam2-rs revision and repeat the patch
(`grep -rn "Instant" sam2-rs/src`), then `make wasm` and the check in
`demos/test/sam-wasm-test.ts`. Better: get the fix upstream and drop this copy.

Natively the patched crate behaves exactly as the original: the wrapper's output equals
sam2-rs's `Sam2ImagePredictor` to the last bit (max |logit difference| 0, see `../README.md`).
