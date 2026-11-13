# Patched copies of two crates (for wasm32)

The `sam` crate builds SAM 2.1 from the henriksson-lab crate **sam2-rs** on **Candle**. Two
small problems stop the published sources from working on `wasm32-unknown-unknown`, so the
crate depends on these copies instead (path dependencies in `../Cargo.toml`). The upstream
repositories are not modified.

| Copy | Source | Patch |
|---|---|---|
| `candle-nn/` | github.com/mahogny/candle, rev `0998c8c655ae84330149ddbff509916254b887a1` (the revision sam2-rs pins), `candle-nn/src/` and the licences | `src/attention/cpu_flash/standard.rs`: `std::is_x86_feature_detected!` does not exist on wasm32, even inside `cfg!(target_arch = "x86_64") && …`, so the crate does not compile. A local macro `x86_feature!` calls it on x86_64 and is `false` elsewhere. Manifest rewritten without the workspace inheritance (same dependencies and versions). |
| `sam2-rs/` | github.com/henriksson-lab/sam2-rs, rev `edb4cb76cfd85bdc69915893a9addea7ebe3bd62` (the commit of the converter in `tools/models/manifest.json`), `src/` without `src/bin/`, and `manifests/` | `std::time::Instant::now()` panics on wasm32 ("time not implemented"), and the mask decoder and prompt encoder call it unconditionally (for optional profiling). Every use now goes through `src/clock.rs`: std's `Instant` natively, a stand-in with zero elapsed time on wasm32. Mechanical replacement in 10 files. Manifest without the CUDA build script and the optional features (download, image/video decoding, CUDA, MKL). |

To update: copy the same directories from the new revisions and repeat the two patches
(`grep -rn "is_x86_feature_detected" candle-nn/src`, `grep -rn "Instant" sam2-rs/src`), then
`make wasm` and the check in `demos/test/sam-wasm-test.ts`. Better: get both fixes upstream
and drop these copies.

Natively the patched crates behave exactly as the originals: the wrapper's output equals
sam2-rs's `Sam2ImagePredictor` to the last bit (max |logit difference| 0, see `../README.md`).
