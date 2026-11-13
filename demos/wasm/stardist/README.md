# demos/wasm/stardist: StarDist 2D in the browser

Wasm module for the `stardistnet` demo (`demos/src/stardistnet/`). It wraps
[henriksson-lab/stardist-rs](https://github.com/henriksson-lab/stardist-rs) (pinned rev
`20cb237`, features `candle` only) and runs the pretrained **2D_versatile_fluo** model on the CPU.
Built by `make wasm` into `demos/dist/wasm/stardist/`; weights come from `make models`
(`demos/dist/models/stardist/2D_versatile_fluo/`).

## What runs in Wasm

All of it: input normalisation, the U-Net, and the decoding.

| Step | Code |
|---|---|
| Weights: `weights_best.h5` read from bytes | `src/h5.rs` (small HDF5 reader). The crate's loader needs a file path, which the browser does not have. Output is identical to the crate's loader (`tests/native.rs`). |
| Percentile normalisation (1 / 99.8 %) | `normalize_percentile` |
| Padding to a multiple of 16 and cropping back | crate: `StarDist2D::_predict_setup`, `StarDistPadAndCropResizer::after` |
| Network → probability (grid 2) + 32 ray distances | two interchangeable engines: **candle** (the crate's `model::candle::StarDist2D`, as published) and **fast** (`src/fast.rs`: the same layers as direct convolutions with Wasm SIMD, the demo's default) |
| Threshold, polygon NMS (bbox pre-check, kd-tree) | crate: `nms::non_maximum_suppression` |
| Polygons → label image, polygon vertices | crate: `geometry::polygons_to_label`, `dist_to_coord` |

Interface (free functions, module state; no exported classes so the generated `.d.ts` typechecks
under `lib: es2020`): `load(config, thresholds, h5)`, `predict(img, w, h, engine)`,
`out_width/out_height/prob/dist`, `instances(prob, dist, …, prob_thresh, nms_thresh)` plus the
`dec_*` getters (candidates, count, points, probs, dists, coords, labels), `normalize_percentile`.

Two patches in this wrapper, none upstream: `getrandom` with `wasm_js` (candle pulls in `rand`;
still needed at the candle revision below) and the candle fork that stardist-rs builds against
(`[patch.crates-io]`, because a dependency's own `[patch]` does not apply here), pinned to rev
`2512982` of mahogny/candle (branch `integration/stardist-clearmap-ops`, 2026-11-13), the same
candle as `demos/wasm/sam`, instead of stardist-rs's branch `pr/stardist-candle-integration`
(`c4e7b3e`), which it contains.

## Checks

- `cargo test --release -- --nocapture` (native, needs `make models`): `h5.rs` gives the same
  38 tensors as the crate's HDF5 loader, and `fast` matches `candle` (prob max |diff| 1e-6, dist
  max relative diff 1e-5). It writes `testdata/native-ref.json`: the native outputs on a
  128 × 128 crop of `images/stardist-img2d.png`.
- A NumPy forward pass written separately from the Keras layer order (in the session scratchpad, not in the repo)
  agrees with the native output on that crop: prob max |diff| 2e-6, dist 2e-4 px.
- `demos/test/stardist-wasm-test.ts` (node): the decoding on synthetic discs with known answers, and both Wasm
  engines against `native-ref.json`. The probability map and rays agree within 1e-4, and the
  candidates, centres and label areas are exactly equal.

## Benchmark (2026-11-13, Xeon Gold 6138 at 2.0 GHz, under load from other jobs; one thread)

Sizes: `stardist_bg.wasm` 1.22 MB (274 kB gzip; 1.06 MB and 257 kB with candle `c4e7b3e`), `weights_best.h5` 5.5 MB (f32, barely
compresses), demo script 35 kB.

512 × 512 sample image (`images/stardist-img2d.png`): 121 objects from 4737 candidate cells.
The labels and counts are the same in every column.

| | network | decoding (NMS + labels) |
|---|---:|---:|
| headless Chrome, fast engine (default) | 1.96–2.04 s | 0.12–0.13 s |
| headless Chrome, candle engine | 5.2–5.5 s | 0.12 s |
| node 20, fast / candle | 1.8–2.0 s / 4.1 s (4.9–5.3 s with candle `c4e7b3e`) | 0.11 s |
| native x86-64, candle, 1 thread / 40 threads | 2.7 s / 1.0 s | 0.06 s |

Candle rev `2512982` instead of `c4e7b3e` (2026-11-13): candle's matrix products come from the
mahogny/gemm fork instead of crates.io `gemm` 0.19, which makes the candle engine about 18 % faster
in node (4.1 vs 5.0 s) and the module 157 kB larger; the fast engine and the decoding are
unchanged. Output is bit-identical (prob, dist and labels of both engines: max |difference| 0).
The Chrome rows were measured with `c4e7b3e`.

696 × 520 (`dsb-fluo-dense`): 2.6 s network, 0.22 s decoding, 305 objects.

Loading in Chrome (module + config + weights from a local server, then building the network):
0.1–0.8 s. From navigation to the first labels: 2.4–3.2 s. On the deployed site the 6.5 MB
download adds to this the first time; after that the Cache API serves it.

Why there is a second engine: candle's CPU matrix kernels have no Wasm SIMD unless the whole
crate graph is built with `-C target-feature=+simd128`. `demos/wasm/build.sh` does not pass
that flag, and this candle fork does not compile with it anyway (`CurrentCpuF16` is missing for
simd128). `fast.rs` turns SIMD on per function (`#[target_feature(enable = "simd128")]`), which
needs no build flag. It uses a 4-pixel × 8-channel register block with named accumulators. A
generic version with accumulator arrays was 1.7–4× slower, because V8 spills the arrays to memory.

Ideas not done: f16 weights (−2.7 MB), several workers on tiles (StarDist tiles with overlap),
relaxed-SIMD FMA (Chrome only).

If `wasm-opt` is installed, `demos/wasm/build.sh` runs it without `--enable-simd`, and that
would reject this module. It is not installed here or in CI.
