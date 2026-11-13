# yolo: YOLO11 nano in the browser (Rust -> WebAssembly)

Wrapper crate for the `yolonet` demo (`demos/src/yolonet/`). Built by `make wasm` into
`demos/dist/wasm/yolo/` (`yolo.js` glue, `yolo_bg.wasm`); the module is fetched (not inlined)
with `loadWasm(glue, "yolo")`, the weights with `loadModelFile("yolo/yolo11n-coco.safetensors")`
(`tools/models/manifest.json`, entry `yolo11n-coco`, made by `make models`).

## What runs in Wasm

The **whole network**: YOLO11n (backbone, feature pyramid, decoupled head; 2.6 M parameters,
COCO-trained, 80 classes) on Burn's portable CPU backend `burn-flex`, single-threaded, plus the
decoding of the head output into boxes and greedy NMS. No precomputed outputs.

API (flat arrays; layouts in `src/lib.rs`):

| function | in | out |
|---|---|---|
| `load(weights)` | safetensors bytes (f16 or f32) | load report ("417 tensors loaded, 0 missing, ...") |
| `run(input, size)` | 3 x size x size RGB 0..1 (CHW), size a multiple of 32 | raw head output, A x 144: 4 sides x 16 distance-bin logits, 80 class logits (A = anchor points of the three scales, stride 8, 16, 32) |
| `decode(raw, size)` | raw | A x 88: x1 y1 x2 y2 (input px), distances l t r b (grid cells), 80 probabilities |
| `nms(decoded, conf, iou, max_det, class_agnostic)` | decoded | rows x1 y1 x2 y2 score class anchor |
| `anchors(size)` | | A |

The demo calls `run` + `decode` in a Web Worker and does thresholding + NMS in the page with the
TypeScript reference (`demos/src/yolonet/post.ts`, same semantics), so sliders are live.

## What is vendored from YOLOv11-rs, and why

`src/model/{anchors,attention,backbone,blocks,conv,head,model,neck,nms}.rs` are copied unchanged
from [henriksson-lab/YOLOv11-rs](https://github.com/henriksson-lab/YOLOv11-rs) at commit
`041c9f0277692f32791688b76fb8926cd21697ca` (the commit `convert_weights.py` is pinned to in the
manifest), except `std::time::Instant` -> `crate::model::Instant` (the crate's optional
profiling timers; `Instant::now()` panics on wasm32, so `model/mod.rs` gives a no-op there).
The crate itself is not a dependency because its `Cargo.toml` hard-wires the `wgpu` backend,
`rayon`, `clap`, `indicatif`, `image`, training code and path patches to sibling checkouts of
Burn and CubeCL (`../burn`, `../cubecl-security`), none of which build for wasm32 or in CI.
Instead this crate depends on the same Burn revision the crate uses (`github.com/mahogny/burn`
at `e20ada6837...`, see YOLOv11-rs `DEPENDENCY_SECURITY.md`) with only `std`, `flex`, `simd`
and `burn-store/safetensors`. The upstream repository is not modified.

Glue added here (`src/lib.rs`): f16 -> f32 widening of the checkpoint (burn-flex keeps the
stored dtype, and mixed f16/f32 ops fail); key remapping of the converted checkpoint
(`head.{box,cls}_branches.i.cK` -> `head.{box,cls}_cK.i`, BatchNorm `weight/bias` -> `gamma/beta`
via burn-store's PyTorch adapter); BatchNorm folded into the convolutions (`YOLO::fuse`);
decoding and NMS over flat arrays; getrandom's JS backend for wasm32 (pulled in by Burn's
`rand`, never used at inference).

## Correctness

- `examples/reference.rs` runs the same code natively (`cargo run --release --example reference
  -- <weights> <input.f32> <size> <outdir> [conf]`); on the bus photo letterboxed to 512 x 512:
  the crate's own inference decode vs the PyTorch original (YOLOv11-pt `v11_n.pt`): max
  difference 1.5e-3 px (boxes), 1.2e-6 (probabilities); `decode_raw` vs the crate's decode:
  1.2e-4 px, 3e-8; the crate's NMS: bus 0.94, person 0.90, 0.83, 0.81, 0.31.
- Wasm vs native raw output on the same input: max difference 7.5e-5; Wasm NMS = crate NMS.
- `demos/test/yolo-wasm-test.ts`: TypeScript decoding/NMS on hand-made cases; Wasm `decode`/`nms`
  vs TypeScript on random head outputs; the full network in Wasm on `stardist-img2d.png` vs the
  native crate's NMS output (10 boxes at score > 0.01; none at > 0.25).

## Benchmark (2026-11-13, Intel Xeon Gold 6138 @ 2.0 GHz, headless Chrome, Web Worker)

| | |
|---|---|
| Wasm module | 2.53 MB (0.64 MB gzip), `opt-level = "s"`, no wasm-opt |
| weights | 5.32 MB safetensors f16 (f32 in memory after loading) |
| load, first visit (local server) | 2.5 s: fetch + compile 1.8 s, weights -> model 0.7 s |
| load, cached (Cache API) | 1.2-1.7 s (model build 0.6 s) |
| run, 512 x 512 input | 0.68-0.80 s network + 16-21 ms decode (worker); threshold + NMS 2-14 ms in the page |
| run, 256 x 256 input | 0.18-0.19 s |
| node (same module) | load 0.65 s, run 512 x 512 0.64-0.72 s |
| native, same code, 1 thread | run 512 x 512 0.28-0.33 s |

Options not taken: `-C target-feature=+simd128` (burn-flex/gemm SIMD128 paths) gave about 20 %
(0.57 s at 512 in node) but needs RUSTFLAGS in `demos/wasm/build.sh`; `opt-level = 3` gave
3.3 MB for the same speed. Threads (rayon + SharedArrayBuffer) would need cross-origin isolation,
which GitHub Pages does not offer.

## Limitations

- The checkpoint is the COCO-trained generic model: no cell or nucleus class. On micrographs it
  finds nothing at the usual threshold (best score on the StarDist nuclei image 0.045, "person"),
  and at very low thresholds it names nuclei "person", "sports ball" or the whole field "bus":
  the demo uses this as the out-of-domain example. A nuclei detector would need a fine-tuned
  checkpoint published somewhere CI can fetch it (the weights are not committed).
- CPU only, one thread; inputs up to 800 x 800 in the demo (about 2 s).
