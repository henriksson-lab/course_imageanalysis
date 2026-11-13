//! Cellpose 3 in the browser (demos/src/cellposenet/). Two parts, both plain numeric
//! functions over flat row-major arrays (no DOM):
//!
//! - `load_network`, `run_tile`: the U-net forward pass of the cyto3 / nuclei models on one
//!   tile (weights from a safetensors file, f16, bf16 or f32);
//! - `compute_masks` and friends: decoding the flows into masks (Euler tracking to sinks,
//!   seeds, flow-error check, hole filling), as in Cellpose 3.
//!
//! Tiling, normalisation and resizing are done in TypeScript (demos/src/cellposenet/pipeline.ts),
//! so several Web Workers can run tiles in parallel.
//!
//! cellpose-rs (github.com/henriksson-lab/cellpose-rs) implements Cellpose-SAM only (a 300 M
//! parameter ViT, 1.2 GB) and depends on candle, rayon, reqwest and file I/O; this crate is a
//! small re-implementation of the Cellpose 3 network plus the 2D decoding, checked against
//! Python Cellpose 3.1 and against cellpose-rs's `dynamics::compute_masks`
//! (demos/test/cellpose-wasm-test.ts, README.md).
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

pub mod conv;
pub mod dynamics;
pub mod net;
pub mod weights;

// Module state instead of exported classes (plain functions keep the generated TypeScript
// simple): the loaded network and the last decoding result.
thread_local! {
    static NET: RefCell<Option<net::Net>> = const { RefCell::new(None) };
    static LAST: RefCell<Option<dynamics::MaskResult>> = const { RefCell::new(None) };
}

/// Load the network from the bytes of a safetensors state dict (cellpose/resnet_torch.py
/// names; F16, BF16 or F32). Returns the number of parameters.
#[wasm_bindgen]
pub fn load_network(safetensors: &[u8]) -> Result<usize, JsError> {
    let w = weights::parse_safetensors(safetensors).map_err(|e| JsError::new(&e))?;
    let n = net::Net::load(&w).map_err(|e| JsError::new(&e))?;
    let p = n.n_params;
    NET.with(|c| *c.borrow_mut() = Some(n));
    Ok(p)
}

/// Run one tile through the loaded network. `input` is (2, h, w) (channel 1 = cytoplasm or
/// grey, channel 2 = nuclei or zeros), normalised so 1st percentile = 0 and 99th = 1; h, w
/// multiples of 8. Returns (3, h, w): flow dY, flow dX, cell probability (logit).
#[wasm_bindgen]
pub fn run_tile(input: &[f32], h: usize, w: usize) -> Result<Vec<f32>, JsError> {
    NET.with(|c| match c.borrow_mut().as_mut() {
        Some(n) => n.forward(input, h, w).map_err(|e| JsError::new(&e)),
        None => Err(JsError::new("no network loaded (call load_network first)")),
    })
}

/// Style vector (256 values, unit length) of the last tile.
#[wasm_bindgen]
pub fn last_style() -> Vec<f32> {
    NET.with(|c| c.borrow().as_ref().map(|n| n.style.clone()).unwrap_or_default())
}

fn last<T: Default>(f: impl FnOnce(&dynamics::MaskResult) -> T) -> T {
    LAST.with(|c| c.borrow().as_ref().map(f).unwrap_or_default())
}

/// Of the last `compute_masks`: final labels (0 = background, 1..count).
#[wasm_bindgen]
pub fn last_masks() -> Vec<u32> {
    last(|r| r.masks.clone())
}
/// Labels before the flow-error check and the size/hole clean-up.
#[wasm_bindgen]
pub fn last_masks_raw() -> Vec<u32> {
    last(|r| r.masks_raw.clone())
}
/// Flow error of each label of the raw masks (index label-1).
#[wasm_bindgen]
pub fn last_flow_errors() -> Vec<f32> {
    last(|r| r.errors.clone())
}
/// End point (y) of every pixel after the tracking; NaN for background.
#[wasm_bindgen]
pub fn last_end_y() -> Vec<f32> {
    last(|r| r.end_y.clone())
}
#[wasm_bindgen]
pub fn last_end_x() -> Vec<f32> {
    last(|r| r.end_x.clone())
}
/// Number of end points per pixel: where the pixels gathered (the sinks).
#[wasm_bindgen]
pub fn last_sink_histogram() -> Vec<u32> {
    last(|r| r.hist.clone())
}
/// Pixel indices (y*w + x) of the seeds (histogram peaks).
#[wasm_bindgen]
pub fn last_seeds() -> Vec<u32> {
    last(|r| r.seeds.clone())
}
/// Number of pixels tracked (cell probability above the threshold).
#[wasm_bindgen]
pub fn last_n_foreground() -> usize {
    last(|r| r.n_fg)
}

/// Decode network outputs (each h*w, row-major) into masks. `niter` Euler steps (Cellpose:
/// 200 at the training scale, 200/rescale at the original scale); pixels with cell
/// probability > `cellprob_threshold` are tracked; masks whose flow error > `flow_threshold`
/// are removed (0 = no check); masks smaller than `min_size` pixels are removed. Returns the
/// number of masks; the arrays are read with the `last_*` functions.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn compute_masks(
    dy: &[f32], dx: &[f32], cellprob: &[f32], h: usize, w: usize, niter: usize, cellprob_threshold: f32,
    flow_threshold: f32, min_size: i32, max_size_fraction: f32,
) -> Result<u32, JsError> {
    let n = h * w;
    if dy.len() != n || dx.len() != n || cellprob.len() != n {
        return Err(JsError::new("compute_masks: dy, dx and cellprob must have h*w values"));
    }
    let r = dynamics::compute_masks(dy, dx, cellprob, h, w, niter, cellprob_threshold, flow_threshold, min_size, max_size_fraction);
    let count = r.masks.iter().copied().max().unwrap_or(0);
    LAST.with(|c| *c.borrow_mut() = Some(r));
    Ok(count)
}

/// Only the flow check and the clean-up, for a new flow threshold or minimum size without
/// tracking again: `masks_raw` and `errors` from `last_masks_raw` and `last_flow_errors`.
#[wasm_bindgen]
pub fn finish_masks(masks_raw: &[u32], errors: &[f32], h: usize, w: usize, flow_threshold: f32, min_size: i32) -> Vec<u32> {
    dynamics::finish_masks(masks_raw, errors, h, w, flow_threshold, min_size)
}

/// Paths of the pixels `starts` (indices y*w + x) through the tracking flow (dP * fg / 5):
/// positions every `stride` steps, (y, x) interleaved, niter/stride + 1 points per start.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn trace_paths(
    dy: &[f32], dx: &[f32], cellprob: &[f32], h: usize, w: usize, cellprob_threshold: f32, starts: &[u32], niter: usize,
    stride: usize,
) -> Vec<f32> {
    let (my, mx) = dynamics::tracking_flow(dy, dx, cellprob, cellprob_threshold);
    dynamics::trace_paths(&my, &mx, h, w, starts, niter, stride)
}

/// Flows of a label image (the training targets): diffusion from each mask's centre,
/// normalised gradient. Returns (2, h, w): dY then dX, unit length inside masks.
#[wasm_bindgen]
pub fn masks_to_flows(masks: &[u32], h: usize, w: usize) -> Vec<f32> {
    let (fy, fx) = dynamics::masks_to_flows(masks, h, w);
    let mut out = fy;
    out.extend_from_slice(&fx);
    out
}

/// Flow error per label (1..=max): mean squared difference between the flows of the masks
/// and the network's flows / 5.
#[wasm_bindgen]
pub fn flow_errors(masks: &[u32], dy: &[f32], dx: &[f32], h: usize, w: usize) -> Vec<f32> {
    dynamics::flow_errors(masks, dy, dx, h, w)
}
