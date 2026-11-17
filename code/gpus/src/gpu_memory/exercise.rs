// Coding exercises for "Memory is the bottleneck".
// Replace each `todo!()` with your code, then check it with `cargo test -p gpus gpu_memory::`
// and look at `cargo run -p gpus -- gpu-memory`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// What one warp's read costs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Traffic {
    /// Distinct 128-byte lines and 32-byte sectors touched.
    pub lines: usize,
    pub sectors: usize,
    /// Bytes moved (whole sectors) and bytes the lanes asked for (4 per distinct float).
    pub bytes_moved: usize,
    pub bytes_used: usize,
    /// `bytes_used / bytes_moved`.
    pub efficiency: f32,
}

/// P1. The memory traffic of one warp in which every lane reads one 4-byte float at the byte
/// address `addresses[lane]`: memory delivers whole 32-byte sectors (four to a 128-byte line).
///
/// Hint: the line of an address is `address / 128`, its sector `address / 32`. Collect them in
/// vectors, then `sort()` and `dedup()` to count the distinct ones.
pub fn warp_traffic(addresses: &[usize]) -> Traffic {
    todo!("P1: warp_traffic")
}

/// P2. How often each input pixel is read from global memory when a `k` × `k` filter
/// (`k` odd) computes an `out_w` × `out_h` output in tiles of `tile` × `tile` output pixels:
/// every tile loads its input window, the tile plus a halo of `(k − 1)/2` on each side, once.
/// The returned image covers the input with its halo, `(out_w + k − 1)` × `(out_h + k − 1)`;
/// input pixel `(x, y)` of the result corresponds to output position `(x − r, y − r)`.
/// `tile = 1` is the filter without tiles: every output pixel reads its own `k × k` window.
///
/// Hint: loop over the tiles with `step_by(tile)`; a tile at `(tx, ty)` with `w × h` outputs
/// (smaller at the right and bottom edges) adds 1 to every pixel of the window `tx .. tx + w + k − 1`,
/// `ty .. ty + h + k − 1` of the result.
pub fn tile_reads(out_w: usize, out_h: usize, tile: usize, k: usize) -> Image<u32> {
    todo!("P2: tile_reads")
}

/// P3. The roofline: the time of a kernel that does `work` operations and moves `bytes` bytes
/// on a processor with peak rate `peak` (operations/s) and bandwidth `bandwidth` (bytes/s),
/// when arithmetic and memory traffic overlap: `max(work/peak, bytes/bandwidth)`. Also says
/// whether the kernel is memory-bound (its intensity `work/bytes` is below `peak/bandwidth`).
///
/// Hint: two divisions and a `max`; memory-bound when `work / bytes < peak / bandwidth`.
pub fn roofline_time(work: f64, bytes: f64, peak: f64, bandwidth: f64) -> (f64, bool) {
    todo!("P3: roofline_time")
}

/// Counts the bytes moved to and from global memory.
#[derive(Clone, Copy, Debug, Default)]
pub struct Meter {
    pub bytes: usize,
}

/// Read one pixel "from global memory": 4 bytes on the meter.
pub fn load(img: &Image<f32>, x: usize, y: usize, m: &mut Meter) -> f32 {
    m.bytes += 4;
    img[(x, y)]
}

/// Write one pixel "to global memory": 4 bytes on the meter.
pub fn store(img: &mut Image<f32>, x: usize, y: usize, v: f32, m: &mut Meter) {
    m.bytes += 4;
    img[(x, y)] = v;
}

/// The four steps of the pipeline, each on one value.
pub fn scale(v: f32, gain: f32) -> f32 {
    v * gain
}
pub fn subtract(v: f32, background: f32) -> f32 {
    v - background
}
pub fn clip(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}
pub fn threshold(v: f32, t: f32) -> f32 {
    if v > t {
        1.0
    } else {
        0.0
    }
}

/// P4a. The pipeline scale → subtract background → clip → threshold as four separate
/// passes ("kernels"): each pass loads every pixel of the previous image and stores a new
/// image, through `load` and `store`.
///
/// Hint: four loops over `img.coords()`, each reading the previous image with `load` and
/// writing a new image with `store`.
pub fn pipeline_separate(img: &Image<f32>, gain: f32, background: f32, t: f32, m: &mut Meter) -> Image<f32> {
    todo!("P4a: pipeline_separate")
}

/// P4b. The same pipeline fused into one pass: load each pixel once, apply all four steps,
/// store the result once.
///
/// Hint: one loop: `load`, the four steps one after the other, `store`.
pub fn pipeline_fused(img: &Image<f32>, gain: f32, background: f32, t: f32, m: &mut Meter) -> Image<f32> {
    todo!("P4b: pipeline_fused")
}
