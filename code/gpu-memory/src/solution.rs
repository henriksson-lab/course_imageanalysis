// Model solution for the coding exercises of "Memory is the bottleneck: bandwidth,
// coalescing, transfers and the roofline". The memory system is a model with the rules of a
// real GPU (128-byte lines in four 32-byte sectors), counted in plain Rust.

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
pub fn warp_traffic(addresses: &[usize]) -> Traffic {
    let mut lines: Vec<usize> = addresses.iter().map(|a| a / 128).collect();
    let mut sectors: Vec<usize> = addresses.iter().map(|a| a / 32).collect();
    let mut floats: Vec<usize> = addresses.to_vec();
    for v in [&mut lines, &mut sectors, &mut floats] {
        v.sort();
        v.dedup(); // count each line, sector and float once
    }
    let bytes_moved = 32 * sectors.len();
    let bytes_used = 4 * floats.len();
    Traffic { lines: lines.len(), sectors: sectors.len(), bytes_moved, bytes_used, efficiency: bytes_used as f32 / bytes_moved as f32 }
}

/// P2. How often each input pixel is read from global memory when a `k` × `k` filter
/// (`k` odd) computes an `out_w` × `out_h` output in tiles of `tile` × `tile` output pixels:
/// every tile loads its input window, the tile plus a halo of `(k − 1)/2` on each side, once.
/// The returned image covers the input with its halo, `(out_w + k − 1)` × `(out_h + k − 1)`;
/// input pixel `(x, y)` of the result corresponds to output position `(x − r, y − r)`.
/// `tile = 1` is the filter without tiles: every output pixel reads its own `k × k` window.
pub fn tile_reads(out_w: usize, out_h: usize, tile: usize, k: usize) -> Image<u32> {
    let mut reads = Image::new(out_w + k - 1, out_h + k - 1, 0u32);
    for ty in (0..out_h).step_by(tile) {
        for tx in (0..out_w).step_by(tile) {
            // The tile's outputs are tx..tx+tile (cut at the image edge); in the coordinates
            // of the result, its input window starts at the same (tx, ty) and is k − 1 wider.
            let (w, h) = (tile.min(out_w - tx), tile.min(out_h - ty));
            for y in ty..ty + h + k - 1 {
                for x in tx..tx + w + k - 1 {
                    reads[(x, y)] += 1;
                }
            }
        }
    }
    reads
}

/// P3. The roofline: the time of a kernel that does `work` operations and moves `bytes` bytes
/// on a processor with peak rate `peak` (operations/s) and bandwidth `bandwidth` (bytes/s),
/// when arithmetic and memory traffic overlap: `max(work/peak, bytes/bandwidth)`. Also says
/// whether the kernel is memory-bound (its intensity `work/bytes` is below `peak/bandwidth`).
pub fn roofline_time(work: f64, bytes: f64, peak: f64, bandwidth: f64) -> (f64, bool) {
    let compute = work / peak;
    let memory = bytes / bandwidth;
    (compute.max(memory), work / bytes < peak / bandwidth)
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
pub fn pipeline_separate(img: &Image<f32>, gain: f32, background: f32, t: f32, m: &mut Meter) -> Image<f32> {
    let (w, h) = (img.width(), img.height());
    let mut a = Image::new(w, h, 0.0);
    for (x, y) in img.coords() {
        let v = load(img, x, y, m);
        store(&mut a, x, y, scale(v, gain), m);
    }
    let mut b = Image::new(w, h, 0.0);
    for (x, y) in img.coords() {
        let v = load(&a, x, y, m);
        store(&mut b, x, y, subtract(v, background), m);
    }
    let mut c = Image::new(w, h, 0.0);
    for (x, y) in img.coords() {
        let v = load(&b, x, y, m);
        store(&mut c, x, y, clip(v), m);
    }
    let mut d = Image::new(w, h, 0.0);
    for (x, y) in img.coords() {
        let v = load(&c, x, y, m);
        store(&mut d, x, y, threshold(v, t), m);
    }
    d
}

/// P4b. The same pipeline fused into one pass: load each pixel once, apply all four steps,
/// store the result once.
pub fn pipeline_fused(img: &Image<f32>, gain: f32, background: f32, t: f32, m: &mut Meter) -> Image<f32> {
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let v = load(img, x, y, m);
        let v = threshold(clip(subtract(scale(v, gain), background)), t);
        store(&mut out, x, y, v, m);
    }
    out
}
