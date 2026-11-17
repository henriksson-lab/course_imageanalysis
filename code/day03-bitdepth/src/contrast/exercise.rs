// Coding exercises for "Contrast and intensity transforms". Replace each `todo!()` with your
// code, then check it with `cargo test -p day03 contrast::` and look at `cargo run -p day03 -- contrast`.
#![allow(unused_variables, unused_imports)]

// Images here hold whole numbers (`u32`): grey levels 0 … L − 1, with L = 256 for 8 bits or
// 4096 for 12 bits. A curve is a lookup table (LUT): `lut[v]` is the new value of `v`.

use imgcourse::Image;

/// C1a. Apply a lookup table to every pixel: the new value of `v` is `lut[v]`.
///
/// Hint: `img.map(|v| ...)`; a `u32` is turned into an index with `v as usize`.
pub fn apply_lut(img: &Image<u32>, lut: &[u32]) -> Image<u32> {
    todo!("C1a: apply_lut")
}

/// C1b. The gamma curve as a table from `levels_in` to `levels_out` levels: scale v to [0, 1],
/// raise it to the power γ, scale to 0 … levels_out − 1 and round.
///
/// Hint: `(0..levels_in).map(|v| ...).collect()`; `x.powf(gamma)`.
pub fn gamma_lut(levels_in: usize, levels_out: usize, gamma: f32) -> Vec<u32> {
    todo!("C1b: gamma_lut")
}

/// C1c. The log curve as a table: s = c ln(1 + v), with c chosen so that the largest input
/// value maps to the largest output value: c = (levels_out − 1) / ln(levels_in).
pub fn log_lut(levels_in: usize, levels_out: usize) -> Vec<u32> {
    todo!("C1c: log_lut")
}

/// C2a. The histogram: how many pixels have each value 0 … levels − 1.
pub fn histogram(img: &Image<u32>, levels: usize) -> Vec<u32> {
    todo!("C2a: histogram")
}

/// C2b. The equalisation curve: s_k = round((L − 1) c_k / N), where c_k is the cumulative
/// count (pixels with value ≤ k), N the number of pixels and L the number of levels.
///
/// Hint: keep a running sum of the counts while going through the levels.
pub fn equalise_lut(hist: &[u32]) -> Vec<u32> {
    todo!("C2b: equalise_lut")
}

/// C3. Histogram matching: for each input level k, the smallest target level j whose
/// cumulative fraction G_j reaches the input's cumulative fraction F_k (T = G⁻¹ ∘ F).
///
/// Hint: first the two cumulative fractions (running sums divided by the totals). Then for
/// each F_k look for the first j with G_j ≥ F_k (allow a tiny tolerance such as 1e-6 for rounding).
pub fn match_lut(hist: &[u32], target: &[u32]) -> Vec<u32> {
    todo!("C3: match_lut")
}

/// C4. CLAHE's clipped histogram: cut every bin at β × the mean count, spread the excess
/// evenly over all bins, and repeat until no bin is above the limit (at most 100 times).
///
/// Hint: the limit is β·N/L for N pixels and L bins. One round: add up the excess above the
/// limit, then every bin becomes min(bin, limit) + excess/L. Stop when the excess is (almost) 0.
pub fn clip_histogram(hist: &[f32], beta: f32) -> Vec<f32> {
    todo!("C4: clip_histogram")
}

/// C5. CLAHE: cut the image into `tiles × tiles` tiles; for each tile, the equalisation curve
/// of its clipped histogram, T(v) = (L − 1)/N_tile · (sum of the clipped counts up to v). Each
/// pixel blends the curves of the four nearest tile centres with bilinear weights.
///
/// Hint: tile (tx, ty) is `img.crop(tx * tw, ty * th, tw, th)` with tw = width / tiles (give the
/// last row and column of tiles any leftover pixels). Store one curve (a `Vec<f32>`) per tile. For a
/// pixel at x, f = (x + ½)/tw − ½ is its position among the tile centres: the two nearest
/// centres are ⌊f⌋ and ⌊f⌋ + 1 (kept inside 0 … tiles − 1), with weight a = f − ⌊f⌋ (kept in
/// 0 … 1) for the second. The same along y, then bilinear weights as in the images unit.
pub fn clahe(img: &Image<u32>, levels: usize, tiles: usize, beta: f32) -> Image<u32> {
    todo!("C5: clahe")
}
