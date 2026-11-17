// Coding exercises for "Intensity inside an object". Replace each `todo!()` with your code,
// then check it with `cargo test -p intensity-features` and look at `cargo run -p intensity-features`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// The statistics of the pixel values inside one object (slide "Statistics of the pixel values").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stats {
    pub n: usize,
    pub integrated: f32,
    pub mean: f32,
    /// Standard deviation, dividing by `n` (the pixels are the whole object).
    pub sd: f32,
    pub min: f32,
    pub max: f32,
    pub median: f32,
    pub q1: f32,
    pub q3: f32,
    pub mad: f32,
}

/// C1a. The quantile `p` (between 0 and 1) of a list of values: sort them, take position
/// `h = (n − 1) p` counting from 0, and interpolate linearly between the two neighbours.
///
/// Hint: copy the values first (`values.to_vec()`), then sort with
/// `v.sort_by(|a, b| a.partial_cmp(b).unwrap())`. Take care at the last position.
pub fn quantile(values: &[f32], p: f32) -> f32 {
    todo!("C1a: quantile")
}

/// C1b. The median absolute deviation: the median of the distances `|v − median|`.
pub fn mad(values: &[f32]) -> f32 {
    todo!("C1b: mad")
}

/// C1c. The robust z-score of `x` against control values: `(x − median) / (1.4826 · MAD)`.
pub fn robust_z(x: f32, controls: &[f32]) -> f32 {
    todo!("C1c: robust_z")
}

/// C2. All statistics of the values of the pixels with `labels[(x, y)] == label`.
///
/// Hint: collect the values into a `Vec<f32>` first, then use `quantile` and `mad`.
pub fn object_stats(img: &Image<f32>, labels: &Image<u32>, label: u32) -> Stats {
    todo!("C2: object_stats")
}

/// C3. The edge band of an object: its pixels that have a 4-neighbour outside the object
/// (another label, or outside the image). This is the object minus its erosion by the cross.
///
/// Hint: `labels.contains(x, y)` takes signed numbers and says whether `(x, y)` is in the image.
pub fn edge_band(labels: &Image<u32>, label: u32) -> Image<bool> {
    todo!("C3: edge_band")
}

/// C4. The geometric centroid, the intensity-weighted centroid, and the mass displacement
/// (the distance between them) of an object. Centroids are `(x, y)`.
///
/// Hint: one pass over the pixels, adding up the six sums m00, m10, m01 (count, x, y) and
/// M00, M10, M01 (the same, each pixel weighted by its value).
pub fn mass_displacement(img: &Image<f32>, labels: &Image<u32>, label: u32) -> ([f32; 2], [f32; 2], f32) {
    todo!("C4: mass_displacement")
}

/// C5a. The ring (1 to `k`) of a pixel at distance `dc` from the centre and `de` from the
/// edge: `ρ = dc / (dc + de)`, ring `min(k, ⌊k ρ⌋ + 1)`.
pub fn ring_of(dc: f32, de: f32, k: usize) -> usize {
    todo!("C5a: ring_of")
}

/// C5b. FracAtD and MeanFrac of each ring, from the pixel values and their rings (1 to `k`):
/// FracAtD = the ring's share of the total intensity, MeanFrac = FracAtD / the ring's
/// share of the pixels. Returns `(frac_at_d, mean_frac)`, each with `k` entries.
///
/// Hint: ring `r` goes into element `r − 1`. Give an empty ring MeanFrac 0.
pub fn radial_distribution(values: &[f32], rings: &[usize], k: usize) -> (Vec<f32>, Vec<f32>) {
    todo!("C5b: radial_distribution")
}
