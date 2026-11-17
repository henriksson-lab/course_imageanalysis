// Coding exercises for "Boundaries, counts and the uncertain ground truth". Replace each `todo!()` with your code,
// then check it with `cargo test -p day37 agreement::` and look at `cargo run -p day37 -- agreement`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1. The boundary pixels of a mask: object pixels with at least one of their 4 neighbours
/// outside the object (pixels outside the image count as outside). Returned as points
/// `(x, y)`.
///
/// Hint: `mask.contains(nx, ny)` says whether `(nx, ny)` (as `isize`) is inside the image.
pub fn boundary(mask: &Image<bool>) -> Vec<(f32, f32)> {
    todo!("C1: boundary")
}

/// C2. For every point of `a`, the distance to the nearest point of `b` (by trying them all).
///
/// Hint: two loops; keep the smallest distance in a variable that starts at `f32::INFINITY`.
pub fn nearest_distances(a: &[(f32, f32)], b: &[(f32, f32)]) -> Vec<f32> {
    todo!("C2: nearest_distances")
}

/// Boundary distances between two outlines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryScores {
    pub hausdorff: f32,
    pub hd95: f32,
    pub asd: f32,
}

/// C3. Hausdorff distance (the largest of all nearest distances, both ways), HD95 (the
/// larger of the two 95th percentiles) and ASD (the mean of all nearest distances of both
/// outlines together). The 95th percentile of `n` sorted values is the value number
/// `ceil(0.95 n)` (counting from 1).
///
/// Hint: call `nearest_distances` both ways. Sort a list with
/// `d.sort_by(|p, q| p.total_cmp(q))`; the value number `k` (from 1) is `d[k - 1]`.
pub fn boundary_scores(a: &[(f32, f32)], b: &[(f32, f32)]) -> BoundaryScores {
    todo!("C3: boundary_scores")
}

/// The Bland–Altman summary of a method against a reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlandAltman {
    pub bias: f32,
    pub sd: f32,
    pub lower: f32,
    pub upper: f32,
}

/// C4. Bland–Altman: the differences `method - reference`, their mean (the bias), their
/// standard deviation (divide by n - 1) and the limits of agreement `bias ± 1.96 sd`.
///
/// Hint: first the list of differences, then its mean, then the SD around that mean.
pub fn bland_altman(reference: &[f32], method: &[f32]) -> BlandAltman {
    todo!("C4: bland_altman")
}

/// C5. The correlation coefficient r of two lists:
/// `sum((x - mean x)(y - mean y)) / sqrt(sum((x - mean x)^2) sum((y - mean y)^2))`.
///
/// Hint: compute the two means first, then the three sums in one loop.
pub fn correlation(x: &[f32], y: &[f32]) -> f32 {
    todo!("C5: correlation")
}
