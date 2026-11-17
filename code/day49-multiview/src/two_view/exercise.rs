// Coding exercises for "Two views: matches, epipolar geometry and triangulation".
// Replace each `todo!()` with your code, then check it with `cargo test -p day49 two_view::`
// and look at `cargo run -p day49 -- two-view`.
// Pixels (u, v), u right, v down; a match is (pixel in image 1, pixel in image 2); a line
// (a, b, c) is the set of pixels with a u + b v + c = 0.
#![allow(unused_variables, unused_imports)]

use imgcourse::linalg::sym_eigen;
use imgcourse::Rng;

/// 3-vectors and 3 × 3 matrices (row by row: `m[row][column]`) with `mat_vec`, `mat_mul`,
/// `transpose` and `dot`, in `src/common.rs`.
pub use crate::common::{dot, mat_mul, mat_vec, transpose, Mat3, Vec3};

/// A pair of matched pixels: (in image 1, in image 2).
pub type Match = ([f64; 2], [f64; 2]);

/// The similarity transform T (3 × 3) that moves 2D points to mean 0 and an RMS distance of
/// √2 from the origin (Hartley's normalisation).
pub fn normalising_transform(points: &[[f64; 2]]) -> Mat3 {
    let n = points.len() as f64;
    let (mx, my) = (points.iter().map(|p| p[0]).sum::<f64>() / n, points.iter().map(|p| p[1]).sum::<f64>() / n);
    let rms = (points.iter().map(|p| (p[0] - mx).powi(2) + (p[1] - my).powi(2)).sum::<f64>() / n).sqrt();
    let s = 2f64.sqrt() / rms;
    [[s, 0.0, -s * mx], [0.0, s, -s * my], [0.0, 0.0, 1.0]]
}

/// C1a. The epipolar line in image 2 of the pixel `x` in image 1: `F (u, v, 1)`.
pub fn epipolar_line(f: &Mat3, x: [f64; 2]) -> Vec3 {
    todo!("C1a: epipolar_line")
}

/// C1b. The distance in pixels of the pixel `x` from the line `l = (a, b, c)`:
/// `|a u + b v + c| / sqrt(a² + b²)`.
pub fn line_distance(l: Vec3, x: [f64; 2]) -> f64 {
    todo!("C1b: line_distance")
}

/// C2. The normalised eight-point algorithm (slide "The math: the eight-point algorithm"):
/// F from n ≥ 8 matches, with x'ᵀ F x = 0, rank 2, up to scale.
///
/// Hint: normalise with `normalising_transform` (one T per image). The row of a match
/// with normalised pixels p (image 1) and q (image 2) is
/// (q₀p₀, q₀p₁, q₀, q₁p₀, q₁p₁, q₁, p₀, p₁, 1). `sym_eigen` sorts the eigenvalues largest first, so
/// the null vector is the last eigenvector of AᵀA. For rank 2, let v be the last eigenvector of
/// F̂ᵀF̂ and multiply F̂ by (I − v vᵀ). Finally F = T₂ᵀ F̂ T₁.
pub fn eight_point(matches: &[Match]) -> Mat3 {
    todo!("C2: eight_point")
}

/// C3. The number of RANSAC samples needed for probability `p` of at least one sample of `s`
/// correct matches, with a fraction `w` of correct matches: `log(1 − p) / log(1 − w^s)`,
/// rounded up.
///
/// Hint: `f64::ln` and `powi`; round up with `ceil`.
pub fn ransac_iterations(p: f64, w: f64, s: i32) -> usize {
    todo!("C3: ransac_iterations")
}

/// C4. RANSAC for F: `iterations` times, draw 8 different matches at random, fit F with
/// `eight_point`, and count the matches whose pixel in image 2 lies within `eps` pixels of
/// the epipolar line of its pixel in image 1. Keep the largest set of inliers, refit F on
/// it, and return that F with the indices of the inliers.
///
/// Hint: `rng.below(n)` gives a random index; draw again if it is already in the sample.
pub fn ransac_fundamental(matches: &[Match], eps: f64, iterations: usize, rng: &mut Rng) -> (Mat3, Vec<usize>) {
    todo!("C4: ransac_fundamental")
}

/// C5. Midpoint triangulation (slide "The math: triangulation"): the rays c1 + s d1 and
/// c2 + t d2; find s and t that bring the two points closest (a 2 × 2 linear system), and
/// return the middle of the segment between them.
///
/// Hint: with w = c2 − c1, the two equations are (d1·d1) s − (d1·d2) t = d1·w and
/// (d1·d2) s − (d2·d2) t = d2·w. Solve them by Cramer's rule.
pub fn triangulate_midpoint(c1: Vec3, d1: Vec3, c2: Vec3, d2: Vec3) -> Vec3 {
    todo!("C5: triangulate_midpoint")
}
