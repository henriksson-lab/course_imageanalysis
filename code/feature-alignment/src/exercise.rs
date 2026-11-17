// Coding exercises for "From matches to a transform". Replace each `todo!()` with your code,
// then check it with `cargo test -p feature-alignment` and look at `cargo run -p feature-alignment`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{linalg, Complex, Rng};

/// A point `(x, y)`.
pub type Point = (f32, f32);

/// A matched pair: a point of image 1 and its match in image 2.
pub type Pair = (Point, Point);

/// An affine transform, the top two rows of the 3 × 3 matrix: `x' = m[0][0] x + m[0][1] y + m[0][2]`,
/// `y' = m[1][0] x + m[1][1] y + m[1][2]`. Translations and similarities are special cases.
pub type Affine = [[f32; 3]; 2];

/// Not an exercise: apply a transform to a point.
pub fn apply(m: &Affine, (x, y): Point) -> Point {
    (m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2])
}

/// C1. The least-squares translation: the mean of the differences `q − p` over all pairs.
///
/// Hint: add up `u − x` and `v − y` over the pairs, then divide by the number of pairs.
pub fn fit_translation(pairs: &[Pair]) -> Point {
    todo!("C1: fit_translation")
}

/// C2. The similarity (rotation, scale, translation) that maps `p1 → q1` and `p2 → q2`
/// exactly. In complex numbers a similarity is `z ↦ a z + b`: `a = (q2 − q1) / (p2 − p1)`,
/// `b = q1 − a p1`. As a matrix: `[[Re a, −Im a, Re b], [Im a, Re a, Im b]]`.
///
/// Hint: `Complex::new(x, y)` makes a point complex. Dividing by `d` is multiplying by
/// `d.conj() * (1.0 / d.norm_sqr())`.
pub fn similarity_from_two(p1: Point, q1: Point, p2: Point, q2: Point) -> Affine {
    todo!("C2: similarity_from_two")
}

/// C3. The least-squares affine transform of three or more pairs (exact for three). Each
/// row of the matrix is its own small problem with the same 3 × 3 normal equations
/// `BᵀB m = Bᵀx'` (rows of `B`: `[x, y, 1]`); solve them with `linalg::solve`.
/// Returns `None` if the points of image 1 lie on a line.
///
/// Hint: add up `BᵀB` (a 3 × 3 table of f64) and the two right-hand sides `Bᵀx'` and `Bᵀy'`
/// pair by pair, with the row `[x, y, 1]`; `linalg::solve(&btb, &rhs)?` returns early with None.
pub fn fit_affine(pairs: &[Pair]) -> Option<Affine> {
    todo!("C3: fit_affine")
}

/// C4a. The inliers of a transform: the indices of the pairs whose point of image 1 is
/// mapped to within `eps` px of its match.
///
/// Hint: `apply(m, p)` maps a point; `(a - b).hypot(c - d)` is a distance.
pub fn inliers(m: &Affine, pairs: &[Pair], eps: f32) -> Vec<usize> {
    todo!("C4a: inliers")
}

/// C4b. The number of RANSAC iterations for inlier fraction `w`, sample size `s` and
/// success probability `p`: `N = log(1 − p) / log(1 − w^s)`, rounded up.
///
/// Hint: `w.powi(s as i32)`, `.ln()` and `.ceil() as usize`.
pub fn ransac_iterations(w: f32, s: u32, p: f32) -> usize {
    todo!("C4b: ransac_iterations")
}

/// C5. RANSAC for an affine transform: `iterations` times, pick 3 different pairs at random,
/// fit them exactly (`fit_affine`), count the inliers; keep the transform with the most.
/// Then refit by least squares on its inliers and count again. Returns the transform and
/// its inliers, or `None` if no sample could be fitted.
///
/// Hint: `rng.below(pairs.len())` picks an index; skip a sample with a repeated index.
/// Keep the best as `Option<(Affine, Vec<usize>)>`; at the end refit with `fit_affine` on the inliers.
pub fn ransac_affine(pairs: &[Pair], eps: f32, iterations: usize, rng: &mut Rng) -> Option<(Affine, Vec<usize>)> {
    todo!("C5: ransac_affine")
}
