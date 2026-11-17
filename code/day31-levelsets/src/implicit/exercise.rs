// Coding exercises for "Curves as level sets". Replace each `todo!()` with your code,
// then check it with `cargo test -p day31 implicit::` and look at `cargo run -p day31 -- implicit`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1a. Union of two shapes (slide "Set operations for free"): inside either, `min(φ1, φ2)`.
///
/// Hint: `a.zip_map(b, |p, q| ...)` combines two images pixel by pixel.
pub fn union(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    todo!("C1a: union")
}

/// C1b. Intersection: inside both, `max(φ1, φ2)`.
pub fn intersection(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    todo!("C1b: intersection")
}

/// C1c. Difference `A \ B`: inside A and outside B, `max(φ1, −φ2)`.
pub fn difference(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    todo!("C1c: difference")
}

/// C2. Where the curve crosses row `y`: for every pair of neighbours `x`, `x + 1` whose
/// values have opposite signs, the crossing by linear interpolation, `x + φ(x) / (φ(x) − φ(x+1))`
/// (in pixels, from left to right). A value of exactly 0 counts as a crossing at that pixel.
///
/// Hint: the sign changes between `a` and `b` when `a * b < 0`.
pub fn row_crossings(phi: &Image<f32>, y: usize) -> Vec<f32> {
    todo!("C2: row_crossings")
}

/// C3. The outward unit normal `∇φ / |∇φ|` at pixel `(x, y)` from central differences
/// `φ_x = (φ(x+1, y) − φ(x−1, y)) / 2` (likewise `φ_y`), and `|∇φ|`. Neighbours outside the
/// image: copy the edge.
///
/// Hint: `phi.get_border(x as isize + 1, y as isize, Border::Clamp)` is the right neighbour.
pub fn normal(phi: &Image<f32>, x: usize, y: usize) -> ([f32; 2], f32) {
    todo!("C3: normal")
}

/// C4. The curvature of the level line through every pixel (slide "The math: normal and
/// curvature"):
/// `κ = (φ_xx φ_y² − 2 φ_x φ_y φ_xy + φ_yy φ_x²) / (φ_x² + φ_y²)^(3/2)`, all derivatives by
/// central differences, `φ_xy = (φ(x+1, y+1) − φ(x+1, y−1) − φ(x−1, y+1) + φ(x−1, y−1)) / 4`.
/// Where the gradient is 0, `κ = 0`.
///
/// Hint: `map_xy` gives you `(x, y)` and the value; write a small closure `at(dx, dy)` that reads
/// a neighbour.
pub fn curvature(phi: &Image<f32>) -> Image<f32> {
    todo!("C4: curvature")
}
