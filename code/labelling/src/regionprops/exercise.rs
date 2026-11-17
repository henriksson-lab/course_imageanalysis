// Coding exercises for "Region properties and moments". Replace each `todo!()` with your code,
// then check it with `cargo test -p labelling regionprops::` and look at `cargo run -p labelling -- regionprops`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// One row of the measurement table: the sums over the pixels of one object.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Region {
    /// number of pixels, m00
    pub area: usize,
    /// centroid (x̄, ȳ) = (m10 / m00, m01 / m00)
    pub cx: f32,
    pub cy: f32,
    /// central moments μ20 = Σ (x − x̄)², μ11 = Σ (x − x̄)(y − ȳ), μ02 = Σ (y − ȳ)²
    pub mu20: f32,
    pub mu11: f32,
    pub mu02: f32,
    /// bounding box, inclusive
    pub x_min: usize,
    pub x_max: usize,
    pub y_min: usize,
    pub y_max: usize,
}

impl Region {
    /// Area / area of the bounding box.
    pub fn extent(&self) -> f32 {
        self.area as f32 / ((self.x_max - self.x_min + 1) * (self.y_max - self.y_min + 1)) as f32
    }
}

/// The equivalent ellipse: the eigenvalues λ1 ≥ λ2 of S = (1/A) [[μ20, μ11], [μ11, μ02]], and
/// the angle `theta` of the eigenvector v1 = (cos θ, sin θ), in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ellipse {
    pub lambda1: f32,
    pub lambda2: f32,
    pub theta: f32,
}

impl Ellipse {
    /// √(1 − λ2/λ1): 0 for a disc, near 1 for a rod.
    pub fn eccentricity(&self) -> f32 {
        (1.0 - self.lambda2 / self.lambda1).sqrt()
    }

    /// The semi-axes 2√λ1 and 2√λ2.
    pub fn semi_axes(&self) -> (f32, f32) {
        (2.0 * self.lambda1.sqrt(), 2.0 * self.lambda2.sqrt())
    }
}

/// C1. One scan over the label image: area, centroid, central moments and bounding box of
/// every object. Element `k - 1` of the result is label `k`, for the labels `1..=max`.
///
/// Hint: sum x, y, x², xy, y² per label (in `f64`) in one scan, then use the
/// formulas μ20 = m20 − m10²/m00 and so on. `labels.pixels().iter().max()` gives the number of labels.
pub fn measure(labels: &Image<u32>) -> Vec<Region> {
    todo!("C1: measure")
}

/// C2. The equivalent ellipse of a region: the eigenvalues of S, largest first, and the angle
/// θ = ½ atan2(2 μ11, μ20 − μ02) of the major axis, measured from the x axis towards the
/// y axis (y points down, so positive θ turns clockwise on screen).
///
/// Hint: S = [[a, b], [b, c]] with a = μ20/A, b = μ11/A, c = μ02/A; λ = (a + c)/2 ± √(((a − c)/2)² + b²).
/// `y.atan2(x)` is atan2(y, x).
pub fn ellipse(r: &Region) -> Ellipse {
    todo!("C2: ellipse")
}

/// Is `(x, y)` an object pixel? Positions outside the image are background. (A helper for C3–C5.)
pub fn at(mask: &Image<bool>, x: isize, y: isize) -> bool {
    mask.contains(x, y) && mask[(x as usize, y as usize)]
}

/// C3. Perimeter by edge count: the number of pixel edges between an object pixel and a
/// background pixel (outside the image counts as background).
///
/// Hint: look at the 4 neighbours of every object pixel with `at(mask, x + dx, y + dy)`
/// (signed numbers: `x as isize + dx`).
pub fn perimeter_edges(mask: &Image<bool>) -> usize {
    todo!("C3: perimeter_edges")
}

/// C4. Perimeter by the Crofton formula: count the crossings of the boundary along rows (n0),
/// columns (n90) and both diagonals (n45, n135), i.e. the pairs of neighbouring pixels in that
/// direction where one is object and the other background; then
/// P ≈ π/8 · (n0 + n90 + (n45 + n135)/√2).
///
/// Hint: loop over positions from −1 to the width (and height), so that pairs with one
/// pixel just outside the image are counted too. The steps (1, 0), (0, 1), (1, −1), (1, 1) give
/// the four directions; `at` reads background outside the image.
pub fn perimeter_crofton(mask: &Image<bool>) -> f32 {
    todo!("C4: perimeter_crofton")
}

/// C5a. The bit-quad counts (Q1, Q3, QD): the 2 × 2 windows with exactly one object pixel,
/// with exactly three, and with two object pixels on a diagonal. All windows that overlap the
/// image are counted, with background outside.
///
/// Hint: loop over the top-left corner (x, y) of the window from −1 to width − 1 (and height − 1).
pub fn bit_quads(mask: &Image<bool>) -> (i32, i32, i32) {
    todo!("C5a: bit_quads")
}

/// C5b. The Euler number (objects − holes) from the bit quads: (Q1 − Q3 − 2 QD)/4 for
/// 8-connected objects, (Q1 − Q3 + 2 QD)/4 for 4-connected objects.
pub fn euler_number(mask: &Image<bool>, eight: bool) -> i32 {
    todo!("C5b: euler_number")
}
