// Coding exercises for "What the microscope does to colocalisation". Replace each `todo!()`
// with your code, then check it with `cargo test -p colocalisation coloc_confounds::` and look at
// `cargo run -p colocalisation -- coloc-confounds`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1a. The bleed-through factor from a single-stained control (slide "Bleed-through"): the
/// slope through the origin of B against A over the mask, Σ A·B / Σ A².
pub fn leak_slope(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>) -> f32 {
    todo!("C1a: leak_slope")
}

/// C1b. Compensation: B − s·A at every pixel.
pub fn compensate(b: &Image<f32>, a: &Image<f32>, s: f32) -> Image<f32> {
    todo!("C1b: compensate")
}

/// C2. The normalised overlap of two Gaussian spots of widths `s1`, `s2` whose centres are
/// `delta` apart (slide "The math: Gaussian spot pairs"):
/// 2 s1 s2 / (s1² + s2²) · exp(−delta² / (2 (s1² + s2²))).
pub fn gaussian_overlap(s1: f32, s2: f32, delta: f32) -> f32 {
    todo!("C2: gaussian_overlap")
}

/// C3. Background subtraction with a white top-hat (slide "Background: offset and shared
/// background"): the opening of the image (the minimum over the (2r+1) × (2r+1) square
/// around every pixel, then the maximum over the same square of that result) is the
/// background; return image − opening. Outside the image, read the nearest edge pixel.
///
/// Hint: write the two passes as two loops over the pixels with an inner double loop over
/// the square, using `get_border(…, Border::Clamp)`; the second pass reads the result of
/// the first.
pub fn subtract_background(img: &Image<f32>, radius: usize) -> Image<f32> {
    todo!("C3: subtract_background")
}

/// C4a. The shift of B against A, measured on bead images (slide "Chromatic shift"): the shift
/// (tx, ty), each from −max_shift to max_shift, that maximises the cross-correlation
/// c(tx, ty) = Σ (A(x, y) − mean A) (B(x + tx, y + ty) − mean B), with zero outside the image;
/// then refined in x and in y by a parabola through the peak and its two neighbours (as in
/// the shift curve of the last unit). If B is A moved right by 1.5 pixels, the answer is
/// close to (1.5, 0).
///
/// Hint: first subtract the means, then try every shift and keep the best. At the peak
/// (tx, ty), the parabola through c(tx − 1, ty), c(tx, ty), c(tx + 1, ty) has its vertex at
/// tx + (c₋ − c₊) / (2 (c₋ − 2c₀ + c₊)); likewise in y.
pub fn estimate_shift(a: &Image<f32>, b: &Image<f32>, max_shift: usize) -> (f32, f32) {
    todo!("C4a: estimate_shift")
}

/// C4b. Shift an image back: the result at (x, y) is `img` at (x + dx, y + dy), read between
/// pixels by bilinear interpolation (the nearest edge pixel outside). With the shift from
/// `estimate_shift`, `shift_image(b, tx, ty)` lines B up with A.
pub fn shift_image(img: &Image<f32>, dx: f32, dy: f32) -> Image<f32> {
    todo!("C4b: shift_image")
}
