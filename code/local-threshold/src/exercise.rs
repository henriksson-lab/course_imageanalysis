// Coding exercises for "When one threshold is not enough". Replace each `todo!()` with your
// code, then check it with `cargo test -p local-threshold` and look at `cargo run -p local-threshold`.
#![allow(unused_variables, unused_imports)]
//
// Objects are dark in this unit: a pixel is object where I(x) < T(x). Pixels outside the
// image are read by copying the edge pixel (Border::Clamp).

use imgcourse::{Border, Image};

/// A rule that turns the local mean `m` and standard deviation `s` into a threshold.
#[derive(Clone, Copy, Debug)]
pub enum Rule {
    /// T = m − C
    MeanC(f32),
    /// T = m + k s (Niblack; k is about −0.2)
    Niblack(f32),
    /// T = m (1 + k (s / R − 1)) (Sauvola; k about 0.2–0.5, R the largest possible s:
    /// 128 for grey levels 0–255, 0.5 for grey values 0–1)
    Sauvola { k: f32, range: f32 },
}

/// C1. The local mean and standard deviation in the (2r + 1) × (2r + 1) window around every
/// pixel: m = box(I), s = sqrt(box(I²) − m²) (slide "Local mean and variance are box filters").
/// The box sums are made in two passes, first along the rows, then along the columns.
///
/// Hint: pass 1 makes an image of row sums of I and of I² (`map_xy` can return a pair
/// `(f64, f64)`); pass 2 adds up 2r + 1 of those along each column. Read outside the image
/// with `get_border(.., Border::Clamp)` in pass 1 and `.clamp(0, height - 1)` in pass 2.
/// Add in `f64`, and take `max(0.0)` before the square root.
pub fn local_mean_std(img: &Image<f32>, r: usize) -> (Image<f32>, Image<f32>) {
    todo!("C1: local_mean_std")
}

/// C2. The threshold of a rule for one pixel, from its local mean `m` and standard deviation `s`.
///
/// Hint: `match rule { Rule::MeanC(c) => ..., Rule::Niblack(k) => ..., Rule::Sauvola { k, range } => ... }`
pub fn rule_threshold(rule: Rule, m: f32, s: f32) -> f32 {
    todo!("C2: rule_threshold")
}

/// C3. A local threshold: object (true) where the pixel is darker than the threshold of its
/// own window, I(x) < T(x).
pub fn local_threshold(img: &Image<f32>, r: usize, rule: Rule) -> Image<bool> {
    todo!("C3: local_threshold")
}

/// C4. Hysteresis for **bright** structures: seeds are the pixels above `high`; keep every
/// pixel above `low` that is connected to a seed by a chain of neighbours above `low`
/// (4-neighbours, or 8 if `eight`).
///
/// Hint: mark all seeds as kept and put them on a stack (`Vec`). Then repeatedly pop a
/// pixel and look at its neighbours: a neighbour inside the image (`img.contains`), not yet
/// kept and above `low` is marked and pushed. Stop when the stack is empty.
pub fn hysteresis(img: &Image<f32>, low: f32, high: f32, eight: bool) -> Image<bool> {
    todo!("C4: hysteresis")
}
