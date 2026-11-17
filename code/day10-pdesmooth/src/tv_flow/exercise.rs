// Coding exercises for "Total variation flow". Replace each `todo!()` with your
// code, then check it with `cargo test -p day10 tv_flow::` and look at `cargo run -p day10 -- tv-flow`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// What a gradient of size `s` costs (slide "Three smoothers, three energies").
#[derive(Clone, Copy, Debug)]
pub enum Penalty {
    /// `s^2 / 2`
    Heat,
    /// `s`
    Tv,
    /// `K^2 / 2 * ln(1 + s^2 / K^2)`, with the contrast parameter `K`
    PeronaMalik(f32),
}

/// C1. The energy of a row: the sum of the penalty of every neighbour difference.
///
/// Hint: `match penalty { Penalty::Heat => ..., Penalty::Tv => ..., Penalty::PeronaMalik(k) => ... }`;
/// `x.ln()` is the natural logarithm.
pub fn energy_1d(row: &[f32], penalty: Penalty) -> f32 {
    todo!("C1: energy_1d")
}

/// C2. One explicit step of TV flow on a row: between every pair of neighbours the flow is
/// `d / sqrt(d^2 + eps^2)` (about the sign of the difference `d`: one unit of flow, however
/// large the step). Mirror border: nothing flows out at the ends. Stable for `dt <= eps / 2`.
///
/// Hint: the Perona–Malik step of the last unit with `g(d) = 1 / sqrt(d^2 + eps^2)`: first the flow
/// of every pair, then every value gains the flow from the right and loses the flow to the left.
pub fn tv_step_1d(row: &[f32], dt: f32, eps: f32) -> Vec<f32> {
    todo!("C2: tv_step_1d")
}

/// C3. One explicit step of TV flow on an image, `u + dt * div(grad u / |grad u|)`
/// (slide "TV flow: level lines move by their curvature"). At every pixel take the
/// differences to the right and downwards, `(ux, uy)`, divide both by
/// `sqrt(ux^2 + uy^2 + eps^2)` to get a vector `(px, py)` of length below 1; then the change
/// is the divergence, `px(x, y) - px(x-1, y) + py(x, y) - py(x, y-1)`. Mirror border: no
/// difference, and so no flow, across the edge of the image. Stable for `dt <= eps / 4`.
///
/// Hint: two passes over the image. The first fills two images `px` and `py`; at the right edge
/// `ux = 0`, at the bottom edge `uy = 0`. The second adds `dt` times the divergence, using 0 for
/// `px(x-1, y)` in the first column and for `py(x, y-1)` in the first row.
pub fn tv_step(img: &Image<f32>, dt: f32, eps: f32) -> Image<f32> {
    todo!("C3: tv_step")
}
