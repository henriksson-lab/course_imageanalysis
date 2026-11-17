// Coding exercises for "Finding a shift by sliding and scoring". Replace each `todo!()` with your code,
// then check it with `cargo test -p fftreg matching::` and look at `cargo run -p fftreg -- matching`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// The three scores of one shift, over the overlap of `n` pixel pairs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scores {
    pub ssd: f32,
    pub cc: f32,
    pub ncc: f32,
    pub n: usize,
}

/// C1. SSD, CC and NCC of B placed at shift `t` on A (B's sample `x` lies on A's sample
/// `x + t`), over the overlap only (slide "Three scores"). NCC is 0 when one side of the
/// overlap is flat (no spread, exercise B2).
///
/// Hint: first collect the overlapping pairs `(a[x + t], b[x])` for the `x` where `x + t` lies
/// inside A (careful with negative numbers: work in `isize`). NCC: subtract the two means, then
/// `sum of products / sqrt(sum of squares of f · sum of squares of g)`.
pub fn scores_1d(a: &[f32], b: &[f32], t: isize) -> Scores {
    todo!("C1: scores_1d")
}

/// C2. NCC of image B placed at shift `(tx, ty)` on image A (B's pixel `(x, y)` lies on A's
/// pixel `(x + tx, y + ty)`), over the overlap. Returns `None` if the overlap has fewer than
/// `min_overlap` pixels; 0 if one side is flat.
///
/// Hint: `a.contains(ax, ay)` tells whether A has the pixel. Collect the sums of f, g, f², g²
/// and fg over the overlap (in `f64`); then `sum (f − mean f)(g − mean g) = sum fg − (sum f)(sum g)/n`,
/// and the same for the squares.
pub fn ncc_2d(a: &Image<f32>, b: &Image<f32>, tx: isize, ty: isize, min_overlap: usize) -> Option<f32> {
    todo!("C2: ncc_2d")
}

/// C3. Brute-force registration: try every shift with `|tx|, |ty| <= w` and keep the one with
/// the largest NCC (at least `min_overlap` pixels). Returns `(tx, ty, ncc)`.
///
/// Hint: two loops over `ty` and `tx`; `if let Some(c) = ncc_2d(...)`.
pub fn register(a: &Image<f32>, b: &Image<f32>, w: isize, min_overlap: usize) -> (isize, isize, f32) {
    todo!("C3: register")
}

/// C4. Template matching: the NCC of the template at every position where it fits completely
/// in the image. `out[(x, y)]` belongs to the template's top-left corner at `(x, y)`; the
/// output is `(W − w + 1) × (H − h + 1)` (slide "Template matching").
///
/// Hint: `Image::from_fn` with `ncc_2d(img, tpl, x as isize, y as isize, 1)` at every position.
pub fn template_match(img: &Image<f32>, tpl: &Image<f32>) -> Image<f32> {
    todo!("C4: template_match")
}
