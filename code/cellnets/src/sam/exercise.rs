// Coding exercises for "Promptable segmentation: SAM". Replace each `todo!()` with your code,
// then check it with `cargo test -p cellnets sam::` and look at `cargo run -p cellnets -- sam`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// A candidate mask of automatic segmentation: its pixels, its box (`x0, y0, x1, y1`,
/// inclusive), its score (the network's own quality estimate) and its stability.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub pixels: Vec<(usize, usize)>,
    pub bbox: (usize, usize, usize, usize),
    pub score: f32,
    pub stability: f32,
}

/// The overlap of two boxes, as IoU of the boxes (counted in pixels, boxes inclusive).
pub fn box_iou(a: (usize, usize, usize, usize), b: (usize, usize, usize, usize)) -> f32 {
    let area = |r: (usize, usize, usize, usize)| ((r.2 - r.0 + 1) * (r.3 - r.1 + 1)) as f32;
    let (x0, y0, x1, y1) = (a.0.max(b.0), a.1.max(b.1), a.2.min(b.2), a.3.min(b.3));
    if x1 < x0 || y1 < y0 {
        return 0.0;
    }
    let inter = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f32;
    inter / (area(a) + area(b) - inter)
}

/// C1. Enlarge a coarse logit map to `width × height` by bilinear interpolation. Output
/// pixel `(x, y)` sits at `u = (x + ½) · lw / width − ½`, `v = (y + ½) · lh / height − ½` in
/// the coarse map (`lw × lh`): the four coarse pixels around `(u, v)` are mixed by their
/// distances. Positions outside the coarse map use its edge (clamp).
///
/// Hint: with `u0 = u.floor()` and `fu = u − u0`, the weights along x are `1 − fu` for
/// column `u0` and `fu` for column `u0 + 1`. `get_border(.., Border::Clamp)` reads the edge.
pub fn upsample_logits(low: &Image<f32>, width: usize, height: usize) -> Image<f32> {
    todo!("C1: upsample_logits")
}

/// C2. The stability of a mask: the number of pixels with logit above `delta`, divided by
/// the number with logit above `−delta` (0 if there are none).
pub fn stability(logits: &Image<f32>, delta: f32) -> f32 {
    todo!("C2: stability")
}

/// C3. The mask of a click: the pixels with logit above 0 that are connected to the
/// clicked pixel `(x, y)` (4-neighbours). Empty if the clicked pixel itself is not above 0.
///
/// Hint: flood fill with a stack, starting at the click.
pub fn mask_from_click(logits: &Image<f32>, x: usize, y: usize) -> Image<bool> {
    todo!("C3: mask_from_click")
}

/// C4. The filters of automatic masks: keep the candidates with `score > score_min` and
/// `stability >= stability_min`; then remove duplicates greedily: from the highest score
/// down, keep a candidate unless its box overlaps (box IoU) a kept candidate's box by more
/// than `box_overlap`. Returns the indices of the kept candidates, best first.
pub fn filter_masks(cands: &[Candidate], score_min: f32, stability_min: f32, box_overlap: f32) -> Vec<usize> {
    todo!("C4: filter_masks")
}
