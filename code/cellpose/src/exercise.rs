// Coding exercises for "Flow fields to objects: Cellpose". Replace each `todo!()` with your code,
// then check it with `cargo test -p cellpose` and look at `cargo run -p cellpose`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// The source of a cell: the cell pixel nearest to (median of the cell's x, median of its y).
pub fn source(mask: &Image<bool>) -> (usize, usize) {
    let pixels: Vec<(usize, usize)> = mask.coords().filter(|&(x, y)| mask[(x, y)]).collect();
    let mut xs: Vec<usize> = pixels.iter().map(|p| p.0).collect();
    let mut ys: Vec<usize> = pixels.iter().map(|p| p.1).collect();
    xs.sort();
    ys.sort();
    let (mx, my) = (xs[xs.len() / 2] as f32, ys[ys.len() / 2] as f32);
    let dist = |p: &(usize, usize)| (p.0 as f32 - mx).powi(2) + (p.1 as f32 - my).powi(2);
    *pixels.iter().min_by(|a, b| dist(a).partial_cmp(&dist(b)).unwrap()).unwrap()
}

/// C1. The heat target of one cell (`mask` is true inside it). Start with 0 everywhere; at
/// every step add 1 at the source, then replace every cell pixel by the mean of the pixels
/// of its 3 × 3 neighbourhood that lie inside the cell (itself included). Outside the cell
/// the heat stays 0.
///
/// Hint: compute the new values into a second image, so that every mean reads the values
/// of the previous step.
pub fn heat_target(mask: &Image<bool>, steps: usize) -> Image<f32> {
    todo!("C1: heat_target")
}

/// C2. The arrows of the target: at every cell pixel the heat difference of its two
/// neighbours, `(right − left, below − above)`, scaled to length 1 (a zero difference stays
/// zero). Neighbours outside the cell count as 0; outside the cell the arrow is (0, 0).
pub fn arrows(heat: &Image<f32>, mask: &Image<bool>) -> Image<[f32; 2]> {
    todo!("C2: arrows")
}

/// C3. Follow the flow: for every pixel of `mask`, put a particle at the pixel and move it
/// `steps` times by the arrow of the pixel it is on (the pixel at its rounded position).
/// Returns every particle's end position (`[x, y]`; `[0, 0]` outside the mask).
///
/// Hint: keep the position in `f32`; clamp the rounded position to the image before you
/// read the arrow.
pub fn follow(flow: &Image<[f32; 2]>, mask: &Image<bool>, steps: usize) -> Image<[f32; 2]> {
    todo!("C3: follow")
}

/// C4. Masks from the end points. Count the particles that ended in every pixel (at their
/// rounded end position). The sink pixels are those with at least `min_count` particles;
/// sink pixels that touch (8-neighbours) form one sink. Every mask pixel whose particle ended
/// in a sink gets the number of that sink, 1, 2, ... in the order the sinks are found (row
/// by row); the others get 0 (no object).
///
/// Hint: number the sinks by flood fill with a stack, as in connected components.
pub fn group_ends(ends: &Image<[f32; 2]>, mask: &Image<bool>, min_count: usize) -> Image<u32> {
    todo!("C4: group_ends")
}

/// C5. The flow error of one mask: the mean, over its pixels, of the squared difference
/// between the predicted arrows and the arrows recomputed from the mask itself.
pub fn flow_error(predicted: &[[f32; 2]], own: &[[f32; 2]]) -> f32 {
    todo!("C5: flow_error")
}
