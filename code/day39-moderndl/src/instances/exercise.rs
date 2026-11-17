// Coding exercises for "From masks to objects". Replace each `todo!()` with your code,
// then check it with `cargo test -p day39 instances::` and look at `cargo run -p day39 -- instances`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;
use std::collections::HashMap;

/// The three classes of the boundary route.
pub const BACKGROUND: u8 = 0;
pub const INTERIOR: u8 = 1;
pub const BOUNDARY: u8 = 2;

/// The 4 neighbours of `(x, y)` that lie inside a `w × h` image.
pub fn neighbours4(x: usize, y: usize, w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if x > 0 {
        out.push((x - 1, y));
    }
    if x + 1 < w {
        out.push((x + 1, y));
    }
    if y > 0 {
        out.push((x, y - 1));
    }
    if y + 1 < h {
        out.push((x, y + 1));
    }
    out
}

/// Connected components of a mask with 4-connectivity (the labelling day): every connected
/// piece of `true` pixels gets its own number 1, 2, ...; background 0.
pub fn label_components(mask: &Image<bool>) -> Image<u32> {
    let (w, h) = (mask.width(), mask.height());
    let mut labels = Image::new(w, h, 0u32);
    let mut next = 0;
    for (x, y) in mask.coords() {
        if mask[(x, y)] && labels[(x, y)] == 0 {
            next += 1;
            labels[(x, y)] = next;
            let mut stack = vec![(x, y)];
            while let Some((px, py)) = stack.pop() {
                for (qx, qy) in neighbours4(px, py, w, h) {
                    if mask[(qx, qy)] && labels[(qx, qy)] == 0 {
                        labels[(qx, qy)] = next;
                        stack.push((qx, qy));
                    }
                }
            }
        }
    }
    labels
}

/// C1. The boundary route: label the connected INTERIOR pixels (4-connectivity), then grow
/// every object over the BOUNDARY pixels: a boundary pixel takes the label of a labelled
/// 4-neighbour, repeated until nothing changes. Background stays 0.
///
/// Hint: `label_components(&classes.map(|c| c == INTERIOR))` labels the interiors. Then repeat
/// passes over the image until a pass changes nothing; read the labels of the previous pass (a
/// `clone()`) so that a label grows by one pixel per pass.
pub fn boundary_route(classes: &Image<u8>) -> Image<u32> {
    todo!("C1: boundary_route")
}

/// C2. The distance route: watershed by flooding from the top, with h-maxima markers.
/// Visit the pixels with `dist > mask_threshold` from the highest value down. A pixel with
/// no labelled 4-neighbour starts a new object (a peak). A pixel that touches one object
/// joins it. A pixel that touches two objects joins the one with the higher peak; and if the
/// lower peak is less than `h` above this pixel (its "dynamic" is below `h`), the two objects
/// are merged: that peak was not a real marker.
///
/// Hint: collect the pixels above the threshold and sort them, highest first:
/// `order.sort_by(|&a, &b| dist[b].total_cmp(&dist[a]))`. Keep the peak height of every object
/// in a `Vec<f32>`, and for merged objects a `parent` list: object `l` now belongs to
/// `parent[l]` (follow it until `parent[l] == l`). At the end give every pixel the number of
/// its final object.
pub fn distance_route(dist: &Image<f32>, mask_threshold: f32, h: f32) -> Image<u32> {
    todo!("C2: distance_route")
}

/// C3. Match predicted objects to true objects one to one at IoU > 0.5 (above 0.5 a match
/// is unique). Returns `(tp, fp, fn, ious)`: the counts and the IoU of every matched pair.
///
/// Hint: one pass over the pixels counts the area of every true object, of every predicted
/// object, and the overlap of every pair `(t, p)` (three `HashMap`s). Then IoU = overlap /
/// (area_t + area_p − overlap) for every overlapping pair.
pub fn match_objects(truth: &Image<u32>, pred: &Image<u32>) -> (usize, usize, usize, Vec<f32>) {
    todo!("C3: match_objects")
}

/// C4. Panoptic quality from the matched IoUs and the unmatched counts:
/// `SQ = mean IoU of the matches`, `RQ = TP / (TP + FP/2 + FN/2)`, `PQ = SQ · RQ`.
/// Returns `(sq, rq, pq)`; SQ is 0 if nothing matched.
pub fn panoptic_quality(ious: &[f32], fp: usize, fn_: usize) -> (f32, f32, f32) {
    todo!("C4: panoptic_quality")
}
