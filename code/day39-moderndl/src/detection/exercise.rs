// Coding exercises for "Finding objects as boxes". Replace each `todo!()` with your code,
// then check it with `cargo test -p day39 detection::` and look at `cargo run -p day39 -- detection`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// A box `[x0, x1] × [y0, y1]` with a score (how sure the detector is).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
}

impl Detection {
    pub fn new(x0: f32, x1: f32, y0: f32, y1: f32, score: f32) -> Detection {
        Detection { x0, y0, x1, y1, score }
    }

    pub fn area(&self) -> f32 {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }
}

/// C1. Intersection over union of two boxes. The intersection of two boxes is a box:
/// width `max(0, min(a.x1, b.x1) − max(a.x0, b.x0))`, and the same for the height.
///
/// Hint: `a.x1.min(b.x1)` is the smaller of two numbers, `.max(0.0)` cuts negative widths to 0.
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    todo!("C1: iou")
}

/// C2. Greedy non-maximum suppression: sort by score (highest first); keep the best
/// remaining box; delete every remaining box with IoU > `t` with it; repeat.
/// Returns the indices (into `boxes`) of the kept boxes, best first.
///
/// Hint: make a list of the indices `0..boxes.len()`, sort it by score, highest first:
/// `sort_by(|&i, &j| boxes[j].score.total_cmp(&boxes[i].score))`. Then `remaining.remove(0)`
/// takes the best, and `remaining.retain(|&i| ...)` keeps only the boxes that do not overlap it
/// too much.
pub fn nms(boxes: &[Detection], t: f32) -> Vec<usize> {
    todo!("C2: nms")
}

/// C3. The box around every labelled object: for label `l` (1, 2, ...), the smallest box
/// that holds all its pixels. A pixel `(x, y)` covers `[x, x + 1] × [y, y + 1]`. Score 1.
/// Entry `l − 1` of the result is the box of label `l`.
///
/// Hint: start every box "empty": `x0 = f32::MAX`, `x1 = f32::MIN` (and the same for y), and
/// widen it with every pixel of its label.
pub fn boxes_from_labels(labels: &Image<u32>) -> Vec<Detection> {
    todo!("C3: boxes_from_labels")
}

/// C4. Score detections against the true boxes: walk through the detections from the
/// highest score down; each is a true positive if its best IoU with a true box that is not
/// matched yet is at least `t`, and that true box is then matched; else a false positive.
/// True boxes left over are false negatives. Returns `(tp, fp, fn)`.
///
/// Hint: sort the indices by score as in C2; keep a `Vec<bool>` of which true boxes are matched
/// already.
pub fn count_matches(detections: &[Detection], truth: &[Detection], t: f32) -> (usize, usize, usize) {
    todo!("C4: count_matches")
}
