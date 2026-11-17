// Coding exercises for "Detecting objects in one pass: YOLO". Replace each `todo!()` with your code,
// then check it with `cargo test -p day40 yolo::` and look at `cargo run -p day40 -- yolo`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// The number of bins of a distance distribution: distances 0, 1, ..., 15 strides.
pub const BINS: usize = 16;

/// A box `(x0, y0)–(x1, y1)` in pixels, with a score and a class (0, 1, ...).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub score: f32,
    pub class: usize,
}

impl Detection {
    pub fn area(&self) -> f32 {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }
}

/// Intersection over union of two boxes (written on the detection day).
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let w = (a.x1.min(b.x1) - a.x0.max(b.x0)).max(0.0);
    let h = (a.y1.min(b.y1) - a.y0.max(b.y0)).max(0.0);
    let inter = w * h;
    if inter == 0.0 { 0.0 } else { inter / (a.area() + b.area() - inter) }
}

/// The logistic function, which turns a class logit into a probability.
pub fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit).exp())
}

/// C1. The training target of one distance `y` (in strides, clamped to 0 ... 15): all of the
/// probability on the two bins around it, `k = ⌊y⌋` and `k + 1`, shared so that the mean
/// of the distribution is exactly `y`: `p_k = k + 1 − y`, `p_{k+1} = y − k`.
///
/// Hint: when `y` is exactly 15, put everything on bin 15.
pub fn distance_target(y: f32) -> [f32; BINS] {
    todo!("C1: distance_target")
}

/// C2. Decode one output cell: the cell in row `row`, column `col` of the grid of stride
/// `stride` has its centre at `(stride (col + ½), stride (row + ½))`. Each of the four
/// distributions (left, top, right, bottom) gives a distance in strides: its mean
/// `Σ_k k p_k`. The box reaches that many strides from the centre on each side.
pub fn decode_cell(stride: f32, row: usize, col: usize, dists: &[[f32; BINS]; 4], score: f32, class: usize) -> Detection {
    todo!("C2: decode_cell")
}

/// C3. Greedy NMS with classes. Boxes with a score below `score_threshold` are dropped.
/// Visit the others from the highest score down; keep a box unless a kept box overlaps it
/// with IoU above `t`. Class-aware (`class_aware = true`): only kept boxes of the **same**
/// class count; class-agnostic: all kept boxes count. Returns the kept indices, best first.
pub fn nms_classes(boxes: &[Detection], score_threshold: f32, t: f32, class_aware: bool) -> Vec<usize> {
    todo!("C3: nms_classes")
}

/// C4. The peaks of a centre heat map: the pixels whose value is above `threshold` and at
/// least as large as every value in the `(2 radius + 1) × (2 radius + 1)` window around them
/// (pixels outside the image are ignored). Returns them as `(x, y)`, row by row.
pub fn heatmap_peaks(map: &Image<f32>, radius: usize, threshold: f32) -> Vec<(usize, usize)> {
    todo!("C4: heatmap_peaks")
}
