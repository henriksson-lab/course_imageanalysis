// Model solution for the coding exercises of "Detecting objects in one pass: YOLO".

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
pub fn distance_target(y: f32) -> [f32; BINS] {
    let y = y.clamp(0.0, (BINS - 1) as f32);
    let mut p = [0.0; BINS];
    let k = y.floor() as usize;
    if k == BINS - 1 {
        p[k] = 1.0;
    } else {
        p[k] = k as f32 + 1.0 - y;
        p[k + 1] = y - k as f32;
    }
    p
}

/// C2. Decode one output cell: the cell in row `row`, column `col` of the grid of stride
/// `stride` has its centre at `(stride (col + ½), stride (row + ½))`. Each of the four
/// distributions (left, top, right, bottom) gives a distance in strides: its mean
/// `Σ_k k p_k`. The box reaches that many strides from the centre on each side.
pub fn decode_cell(stride: f32, row: usize, col: usize, dists: &[[f32; BINS]; 4], score: f32, class: usize) -> Detection {
    let mut d = [0.0; 4];
    for side in 0..4 {
        for k in 0..BINS {
            d[side] += k as f32 * dists[side][k];
        }
    }
    let (cx, cy) = (stride * (col as f32 + 0.5), stride * (row as f32 + 0.5));
    Detection { x0: cx - stride * d[0], y0: cy - stride * d[1], x1: cx + stride * d[2], y1: cy + stride * d[3], score, class }
}

/// C3. Greedy NMS with classes. Boxes with a score below `score_threshold` are dropped.
/// Visit the others from the highest score down; keep a box unless a kept box overlaps it
/// with IoU above `t`. Class-aware (`class_aware = true`): only kept boxes of the **same**
/// class count; class-agnostic: all kept boxes count. Returns the kept indices, best first.
pub fn nms_classes(boxes: &[Detection], score_threshold: f32, t: f32, class_aware: bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).filter(|&i| boxes[i].score >= score_threshold).collect();
    order.sort_by(|&i, &j| boxes[j].score.total_cmp(&boxes[i].score));
    let mut kept: Vec<usize> = Vec::new();
    for i in order {
        let suppressed = kept.iter().any(|&k| (!class_aware || boxes[k].class == boxes[i].class) && iou(&boxes[k], &boxes[i]) > t);
        if !suppressed {
            kept.push(i);
        }
    }
    kept
}

/// C4. The peaks of a centre heat map: the pixels whose value is above `threshold` and at
/// least as large as every value in the `(2 radius + 1) × (2 radius + 1)` window around them
/// (pixels outside the image are ignored). Returns them as `(x, y)`, row by row.
pub fn heatmap_peaks(map: &Image<f32>, radius: usize, threshold: f32) -> Vec<(usize, usize)> {
    let r = radius as isize;
    let mut peaks = Vec::new();
    for (x, y) in map.coords() {
        let v = map[(x, y)];
        if v <= threshold {
            continue;
        }
        let mut is_peak = true;
        for dy in -r..=r {
            for dx in -r..=r {
                let (u, w) = (x as isize + dx, y as isize + dy);
                if map.contains(u, w) && map[(u as usize, w as usize)] > v {
                    is_peak = false;
                }
            }
        }
        if is_peak {
            peaks.push((x, y));
        }
    }
    peaks
}
