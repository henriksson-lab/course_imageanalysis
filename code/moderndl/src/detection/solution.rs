// Model solution for the coding exercises of "Finding objects as boxes".

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
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let w = (a.x1.min(b.x1) - a.x0.max(b.x0)).max(0.0);
    let h = (a.y1.min(b.y1) - a.y0.max(b.y0)).max(0.0);
    let inter = w * h;
    inter / (a.area() + b.area() - inter)
}

/// C2. Greedy non-maximum suppression: sort by score (highest first); keep the best
/// remaining box; delete every remaining box with IoU > `t` with it; repeat.
/// Returns the indices (into `boxes`) of the kept boxes, best first.
pub fn nms(boxes: &[Detection], t: f32) -> Vec<usize> {
    let mut remaining: Vec<usize> = (0..boxes.len()).collect();
    remaining.sort_by(|&i, &j| boxes[j].score.total_cmp(&boxes[i].score));
    let mut kept = Vec::new();
    while !remaining.is_empty() {
        let best = remaining.remove(0);
        kept.push(best);
        remaining.retain(|&i| iou(&boxes[best], &boxes[i]) <= t);
    }
    kept
}

/// C3. The box around every labelled object: for label `l` (1, 2, ...), the smallest box
/// that holds all its pixels. A pixel `(x, y)` covers `[x, x + 1] × [y, y + 1]`. Score 1.
/// Entry `l − 1` of the result is the box of label `l`.
pub fn boxes_from_labels(labels: &Image<u32>) -> Vec<Detection> {
    let n = labels.pixels().iter().cloned().max().unwrap_or(0) as usize;
    let mut boxes = vec![Detection::new(f32::MAX, f32::MIN, f32::MAX, f32::MIN, 1.0); n];
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        if l > 0 {
            let b = &mut boxes[l - 1];
            b.x0 = b.x0.min(x as f32);
            b.x1 = b.x1.max(x as f32 + 1.0);
            b.y0 = b.y0.min(y as f32);
            b.y1 = b.y1.max(y as f32 + 1.0);
        }
    }
    boxes
}

/// C4. Score detections against the true boxes: walk through the detections from the
/// highest score down; each is a true positive if its best IoU with a true box that is not
/// matched yet is at least `t`, and that true box is then matched; else a false positive.
/// True boxes left over are false negatives. Returns `(tp, fp, fn)`.
pub fn count_matches(detections: &[Detection], truth: &[Detection], t: f32) -> (usize, usize, usize) {
    let mut order: Vec<usize> = (0..detections.len()).collect();
    order.sort_by(|&i, &j| detections[j].score.total_cmp(&detections[i].score));
    let mut matched = vec![false; truth.len()];
    let (mut tp, mut fp) = (0, 0);
    for i in order {
        let mut best: Option<(usize, f32)> = None;
        for (k, tb) in truth.iter().enumerate() {
            let v = iou(&detections[i], tb);
            let better = match best {
                None => true,
                Some((_, best_iou)) => v > best_iou,
            };
            if !matched[k] && v >= t && better {
                best = Some((k, v));
            }
        }
        match best {
            Some((k, _)) => {
                matched[k] = true;
                tp += 1;
            }
            None => fp += 1,
        }
    }
    (tp, fp, truth.len() - tp)
}
