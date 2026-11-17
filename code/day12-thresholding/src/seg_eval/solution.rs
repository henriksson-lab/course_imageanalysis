// Model solution for the coding exercises of "Is the segmentation good? Evaluation against
// ground truth".

use imgcourse::Image;

/// The four counts of the confusion matrix (`fn_` because `fn` is a word of Rust).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counts {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub tn: usize,
}

/// The bin of a grey value `v` in [0, 1] for `bins` equal bins (as in the histogram unit, given
/// in `src/common.rs`): `floor(v · bins)`, kept inside `0..bins`.
pub use crate::common::bin_of;

/// C1. The confusion matrix of a predicted mask against the ground truth (true = object).
pub fn confusion(pred: &Image<bool>, truth: &Image<bool>) -> Counts {
    let mut c = Counts { tp: 0, fp: 0, fn_: 0, tn: 0 };
    for (x, y) in pred.coords() {
        match (pred[(x, y)], truth[(x, y)]) {
            (true, true) => c.tp += 1,
            (true, false) => c.fp += 1,
            (false, true) => c.fn_ += 1,
            (false, false) => c.tn += 1,
        }
    }
    c
}

/// C2. The measures of the slide "Rates from the four counts" and "Dice and Jaccard".
impl Counts {
    pub fn precision(&self) -> f32 {
        self.tp as f32 / (self.tp + self.fp) as f32
    }

    /// Recall = true-positive rate (TPR).
    pub fn recall(&self) -> f32 {
        self.tp as f32 / (self.tp + self.fn_) as f32
    }

    /// False-positive rate.
    pub fn fpr(&self) -> f32 {
        self.fp as f32 / (self.fp + self.tn) as f32
    }

    pub fn accuracy(&self) -> f32 {
        (self.tp + self.tn) as f32 / (self.tp + self.fp + self.fn_ + self.tn) as f32
    }

    pub fn dice(&self) -> f32 {
        2.0 * self.tp as f32 / (2 * self.tp + self.fp + self.fn_) as f32
    }

    pub fn jaccard(&self) -> f32 {
        self.tp as f32 / (self.tp + self.fp + self.fn_) as f32
    }
}

/// C3. The ROC curve from two histograms (slide "The math: ROC from two histograms"):
/// for every threshold `t = 0, 1, ..., bins − 1` (object where the bin of the grey value is
/// above `t`) the point `(FPR(t), TPR(t))`, with TPR(t) = fraction of object pixels above t
/// and FPR(t) = fraction of background pixels above t.
pub fn roc(img: &Image<f32>, truth: &Image<bool>, bins: usize) -> Vec<(f32, f32)> {
    // the histograms of the object pixels and of the background pixels
    let mut h_obj = vec![0usize; bins];
    let mut h_bg = vec![0usize; bins];
    for (x, y) in img.coords() {
        let b = bin_of(img[(x, y)], bins);
        if truth[(x, y)] {
            h_obj[b] += 1;
        } else {
            h_bg[b] += 1;
        }
    }
    let n_obj: usize = h_obj.iter().sum();
    let n_bg: usize = h_bg.iter().sum();
    // count down from "everything above" as t rises
    let (mut above_obj, mut above_bg) = (n_obj, n_bg);
    let mut points = Vec::with_capacity(bins);
    for t in 0..bins {
        above_obj -= h_obj[t];
        above_bg -= h_bg[t];
        points.push((above_bg as f32 / n_bg as f32, above_obj as f32 / n_obj as f32));
    }
    points
}

/// C4. The area under the ROC curve: add the corners (1, 1) and (0, 0) and add up the
/// trapezoids between neighbouring points. The points are in the order of `roc`
/// (FPR falling).
pub fn auc(points: &[(f32, f32)]) -> f32 {
    let mut all = vec![(1.0, 1.0)];
    all.extend_from_slice(points);
    all.push((0.0, 0.0));
    let mut area = 0.0;
    for k in 0..all.len() - 1 {
        let (f0, t0) = all[k];
        let (f1, t1) = all[k + 1];
        area += (f0 - f1) * (t0 + t1) / 2.0;
    }
    area
}

/// C5. The threshold with the best Dice, and that Dice: try every `t` from 0 to `bins − 1`
/// (object where the bin is above `t`). On a tie the smallest `t` wins.
pub fn best_dice(img: &Image<f32>, truth: &Image<bool>, bins: usize) -> (usize, f32) {
    let mut best = (0, -1.0);
    for t in 0..bins {
        let pred = img.map(|v| bin_of(v, bins) > t);
        let d = confusion(&pred, truth).dice();
        if d > best.1 {
            best = (t, d);
        }
    }
    best
}
