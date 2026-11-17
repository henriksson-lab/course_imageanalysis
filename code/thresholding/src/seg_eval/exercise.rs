// Coding exercises for "Is the segmentation good? Evaluation against ground truth". Replace
// each `todo!()` with your code, then check it with `cargo test -p thresholding seg_eval::` and look at
// `cargo run -p thresholding -- seg-eval`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `match (pred[(x, y)], truth[(x, y)]) { (true, true) => ..., ... }`
pub fn confusion(pred: &Image<bool>, truth: &Image<bool>) -> Counts {
    todo!("C1: confusion")
}

/// C2. The measures of the slide "Rates from the four counts" and "Dice and Jaccard".
///
/// Hint: convert the counts to `f32` before dividing: `self.tp as f32 / (self.tp + self.fp) as f32`.
impl Counts {
    pub fn precision(&self) -> f32 {
        todo!("C2: precision")
    }

    /// Recall = true-positive rate (TPR).
    pub fn recall(&self) -> f32 {
        todo!("C2: recall")
    }

    /// False-positive rate.
    pub fn fpr(&self) -> f32 {
        todo!("C2: fpr")
    }

    pub fn accuracy(&self) -> f32 {
        todo!("C2: accuracy")
    }

    pub fn dice(&self) -> f32 {
        todo!("C2: dice")
    }

    pub fn jaccard(&self) -> f32 {
        todo!("C2: jaccard")
    }
}

/// C3. The ROC curve from two histograms (slide "The math: ROC from two histograms"):
/// for every threshold `t = 0, 1, ..., bins − 1` (object where the bin of the grey value is
/// above `t`) the point `(FPR(t), TPR(t))`, with TPR(t) = fraction of object pixels above t
/// and FPR(t) = fraction of background pixels above t.
///
/// Hint: one pass over the image fills a histogram of the object pixels and one of the
/// background pixels. Then go through `t = 0, 1, ...` and subtract `h[t]` from the number
/// of pixels above: no second pass over the image.
pub fn roc(img: &Image<f32>, truth: &Image<bool>, bins: usize) -> Vec<(f32, f32)> {
    todo!("C3: roc")
}

/// C4. The area under the ROC curve: add the corners (1, 1) and (0, 0) and add up the
/// trapezoids between neighbouring points. The points are in the order of `roc`
/// (FPR falling).
///
/// Hint: a trapezoid between `(f0, t0)` and `(f1, t1)` has the area `(f0 − f1) · (t0 + t1) / 2`.
pub fn auc(points: &[(f32, f32)]) -> f32 {
    todo!("C4: auc")
}

/// C5. The threshold with the best Dice, and that Dice: try every `t` from 0 to `bins − 1`
/// (object where the bin is above `t`). On a tie the smallest `t` wins.
pub fn best_dice(img: &Image<f32>, truth: &Image<bool>, bins: usize) -> (usize, f32) {
    todo!("C5: best_dice")
}
