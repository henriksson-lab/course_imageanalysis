// Coding exercises for "Matching objects: IoU, F1 and average precision". Replace each `todo!()` with your code,
// then check it with `cargo test -p evaluation object_eval::` and look at `cargo run -p evaluation -- object-eval`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1. The IoU table of two label images (0 = background, objects 1, 2, ...): row `i` is
/// true object `i + 1`, column `j` predicted object `j + 1`. One pass over the pixels counts
/// the area of every object and the overlap of every pair; then IoU = overlap / (area of
/// the true object + area of the predicted one - overlap).
///
/// Hint: the number of objects is the largest label. Keep `area_true`, `area_pred` (one
/// count per object) and `overlap` (one count per pair) as vectors; label `l` is at index `l - 1`.
pub fn iou_table(truth: &Image<u32>, pred: &Image<u32>) -> Vec<Vec<f32>> {
    todo!("C1: iou_table")
}

/// C2. Greedy matching at the threshold `tau`: of all pairs with IoU >= tau (and > 0), take
/// the largest IoU first, then the largest among the objects still free, and so on.
/// Returns the matched pairs `(row, column)` of the table.
///
/// Hint: collect `(iou, i, j)` for the allowed pairs, sort them with
/// `pairs.sort_by(|a, b| b.0.total_cmp(&a.0))` (largest first), and keep one `bool` per row and
/// one per column that says whether it is taken.
pub fn match_greedy(iou: &[Vec<f32>], tau: f32) -> Vec<(usize, usize)> {
    todo!("C2: match_greedy")
}

/// Precision, recall, F1 and matching accuracy of objects.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scores {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub accuracy: f32,
}

/// C3. The scores from the numbers of true and predicted objects and of matches:
/// FP = predicted - TP, FN = true - TP, precision = TP/(TP+FP), recall = TP/(TP+FN),
/// F1 = 2TP/(2TP+FP+FN), matching accuracy = TP/(TP+FP+FN). (0 where a denominator is 0.)
///
/// Hint: a small helper `ratio(a, b)` that returns 0 when `b` is 0 keeps the four formulas
/// short.
pub fn scores(n_true: usize, n_pred: usize, tp: usize) -> Scores {
    todo!("C3: scores")
}

/// C4. Average precision: the predictions are sorted by score and marked TP or FP
/// (`is_tp`). Walk down the list keeping recall and precision after every prediction;
/// make the precision non-increasing from the right (each value becomes the largest
/// precision at this or a later step); sum (recall step) × (that precision).
///
/// Hint: three loops: (1) recall and precision after each prediction; (2) from the second
/// last to the first, `precision[k] = max(precision[k], precision[k + 1])`; (3) add up
/// `(recall[k] - recall[k - 1]) * precision[k]`, with recall 0 before the first.
pub fn average_precision(is_tp: &[bool], n_true: usize) -> f32 {
    todo!("C4: average_precision")
}
