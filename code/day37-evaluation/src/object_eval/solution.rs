// Model solution for the coding exercises of "Matching objects: IoU, F1 and average precision".

use imgcourse::Image;

/// C1. The IoU table of two label images (0 = background, objects 1, 2, ...): row `i` is
/// true object `i + 1`, column `j` predicted object `j + 1`. One pass over the pixels counts
/// the area of every object and the overlap of every pair; then IoU = overlap / (area of
/// the true object + area of the predicted one - overlap).
pub fn iou_table(truth: &Image<u32>, pred: &Image<u32>) -> Vec<Vec<f32>> {
    let n_true = truth.pixels().iter().copied().max().unwrap_or(0) as usize;
    let n_pred = pred.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut area_true = vec![0usize; n_true];
    let mut area_pred = vec![0usize; n_pred];
    let mut overlap = vec![vec![0usize; n_pred]; n_true];
    for (x, y) in truth.coords() {
        let (g, p) = (truth[(x, y)] as usize, pred[(x, y)] as usize);
        if g > 0 {
            area_true[g - 1] += 1;
        }
        if p > 0 {
            area_pred[p - 1] += 1;
        }
        if g > 0 && p > 0 {
            overlap[g - 1][p - 1] += 1;
        }
    }
    let mut iou = vec![vec![0.0; n_pred]; n_true];
    for i in 0..n_true {
        for j in 0..n_pred {
            let union = area_true[i] + area_pred[j] - overlap[i][j];
            if overlap[i][j] > 0 {
                iou[i][j] = overlap[i][j] as f32 / union as f32;
            }
        }
    }
    iou
}

/// C2. Greedy matching at the threshold `tau`: of all pairs with IoU >= tau (and > 0), take
/// the largest IoU first, then the largest among the objects still free, and so on.
/// Returns the matched pairs `(row, column)` of the table.
pub fn match_greedy(iou: &[Vec<f32>], tau: f32) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..iou.len() {
        for j in 0..iou[i].len() {
            if iou[i][j] >= tau && iou[i][j] > 0.0 {
                pairs.push((iou[i][j], i, j));
            }
        }
    }
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0)); // largest IoU first
    let n_pred = iou.first().map_or(0, |row| row.len());
    let mut true_taken = vec![false; iou.len()];
    let mut pred_taken = vec![false; n_pred];
    let mut matches = Vec::new();
    for (_, i, j) in pairs {
        if !true_taken[i] && !pred_taken[j] {
            true_taken[i] = true;
            pred_taken[j] = true;
            matches.push((i, j));
        }
    }
    matches
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
pub fn scores(n_true: usize, n_pred: usize, tp: usize) -> Scores {
    let (fp, fn_) = (n_pred - tp, n_true - tp);
    let ratio = |a: usize, b: usize| if b == 0 { 0.0 } else { a as f32 / b as f32 };
    Scores {
        tp,
        fp,
        fn_,
        precision: ratio(tp, tp + fp),
        recall: ratio(tp, tp + fn_),
        f1: ratio(2 * tp, 2 * tp + fp + fn_),
        accuracy: ratio(tp, tp + fp + fn_),
    }
}

/// C4. Average precision: the predictions are sorted by score and marked TP or FP
/// (`is_tp`). Walk down the list keeping recall and precision after every prediction;
/// make the precision non-increasing from the right (each value becomes the largest
/// precision at this or a later step); sum (recall step) × (that precision).
pub fn average_precision(is_tp: &[bool], n_true: usize) -> f32 {
    let mut recall = Vec::new();
    let mut precision = Vec::new();
    let mut tp = 0;
    for (k, &hit) in is_tp.iter().enumerate() {
        if hit {
            tp += 1;
        }
        recall.push(tp as f32 / n_true as f32);
        precision.push(tp as f32 / (k + 1) as f32);
    }
    // from the right: the best precision at this or a higher recall
    for k in (0..precision.len().saturating_sub(1)).rev() {
        precision[k] = precision[k].max(precision[k + 1]);
    }
    let mut ap = 0.0;
    let mut last_recall = 0.0;
    for k in 0..recall.len() {
        ap += (recall[k] - last_recall) * precision[k];
        last_recall = recall[k];
    }
    ap
}
