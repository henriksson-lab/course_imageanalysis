// Coding exercises for "Crowded spots and evaluating a spot detector".
// Replace each `todo!()` with your code, then check it with `cargo test -p pointdetection spot_crowd::`
// and look at `cargo run -p pointdetection -- spot-crowd`.
#![allow(unused_variables, unused_imports)]

/// C1. The number of local maxima of a 1D profile: samples larger than both neighbours (the
/// first and last sample are not counted).
///
/// Hint: loop `i` over `1..profile.len() - 1`.
pub fn count_peaks(profile: &[f32]) -> usize {
    todo!("C1: count_peaks")
}

/// The distance between two points.
pub fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// C2. Greedy matching: of all truth–detection pairs within `gate`, take the closest first,
/// then the next closest whose truth and detection are both still free, and so on.
/// Returns the pairs `(truth index, detection index)`.
///
/// Hint: collect `(distance, i, j)` for all pairs within the gate, sort them with
/// `sort_by(|a, b| a.0.total_cmp(&b.0))`, and keep two lists of booleans: which truths and which
/// detections are already used.
pub fn match_greedy(truth: &[[f32; 2]], det: &[[f32; 2]], gate: f32) -> Vec<(usize, usize)> {
    todo!("C2: match_greedy")
}

/// C3. Optimal matching for a small group: of all one-to-one matchings with pairs within `gate`,
/// the one with the **most** pairs, and among those the smallest total distance. Tries every
/// possibility (truth by truth: unmatched, or with any free detection in its gate), so use it
/// only on small groups (`sim::match_in_groups` splits a whole image into such groups).
///
/// Hint: write an inner recursive function
/// `search(i, ..., used: &mut [bool]) -> (count, total distance, pairs)` that returns the best
/// result for the truths `i..`. It tries "truth i unmatched" (`search(i + 1, ...)`) and, for every
/// free detection `j` in the gate of truth `i`, marks `j` used, calls `search(i + 1, ...)`, adds
/// 1 to the count and the distance to the total, and unmarks `j`. Keep the best: more pairs first,
/// then the smaller total.
pub fn match_optimal(truth: &[[f32; 2]], det: &[[f32; 2]], gate: f32) -> Vec<(usize, usize)> {
    todo!("C3: match_optimal")
}

/// The scores of a matching.
#[derive(Clone, Copy, Debug)]
pub struct Scores {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub jaccard: f32,
    /// root mean square distance of the matched pairs
    pub rmse: f32,
}

/// C4. TP, FP, FN, precision, recall, F1, Jaccard and the RMSE of the matched distances, from the
/// pairs of a matching.
///
/// Hint: TP is `pairs.len()`, FP and FN are what is left over; guard every division
/// against 0.
pub fn scores(truth: &[[f32; 2]], det: &[[f32; 2]], pairs: &[(usize, usize)]) -> Scores {
    todo!("C4: scores")
}

/// C5. The precision–recall curve of detections ranked by score (`is_tp[k]`: is the k-th
/// strongest detection matched?), with `n_truth` true spots. Returns the points
/// `(precision, recall)` after each detection, and the average precision: precision made
/// non-increasing from the right, then summed over the recall steps (precision × step).
///
/// Hint: first the curve; then a copy of the precisions, made non-increasing by walking
/// from the right with `p[k] = p[k].max(p[k + 1])`; then sum `p[k] × (recall[k] − recall[k − 1])`.
pub fn pr_curve(is_tp: &[bool], n_truth: usize) -> (Vec<(f32, f32)>, f32) {
    todo!("C5: pr_curve")
}
