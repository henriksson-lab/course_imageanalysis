// Coding exercises for "Few labels: annotation and the human in the loop". Replace each `todo!()`
// with your code, then check it with `cargo test -p finetuning few_labels::` and look at
// `cargo run -p finetuning -- few-labels`.
#![allow(unused_variables, unused_imports)]

pub use super::model::*;

/// C1. Uncertainty of a two-class prediction p = p(class 1): 1 when p = 0.5, 0 when p is 0 or 1
/// (one minus the margin |2p − 1|).
pub fn uncertainty(p: f32) -> f32 {
    todo!("C1: uncertainty")
}

/// C2. The unlabelled cell whose prediction is most uncertain (`None` if all are labelled).
/// Ties go to the lowest index.
///
/// Hint: loop over all cells, skip `labelled[j]`, keep the index with the largest
/// `uncertainty(probs[j])` (replace only when strictly larger).
pub fn most_uncertain(probs: &[f32], labelled: &[bool]) -> Option<usize> {
    todo!("C2: most_uncertain")
}

/// C3. The unlabelled cell farthest from its nearest labelled cell (`None` if all are labelled):
/// the most different from everything labelled so far.
///
/// Hint: for each unlabelled cell j, the distance to its nearest labelled cell is the minimum
/// over labelled i of the distance between (x, y) of i and j; keep the j where it is largest.
pub fn most_different(cells: &[Cell], labelled: &[bool]) -> Option<usize> {
    todo!("C3: most_different")
}

/// C4. Binary entropy of a prediction in bits: −p log₂ p − (1 − p) log₂ (1 − p) (0 at p = 0 or 1).
///
/// Hint: `q.log2()`; treat 0 · log 0 as 0.
pub fn entropy(p: f32) -> f32 {
    todo!("C4: entropy")
}
