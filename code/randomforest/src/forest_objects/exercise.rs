// Coding exercises for "Forests on feature tables, and feature importance". Replace each
// `todo!()` with your code, then check it with `cargo test -p randomforest forest_objects::` and look at
// `cargo run -p randomforest -- forest-objects`.
#![allow(unused_variables, unused_imports)]

use crate::forest_objects::{Forest, Tree};
use imgcourse::Rng;

/// C1. The Gini impurity `1 − Σ p_k²` from (possibly weighted) class counts; 0 if empty.
pub fn gini(counts: &[f32]) -> f32 {
    todo!("C1: gini")
}

/// C2. The impurity importance (mean decrease in impurity) of every feature: in every tree,
/// add `(n_t / n) ΔG_t` of every split to the feature it splits on (`n_t` = points in the
/// node, `n` = points at the root, `ΔG_t` = the Gini decrease of the split); then average
/// over the trees.
///
/// Hint: a recursive function of your own that visits every node: for a
/// `Tree::Split { feature, left, right, counts, .. }` it adds the weighted decrease and calls
/// itself on `left` and `right`. `node.counts()` gives the class counts of any node; the sum of
/// the counts is the number of points `n_t`.
pub fn impurity_importance(trees: &[Tree], n_features: usize) -> Vec<f32> {
    todo!("C2: impurity_importance")
}

/// C3. The permutation importance of a group of columns (one column, or several shuffled
/// together): shuffle the rows of these columns with one random permutation, and measure how
/// much the out-of-bag error rises. Averaged over `repeats` shuffles.
///
/// Hint: `forest.oob_error(points, labels)` gives the error. A random permutation: start from
/// 0, 1, ..., n − 1 and, for i from n − 1 down to 1, swap place i with place `rng.below(i + 1)`.
/// Copy the points (`points.to_vec()`) and replace the chosen columns of row i by those of
/// row perm[i].
pub fn permutation_importance(
    forest: &Forest,
    points: &[Vec<f32>],
    labels: &[usize],
    columns: &[usize],
    repeats: usize,
    rng: &mut Rng,
) -> f32 {
    todo!("C3: permutation_importance")
}

/// C4. The (Pearson) correlation of two columns: their covariance divided by the product of
/// their standard deviations, between −1 and 1.
pub fn correlation(a: &[f32], b: &[f32]) -> f32 {
    todo!("C4: correlation")
}
