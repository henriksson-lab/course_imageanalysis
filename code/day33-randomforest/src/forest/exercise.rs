// Coding exercises for "Bagging and random forests". Replace each `todo!()` with your code, then
// check it with `cargo test -p day33 forest::` and look at `cargo run -p day33 -- forest`.
#![allow(unused_variables, unused_imports)]

use crate::forest::{grow_tree, Tree};
use imgcourse::Rng;

/// C1a. A bootstrap sample: `n` draws from the point numbers 0 .. n − 1, with replacement.
///
/// Hint: `rng.below(n)` is a random whole number in 0 .. n − 1.
pub fn bootstrap(n: usize, rng: &mut Rng) -> Vec<usize> {
    todo!("C1a: bootstrap")
}

/// C1b. Which of the `n` points are in the sample (`true`) and which are out of bag (`false`).
pub fn in_bag(sample: &[usize], n: usize) -> Vec<bool> {
    todo!("C1b: in_bag")
}

/// C2. A random subset of `m` of the features 0 .. d − 1, all different, in increasing order.
///
/// Hint: start from the list 0, 1, ..., d − 1. For i = 0 .. m − 1, swap place i with a random
/// place from i to d − 1 (`i + rng.below(d − i)`). The first m places are then a random subset.
pub fn choose_features(d: usize, m: usize, rng: &mut Rng) -> Vec<usize> {
    todo!("C2: choose_features")
}

/// A trained forest: the trees, and for every tree which training points were in its
/// bootstrap sample (`in_bag[tree][point]`).
pub struct Forest {
    pub trees: Vec<Tree>,
    pub in_bag: Vec<Vec<bool>>,
}

/// C3. Train a random forest: for every tree, draw a bootstrap sample (C1) and grow a full
/// tree on it (`grow_tree`), trying `m` random features at every split (C2).
/// With `m` = the number of features, this is bagging.
///
/// Hint: `grow_tree(points, labels, &sample, n_classes, &mut || choose_features(d, m, rng))`
/// grows one tree; the closure hands it new candidate features at every split.
pub fn train_forest(points: &[Vec<f32>], labels: &[usize], n_classes: usize, n_trees: usize, m: usize, rng: &mut Rng) -> Forest {
    todo!("C3: train_forest")
}

/// C4. The forest's class probabilities at `x`: the average of the trees' leaf shares (with
/// fully grown, pure leaves: the share of the trees that vote for each class).
///
/// Hint: `tree.shares(x)` gives one tree's shares.
pub fn vote(trees: &[Tree], x: &[f32], n_classes: usize) -> Vec<f32> {
    todo!("C4: vote")
}

/// The most probable class (the lowest class number on a tie).
pub fn label(p: &[f32]) -> usize {
    let mut best = 0;
    for k in 1..p.len() {
        if p[k] > p[best] {
            best = k;
        }
    }
    best
}

/// C5. The out-of-bag error: classify every training point with only the trees that did not
/// have it in their bootstrap sample, and count the mistakes. Points that every tree saw are
/// left out of the count.
pub fn oob_error(forest: &Forest, points: &[Vec<f32>], labels: &[usize], n_classes: usize) -> f32 {
    todo!("C5: oob_error")
}

/// The share of the points that the whole forest classifies wrongly.
pub fn error_rate(trees: &[Tree], points: &[Vec<f32>], labels: &[usize], n_classes: usize) -> f32 {
    let wrong = points.iter().zip(labels).filter(|(x, &l)| label(&vote(trees, x, n_classes)) != l).count();
    wrong as f32 / points.len() as f32
}
