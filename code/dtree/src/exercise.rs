// Coding exercises for "Decision trees". Replace each `todo!()` with your code, then check it
// with `cargo test -p dtree` and look at `cargo run -p dtree`.
#![allow(unused_variables, unused_imports)]

/// A decision tree. Each split asks "is feature `feature` ≤ `threshold`?": yes goes left.
#[derive(Clone, Debug, PartialEq)]
pub enum Tree {
    /// A leaf: the number of training points of each class that ended here.
    Leaf { counts: Vec<usize> },
    Split { feature: usize, threshold: f32, left: Box<Tree>, right: Box<Tree> },
}

/// A split of a node, and how much it lowers the weighted Gini impurity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Split {
    pub feature: usize,
    pub threshold: f32,
    pub decrease: f32,
}

/// The number of points of each class (classes 0 .. n_classes − 1).
pub fn class_counts(labels: &[usize], n_classes: usize) -> Vec<usize> {
    let mut counts = vec![0; n_classes];
    for &l in labels {
        counts[l] += 1;
    }
    counts
}

/// C1. The Gini impurity of a node from its class counts: `1 − Σ p_k²`, with `p_k` the share
/// of class k. 0 for an empty node.
pub fn gini(counts: &[usize]) -> f32 {
    todo!("C1: gini")
}

/// C2. The best split of the points (`points[i]` is the feature vector of point i): for
/// every feature and every threshold halfway between two neighbouring distinct values, the
/// impurity decrease `G(parent) − (n_L/n) G(L) − (n_R/n) G(R)`. Only splits that leave at
/// least `min_leaf` points on each side count. Returns the split with the largest decrease
/// (the first one found on a tie: lowest feature, then lowest threshold), or `None` if
/// there is no allowed split.
///
/// Hint: for one feature, sort the point numbers by that feature
/// (`order.sort_by(|&a, &b| points[a][feature].total_cmp(&points[b][feature]))`). Then move
/// the points one by one from the right child to the left child, updating both class counts
/// (`class_counts` gives the start), and try the threshold after each point.
pub fn best_split(points: &[Vec<f32>], labels: &[usize], n_classes: usize, min_leaf: usize) -> Option<Split> {
    todo!("C2: best_split")
}

/// C3. Grow a tree greedily (recursive partitioning): a node becomes a leaf when it is pure,
/// when it is at depth `max_depth` (the root has depth 0), or when no allowed split lowers
/// the impurity; otherwise it takes the best split (C2) and grows both children.
///
/// Hint: a recursive function. Split the points and labels into two new `Vec`s with the
/// question of the best split, and call `grow` on each with `max_depth − 1`.
pub fn grow(points: &[Vec<f32>], labels: &[usize], n_classes: usize, max_depth: usize, min_leaf: usize) -> Tree {
    todo!("C3: grow")
}

/// C4. Classify a point: follow the questions down to a leaf and return its majority class
/// (the lowest class number on a tie).
///
/// Hint: `match tree { Tree::Leaf { counts } => ..., Tree::Split { feature, threshold, left, right } => ... }`.
pub fn predict(tree: &Tree, x: &[f32]) -> usize {
    todo!("C4: predict")
}

/// The share of the points that the tree classifies wrongly.
pub fn error_rate(tree: &Tree, points: &[Vec<f32>], labels: &[usize]) -> f32 {
    let wrong = points.iter().zip(labels).filter(|(p, &l)| predict(tree, p) != l).count();
    wrong as f32 / points.len() as f32
}

/// The number of leaves of a tree.
pub fn leaves(tree: &Tree) -> usize {
    match tree {
        Tree::Leaf { .. } => 1,
        Tree::Split { left, right, .. } => leaves(left) + leaves(right),
    }
}

/// C5. A regression stump on a profile: the values `ys` at positions `xs` (sorted, distinct).
/// For every split between two neighbouring positions, predict each side by its mean and
/// add up the squared errors. Returns the threshold (halfway between the two positions) and
/// the sum of squared errors of the best split.
pub fn regression_stump(xs: &[f32], ys: &[f32]) -> (f32, f32) {
    todo!("C5: regression_stump")
}
