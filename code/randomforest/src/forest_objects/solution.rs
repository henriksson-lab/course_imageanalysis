// Model solution for the coding exercises of "Forests on feature tables, and feature importance".

use crate::forest_objects::{Forest, Tree};
use imgcourse::Rng;

/// C1. The Gini impurity `1 − Σ p_k²` from (possibly weighted) class counts; 0 if empty.
pub fn gini(counts: &[f32]) -> f32 {
    let n: f32 = counts.iter().sum();
    if n == 0.0 {
        return 0.0;
    }
    let mut sum_p2 = 0.0;
    for &c in counts {
        sum_p2 += (c / n) * (c / n);
    }
    1.0 - sum_p2
}

/// Add the weighted impurity decreases of all splits below `node` to `importance`.
fn add_decreases(node: &Tree, n_root: f32, importance: &mut [f32]) {
    if let Tree::Split { feature, left, right, counts, .. } = node {
        let n_t: f32 = counts.iter().sum();
        let (n_l, n_r): (f32, f32) = (left.counts().iter().sum(), right.counts().iter().sum());
        let decrease = gini(counts) - n_l / n_t * gini(left.counts()) - n_r / n_t * gini(right.counts());
        importance[*feature] += n_t / n_root * decrease;
        add_decreases(left, n_root, importance);
        add_decreases(right, n_root, importance);
    }
}

/// C2. The impurity importance (mean decrease in impurity) of every feature: in every tree,
/// add `(n_t / n) ΔG_t` of every split to the feature it splits on (`n_t` = points in the
/// node, `n` = points at the root, `ΔG_t` = the Gini decrease of the split); then average
/// over the trees.
pub fn impurity_importance(trees: &[Tree], n_features: usize) -> Vec<f32> {
    let mut importance = vec![0.0; n_features];
    for tree in trees {
        let n_root: f32 = tree.counts().iter().sum();
        add_decreases(tree, n_root, &mut importance);
    }
    for v in importance.iter_mut() {
        *v /= trees.len() as f32;
    }
    importance
}

/// C3. The permutation importance of a group of columns (one column, or several shuffled
/// together): shuffle the rows of these columns with one random permutation, and measure how
/// much the out-of-bag error rises. Averaged over `repeats` shuffles.
pub fn permutation_importance(
    forest: &Forest,
    points: &[Vec<f32>],
    labels: &[usize],
    columns: &[usize],
    repeats: usize,
    rng: &mut Rng,
) -> f32 {
    let base = forest.oob_error(points, labels);
    let n = points.len();
    let mut rise = 0.0;
    for _ in 0..repeats {
        // a random permutation of 0 .. n − 1 (shuffle)
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(i, rng.below(i + 1));
        }
        let mut shuffled = points.to_vec();
        for i in 0..n {
            for &c in columns {
                shuffled[i][c] = points[perm[i]][c];
            }
        }
        rise += forest.oob_error(&shuffled, labels) - base;
    }
    rise / repeats as f32
}

/// C4. The (Pearson) correlation of two columns: their covariance divided by the product of
/// their standard deviations, between −1 and 1.
pub fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}
