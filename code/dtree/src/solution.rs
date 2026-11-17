// Model solution for the coding exercises of "Decision trees".

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
    let n: usize = counts.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let mut sum_p2 = 0.0;
    for &c in counts {
        let p = c as f32 / n as f32;
        sum_p2 += p * p;
    }
    1.0 - sum_p2
}

/// C2. The best split of the points (`points[i]` is the feature vector of point i): for
/// every feature and every threshold halfway between two neighbouring distinct values, the
/// impurity decrease `G(parent) − (n_L/n) G(L) − (n_R/n) G(R)`. Only splits that leave at
/// least `min_leaf` points on each side count. Returns the split with the largest decrease
/// (the first one found on a tie: lowest feature, then lowest threshold), or `None` if
/// there is no allowed split.
pub fn best_split(points: &[Vec<f32>], labels: &[usize], n_classes: usize, min_leaf: usize) -> Option<Split> {
    let n = points.len();
    let parent = gini(&class_counts(labels, n_classes));
    let mut best: Option<Split> = None;
    for feature in 0..points.first()?.len() {
        // sort the points by this feature, then sweep the threshold from left to right
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| points[a][feature].total_cmp(&points[b][feature]));
        let mut left = vec![0; n_classes];
        let mut right = class_counts(labels, n_classes);
        for i in 0..n.saturating_sub(1) {
            // move point order[i] from the right child to the left child
            left[labels[order[i]]] += 1;
            right[labels[order[i]]] -= 1;
            let (here, next) = (points[order[i]][feature], points[order[i + 1]][feature]);
            let n_left = i + 1;
            if here == next || n_left < min_leaf || n - n_left < min_leaf {
                continue; // no threshold between equal values, or a child too small
            }
            let weighted = (n_left as f32 * gini(&left) + (n - n_left) as f32 * gini(&right)) / n as f32;
            let decrease = parent - weighted;
            if best.is_none() || decrease > best.unwrap().decrease {
                best = Some(Split { feature, threshold: (here + next) / 2.0, decrease });
            }
        }
    }
    best
}

/// C3. Grow a tree greedily (recursive partitioning): a node becomes a leaf when it is pure,
/// when it is at depth `max_depth` (the root has depth 0), or when no allowed split lowers
/// the impurity; otherwise it takes the best split (C2) and grows both children.
pub fn grow(points: &[Vec<f32>], labels: &[usize], n_classes: usize, max_depth: usize, min_leaf: usize) -> Tree {
    let counts = class_counts(labels, n_classes);
    if max_depth == 0 || gini(&counts) == 0.0 {
        return Tree::Leaf { counts };
    }
    let split = match best_split(points, labels, n_classes, min_leaf) {
        Some(s) if s.decrease > 0.0 => s,
        _ => return Tree::Leaf { counts },
    };
    let (mut lp, mut ll, mut rp, mut rl) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (p, &l) in points.iter().zip(labels) {
        if p[split.feature] <= split.threshold {
            lp.push(p.clone());
            ll.push(l);
        } else {
            rp.push(p.clone());
            rl.push(l);
        }
    }
    Tree::Split {
        feature: split.feature,
        threshold: split.threshold,
        left: Box::new(grow(&lp, &ll, n_classes, max_depth - 1, min_leaf)),
        right: Box::new(grow(&rp, &rl, n_classes, max_depth - 1, min_leaf)),
    }
}

/// C4. Classify a point: follow the questions down to a leaf and return its majority class
/// (the lowest class number on a tie).
pub fn predict(tree: &Tree, x: &[f32]) -> usize {
    match tree {
        Tree::Leaf { counts } => {
            let mut best = 0;
            for k in 1..counts.len() {
                if counts[k] > counts[best] {
                    best = k;
                }
            }
            best
        }
        Tree::Split { feature, threshold, left, right } => {
            if x[*feature] <= *threshold {
                predict(left, x)
            } else {
                predict(right, x)
            }
        }
    }
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
    let sse = |v: &[f32]| {
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        v.iter().map(|y| (y - mean) * (y - mean)).sum::<f32>()
    };
    let mut best = (f32::NAN, f32::INFINITY);
    for i in 1..xs.len() {
        let total = sse(&ys[..i]) + sse(&ys[i..]);
        if total < best.1 {
            best = ((xs[i - 1] + xs[i]) / 2.0, total);
        }
    }
    best
}
