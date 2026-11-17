//! The decision trees of the forest (the decision-trees unit, filled in here; not part of
//! these exercises). Two changes from that unit: a leaf holds the class **shares**, and every
//! split looks only at the candidate features that `candidates()` hands it.

/// A decision tree. Each split asks "is feature `feature` ≤ `threshold`?": yes goes left.
#[derive(Clone, Debug, PartialEq)]
pub enum Tree {
    /// A leaf: the share of each class among the training points that ended here.
    Leaf { shares: Vec<f32> },
    Split { feature: usize, threshold: f32, left: Box<Tree>, right: Box<Tree> },
}

impl Tree {
    /// A leaf that votes for one class only (handy for small examples by hand).
    pub fn vote_for(class: usize, n_classes: usize) -> Tree {
        let mut shares = vec![0.0; n_classes];
        shares[class] = 1.0;
        Tree::Leaf { shares }
    }

    /// The class shares of the leaf that `x` lands in.
    pub fn shares(&self, x: &[f32]) -> &[f32] {
        match self {
            Tree::Leaf { shares } => shares,
            Tree::Split { feature, threshold, left, right } => {
                if x[*feature] <= *threshold {
                    left.shares(x)
                } else {
                    right.shares(x)
                }
            }
        }
    }
}

fn gini(counts: &[usize]) -> f32 {
    let n: usize = counts.iter().sum();
    1.0 - counts.iter().map(|&c| (c as f32 / n as f32).powi(2)).sum::<f32>()
}

/// Grow a tree until its leaves are pure, on the training points with the numbers `rows`
/// (a bootstrap sample: a number may appear several times). At every split, `candidates()`
/// gives the features to try.
pub fn grow_tree(
    points: &[Vec<f32>],
    labels: &[usize],
    rows: &[usize],
    n_classes: usize,
    candidates: &mut dyn FnMut() -> Vec<usize>,
) -> Tree {
    let mut counts = vec![0; n_classes];
    for &r in rows {
        counts[labels[r]] += 1;
    }
    let parent = gini(&counts);
    if parent == 0.0 {
        return Tree::vote_for(labels[rows[0]], n_classes);
    }
    let n = rows.len();
    let mut best: Option<(usize, f32, f32)> = None; // feature, threshold, decrease
    for f in candidates() {
        let mut order = rows.to_vec();
        order.sort_by(|&a, &b| points[a][f].total_cmp(&points[b][f]));
        let mut left = vec![0; n_classes];
        let mut right = counts.clone();
        for i in 0..n - 1 {
            left[labels[order[i]]] += 1;
            right[labels[order[i]]] -= 1;
            let (here, next) = (points[order[i]][f], points[order[i + 1]][f]);
            if here == next {
                continue;
            }
            let decrease = parent - ((i + 1) as f32 * gini(&left) + (n - i - 1) as f32 * gini(&right)) / n as f32;
            if best.is_none() || decrease > best.unwrap().2 {
                best = Some((f, (here + next) / 2.0, decrease));
            }
        }
    }
    match best {
        Some((feature, threshold, decrease)) if decrease > 1e-7 => {
            let (l, r): (Vec<usize>, Vec<usize>) = rows.iter().partition(|&&i| points[i][feature] <= threshold);
            Tree::Split {
                feature,
                threshold,
                left: Box::new(grow_tree(points, labels, &l, n_classes, candidates)),
                right: Box::new(grow_tree(points, labels, &r, n_classes, candidates)),
            }
        }
        _ => Tree::Leaf { shares: counts.iter().map(|&c| c as f32 / n as f32).collect() },
    }
}
