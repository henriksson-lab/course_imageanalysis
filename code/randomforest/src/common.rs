//! Helpers shared by the units of day 34 (`pixelclass` and `forest_objects`): a small random
//! forest, the trees of the decision-trees unit, grown on bootstrap samples with a random
//! subset of `m` features at every split (the forest unit). Filled in as a tool; it is not part
//! of the exercises.
//!
//! Every node keeps its (weighted) class counts, so that the impurity decrease of each split
//! can be read off the tree afterwards. With class weights, each training point of class k
//! counts `weights[k]` times.

use imgcourse::Rng;

/// A decision tree. Each split asks "is feature `feature` ≤ `threshold`?": yes goes left.
#[derive(Clone, Debug, PartialEq)]
pub enum Tree {
    Leaf { counts: Vec<f32> },
    Split { feature: usize, threshold: f32, counts: Vec<f32>, left: Box<Tree>, right: Box<Tree> },
}

impl Tree {
    /// The (weighted) class counts of the training points that reached this node.
    pub fn counts(&self) -> &[f32] {
        match self {
            Tree::Leaf { counts } | Tree::Split { counts, .. } => counts,
        }
    }

    /// The class shares of the leaf that `x` lands in.
    pub fn shares(&self, x: &[f32]) -> Vec<f32> {
        match self {
            Tree::Leaf { counts } => {
                let total: f32 = counts.iter().sum();
                counts.iter().map(|c| c / total).collect()
            }
            Tree::Split { feature, threshold, left, right, .. } => {
                if x[*feature] <= *threshold {
                    left.shares(x)
                } else {
                    right.shares(x)
                }
            }
        }
    }
}

/// A trained forest: the trees, and for every tree which training points were in its
/// bootstrap sample (`in_bag[tree][point]`).
pub struct Forest {
    pub trees: Vec<Tree>,
    pub in_bag: Vec<Vec<bool>>,
    pub n_classes: usize,
}

/// The settings of a forest.
#[derive(Clone, Debug)]
pub struct Settings {
    pub n_trees: usize,
    /// Features tried at every split (`m`); at least 1, at most the number of features.
    pub m: usize,
    /// Weight of every class (all 1 for no weighting).
    pub class_weights: Vec<f32>,
    pub seed: u64,
}

impl Forest {
    /// Grow `settings.n_trees` fully grown trees, each on its own bootstrap sample.
    pub fn train(points: &[Vec<f32>], labels: &[usize], n_classes: usize, settings: &Settings) -> Forest {
        let mut rng = Rng::new(settings.seed);
        let (n, d) = (points.len(), points[0].len());
        let m = settings.m.clamp(1, d);
        let (mut trees, mut in_bag) = (Vec::new(), Vec::new());
        for _ in 0..settings.n_trees {
            let rows: Vec<usize> = (0..n).map(|_| rng.below(n)).collect();
            let mut bag = vec![false; n];
            for &r in &rows {
                bag[r] = true;
            }
            let grower = Grower { points, labels, n_classes, m, weights: &settings.class_weights };
            trees.push(grower.grow(rows, &mut rng));
            in_bag.push(bag);
        }
        Forest { trees, in_bag, n_classes }
    }

    /// The forest's class probabilities at `x`: the average of the trees' leaf shares.
    pub fn predict(&self, x: &[f32]) -> Vec<f32> {
        let mut p = vec![0.0; self.n_classes];
        for tree in &self.trees {
            for (pk, s) in p.iter_mut().zip(tree.shares(x)) {
                *pk += s / self.trees.len() as f32;
            }
        }
        p
    }

    /// The out-of-bag error: every training point is classified only by the trees that did
    /// not have it in their bootstrap sample (points that every tree saw are skipped).
    pub fn oob_error(&self, points: &[Vec<f32>], labels: &[usize]) -> f32 {
        let (mut wrong, mut counted) = (0, 0);
        for (i, x) in points.iter().enumerate() {
            let mut p = vec![0.0; self.n_classes];
            let mut votes = 0;
            for (tree, bag) in self.trees.iter().zip(&self.in_bag) {
                if !bag[i] {
                    for (pk, s) in p.iter_mut().zip(tree.shares(x)) {
                        *pk += s;
                    }
                    votes += 1;
                }
            }
            if votes > 0 {
                counted += 1;
                if argmax(&p) != labels[i] {
                    wrong += 1;
                }
            }
        }
        wrong as f32 / counted.max(1) as f32
    }
}

/// The index of the largest value (the first one on a tie).
pub fn argmax(p: &[f32]) -> usize {
    let mut best = 0;
    for k in 1..p.len() {
        if p[k] > p[best] {
            best = k;
        }
    }
    best
}

fn gini(counts: &[f32]) -> f32 {
    let n: f32 = counts.iter().sum();
    if n == 0.0 {
        return 0.0;
    }
    1.0 - counts.iter().map(|c| (c / n) * (c / n)).sum::<f32>()
}

struct Grower<'a> {
    points: &'a [Vec<f32>],
    labels: &'a [usize],
    n_classes: usize,
    m: usize,
    weights: &'a [f32],
}

impl Grower<'_> {
    fn counts(&self, rows: &[usize]) -> Vec<f32> {
        let mut c = vec![0.0; self.n_classes];
        for &r in rows {
            c[self.labels[r]] += self.weights[self.labels[r]];
        }
        c
    }

    fn grow(&self, rows: Vec<usize>, rng: &mut Rng) -> Tree {
        let counts = self.counts(&rows);
        let parent = gini(&counts);
        if parent == 0.0 {
            return Tree::Leaf { counts };
        }
        // m random candidate features (a partial shuffle), tried in increasing order
        let d = self.points[0].len();
        let mut all: Vec<usize> = (0..d).collect();
        for i in 0..self.m {
            let j = i + rng.below(d - i);
            all.swap(i, j);
        }
        let mut candidates = all[..self.m].to_vec();
        candidates.sort();
        // the best split among the candidates: sort, then sweep the threshold
        let total: f32 = counts.iter().sum();
        let mut best: Option<(usize, f32, f32)> = None; // feature, threshold, decrease
        for &f in &candidates {
            let mut order = rows.clone();
            order.sort_by(|&a, &b| self.points[a][f].total_cmp(&self.points[b][f]));
            let mut left = vec![0.0; self.n_classes];
            let mut right = counts.clone();
            for i in 0..order.len() - 1 {
                let l = self.labels[order[i]];
                left[l] += self.weights[l];
                right[l] -= self.weights[l];
                let (here, next) = (self.points[order[i]][f], self.points[order[i + 1]][f]);
                if here == next {
                    continue;
                }
                let (wl, wr): (f32, f32) = (left.iter().sum(), right.iter().sum());
                let decrease = parent - (wl * gini(&left) + wr * gini(&right)) / total;
                if best.is_none() || decrease > best.unwrap().2 {
                    best = Some((f, (here + next) / 2.0, decrease));
                }
            }
        }
        let (feature, threshold) = match best {
            Some((f, t, dec)) if dec > 1e-7 => (f, t),
            _ => return Tree::Leaf { counts },
        };
        let (l, r): (Vec<usize>, Vec<usize>) = rows.iter().partition(|&&i| self.points[i][feature] <= threshold);
        Tree::Split { feature, threshold, counts, left: Box::new(self.grow(l, rng)), right: Box::new(self.grow(r, rng)) }
    }
}
