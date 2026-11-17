// Model solution for the coding exercises of "Networks and classical pipelines compared".

use imgcourse::{Image, Rng};

/// The overlap table of two label images (0 = background): `inter[t][p]` is the number of
/// pixels with true label `t` and predicted label `p`. Row and column 0 are the background.
pub fn overlaps(truth: &Image<u32>, pred: &Image<u32>) -> Vec<Vec<usize>> {
    let nt = truth.pixels().iter().copied().max().unwrap_or(0) as usize;
    let np = pred.pixels().iter().copied().max().unwrap_or(0) as usize;
    let mut inter = vec![vec![0; np + 1]; nt + 1];
    for (&t, &p) in truth.pixels().iter().zip(pred.pixels()) {
        inter[t as usize][p as usize] += 1;
    }
    inter
}

/// The IoU table of the objects (row t − 1 = true object t, column p − 1 = predicted object
/// p), computed from the overlap table. Labels that do not occur get no row or column.
pub fn iou_table(truth: &Image<u32>, pred: &Image<u32>) -> Vec<Vec<f32>> {
    let inter = overlaps(truth, pred);
    let area_t: Vec<usize> = inter.iter().map(|row| row.iter().sum()).collect();
    let area_p: Vec<usize> = (0..inter[0].len()).map(|p| inter.iter().map(|row| row[p]).sum()).collect();
    let ts: Vec<usize> = (1..inter.len()).filter(|&t| area_t[t] > 0).collect();
    let ps: Vec<usize> = (1..inter[0].len()).filter(|&p| area_p[p] > 0).collect();
    ts.iter().map(|&t| ps.iter().map(|&p| inter[t][p] as f32 / (area_t[t] + area_p[p] - inter[t][p]) as f32).collect()).collect()
}

/// C1. F1 at every threshold τ of `taus` (all at least 0.5) from the IoU table (rows: true
/// objects, columns: predicted objects). Above 0.5 every object has at most one partner, so
/// TP is simply the number of pairs with IoU ≥ τ, and `F1 = 2 TP / (N_true + N_pred)`.
pub fn f1_curve(iou: &[Vec<f32>], n_pred: usize, taus: &[f32]) -> Vec<f32> {
    let n_true = iou.len();
    let mut out = Vec::new();
    for &tau in taus {
        let mut tp = 0;
        for row in iou {
            for &v in row {
                if v >= tau {
                    tp += 1;
                }
            }
        }
        out.push(if n_true + n_pred == 0 { 1.0 } else { 2.0 * tp as f32 / (n_true + n_pred) as f32 });
    }
    out
}

/// C2. Merges and splits from the overlap table `inter` (see `overlaps`). A **merge** is a
/// predicted object that contains at least half of each of two or more true objects. A
/// **split** is a true object that contains at least half of each of two or more predicted
/// objects. Returns `(merges, splits)`.
pub fn merges_and_splits(inter: &[Vec<usize>]) -> (usize, usize) {
    let (nt, np) = (inter.len(), inter[0].len());
    let area_t: Vec<usize> = (0..nt).map(|t| inter[t].iter().sum()).collect();
    let area_p: Vec<usize> = (0..np).map(|p| (0..nt).map(|t| inter[t][p]).sum()).collect();
    let mut merges = 0;
    for p in 1..np {
        let covered = (1..nt).filter(|&t| area_t[t] > 0 && 2 * inter[t][p] >= area_t[t]).count();
        if covered >= 2 {
            merges += 1;
        }
    }
    let mut splits = 0;
    for t in 1..nt {
        let pieces = (1..np).filter(|&p| area_p[p] > 0 && 2 * inter[t][p] >= area_p[p]).count();
        if pieces >= 2 {
            splits += 1;
        }
    }
    (merges, splits)
}

/// C3. A bootstrap interval for the mean of the per-image differences `d`: `resamples` times,
/// draw `d.len()` values from `d` with replacement and take their mean; sort these means and
/// return `(mean of d, 2.5th percentile, 97.5th percentile)`.
pub fn bootstrap_interval(d: &[f32], resamples: usize, seed: u64) -> (f32, f32, f32) {
    let n = d.len();
    let mean = d.iter().sum::<f32>() / n as f32;
    let mut rng = Rng::new(seed);
    let mut means = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += d[rng.below(n)];
        }
        means.push(sum / n as f32);
    }
    means.sort_by(|a, b| a.total_cmp(b));
    let lo = means[(0.025 * resamples as f32) as usize];
    let hi = means[((0.975 * resamples as f32) as usize).min(resamples - 1)];
    (mean, lo, hi)
}
