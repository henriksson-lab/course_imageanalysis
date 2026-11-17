// Model solution for the coding exercises of "Training GANs and what goes wrong".

pub use crate::common::{GenLoss, LineGan, Mixture, Settings};

/// C1. Jensen–Shannon divergence between two densities given on the same grid of spacing `dx`:
/// JS = ½ Σ p log(2p/(p + q)) dx + ½ Σ q log(2q/(p + q)) dx, with 0 log 0 = 0. At most ln 2.
pub fn js_divergence(p: &[f64], q: &[f64], dx: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..p.len() {
        let m = p[i] + q[i];
        if p[i] > 0.0 {
            s += 0.5 * p[i] * (2.0 * p[i] / m).ln() * dx;
        }
        if q[i] > 0.0 {
            s += 0.5 * q[i] * (2.0 * q[i] / m).ln() * dx;
        }
    }
    s
}

/// C2. Wasserstein-1 distance ("earth mover's distance") between two equally large samples in
/// 1D: sort both, then the mean of |a_i − b_i| over matching ranks.
pub fn wasserstein1(a: &[f64], b: &[f64]) -> f64 {
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(|x, y| x.total_cmp(y));
    b.sort_by(|x, y| x.total_cmp(y));
    a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}

/// C3. For every point of `xs`: the distance to its k-th nearest other point of `xs` (1D).
pub fn knn_radii(xs: &[f64], k: usize) -> Vec<f64> {
    xs.iter()
        .enumerate()
        .map(|(i, &x)| {
            let mut d: Vec<f64> = xs.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, &y)| (x - y).abs()).collect();
            d.sort_by(|p, q| p.total_cmp(q));
            d[k - 1]
        })
        .collect()
}

/// C3. Precision and recall of generated points `fake` against real points `real` (1D):
/// precision = share of fake points within the k-NN radius of some real point (realism);
/// recall = share of real points within the k-NN radius of some fake point (coverage).
pub fn precision_recall(real: &[f64], fake: &[f64], k: usize) -> (f64, f64) {
    let rr = knn_radii(real, k);
    let rf = knn_radii(fake, k);
    let inside = |x: f64, set: &[f64], radii: &[f64]| set.iter().zip(radii).any(|(&y, &r)| (x - y).abs() <= r);
    let p = fake.iter().filter(|&&x| inside(x, real, &rr)).count() as f64 / fake.len() as f64;
    let r = real.iter().filter(|&&x| inside(x, fake, &rf)).count() as f64 / real.len() as f64;
    (p, r)
}
