// Model solution for the coding exercises of "Is it more than chance? Simulation envelopes".

use imgcourse::Rng;

pub use crate::common::{csr, dist, hard_core, poisson, thomas, Point};

/// `L(r) - r` with the translation correction (the ripley-k unit) at every radius of `radii`,
/// from one pass over the pairs. Given.
pub fn l_curve(points: &[Point], a: f64, b: f64, radii: &[f64]) -> Vec<f64> {
    let n = points.len() as f64;
    let mut sums = vec![0.0; radii.len()];
    for (i, &p) in points.iter().enumerate() {
        for &q in &points[i + 1..] {
            let d = dist(p, q);
            let w = a * b / ((a - (q.0 - p.0).abs()) * (b - (q.1 - p.1).abs()));
            for (k, &r) in radii.iter().enumerate() {
                if d <= r {
                    sums[k] += 2.0 * w; // the pair counts in both orders
                }
            }
        }
    }
    radii.iter().zip(&sums).map(|(&r, &s)| (a * b / (n * (n - 1.0)) * s / std::f64::consts::PI).sqrt() - r).collect()
}

/// C1. The one-sided Monte Carlo p-value (slide "The Monte Carlo p-value"):
/// `(1 + number of simulated values at least as extreme as the observed) / (1 + s)`.
/// "As extreme" means `>=` the observed value when `larger` is true, `<=` otherwise.
pub fn mc_p_value(observed: f64, simulated: &[f64], larger: bool) -> f64 {
    let extreme = simulated.iter().filter(|&&t| if larger { t >= observed } else { t <= observed }).count();
    (1 + extreme) as f64 / (1 + simulated.len()) as f64
}

/// C2a. The pointwise envelope: at every radius (index `k`), the smallest and the largest
/// value of the simulated curves. Returns `(lower, upper)`.
pub fn pointwise_envelope(curves: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let m = curves[0].len();
    let mut lower = vec![f64::INFINITY; m];
    let mut upper = vec![f64::NEG_INFINITY; m];
    for c in curves {
        for k in 0..m {
            lower[k] = lower[k].min(c[k]);
            upper[k] = upper[k].max(c[k]);
        }
    }
    (lower, upper)
}

/// C2b. The indices of the radii at which the observed curve lies outside the envelope.
pub fn escapes(observed: &[f64], lower: &[f64], upper: &[f64]) -> Vec<usize> {
    (0..observed.len()).filter(|&k| observed[k] < lower[k] || observed[k] > upper[k]).collect()
}

/// C3. The MAD test (slide "A global test: the MAD test"): `T = max over r of |curve(r)|` for
/// the observed curve and for every simulated one (the curves are deviations from the CSR
/// value, such as L(r) - r), and the Monte Carlo p-value of the observed T (large T is extreme).
/// Returns `(T_observed, p)`.
pub fn mad_test(observed: &[f64], simulated: &[Vec<f64>]) -> (f64, f64) {
    let mad = |c: &[f64]| c.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let t0 = mad(observed);
    let ts: Vec<f64> = simulated.iter().map(|c| mad(c)).collect();
    (t0, mc_p_value(t0, &ts, true))
}

/// C4. How often is CSR rejected? Simulate `patterns` CSR patterns of `n` points in the window;
/// for each, compute its `L(r) - r` curve and those of `s` further CSR patterns (the null), and
/// record whether it leaves the pointwise envelope at some radius, and whether the MAD test
/// gives p <= 0.05. Returns the two shares `(pointwise, mad)`.
pub fn false_alarm_rates(n: usize, a: f64, b: f64, radii: &[f64], s: usize, patterns: usize, rng: &mut Rng) -> (f64, f64) {
    let (mut pointwise, mut mad) = (0, 0);
    for _ in 0..patterns {
        let observed = l_curve(&csr(n, a, b, rng), a, b, radii);
        let sims: Vec<Vec<f64>> = (0..s).map(|_| l_curve(&csr(n, a, b, rng), a, b, radii)).collect();
        let (lower, upper) = pointwise_envelope(&sims);
        if !escapes(&observed, &lower, &upper).is_empty() {
            pointwise += 1;
        }
        if mad_test(&observed, &sims).1 <= 0.05 {
            mad += 1;
        }
    }
    (pointwise as f64 / patterns as f64, mad as f64 / patterns as f64)
}
