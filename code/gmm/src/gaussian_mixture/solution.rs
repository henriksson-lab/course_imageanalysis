// Model solution for the coding exercises of "The multivariate Gaussian and the mixture model".

use imgcourse::Rng;

/// A Gaussian in 2D: centre `mu` and covariance matrix `sigma = [[σ1², c], [c, σ2²]]`.
#[derive(Clone, Copy, Debug)]
pub struct Gaussian {
    pub mu: [f32; 2],
    pub sigma: [[f32; 2]; 2],
}

/// C1. The Mahalanobis distance `d_M = sqrt((x − μ)ᵀ Σ⁻¹ (x − μ))` (slide "Mahalanobis
/// distance"), with the inverse of the 2 × 2 matrix written out:
/// `[[a, b], [b, d]]⁻¹ = [[d, −b], [−b, a]] / (ad − b²)`.
pub fn mahalanobis(x: [f32; 2], g: &Gaussian) -> f32 {
    let [[a, b], [_, d]] = g.sigma;
    let det = a * d - b * b;
    let (u, v) = (x[0] - g.mu[0], x[1] - g.mu[1]);
    ((d * u * u - 2.0 * b * u * v + a * v * v) / det).sqrt()
}

/// C2. The density `N(x | μ, Σ) = exp(−d_M² / 2) / (2π sqrt(|Σ|))` in 2D (slide "The math: the
/// density and its ellipses").
pub fn density(x: [f32; 2], g: &Gaussian) -> f32 {
    let [[a, b], [_, d]] = g.sigma;
    let det = a * d - b * b;
    let dm = mahalanobis(x, g);
    (-0.5 * dm * dm).exp() / (2.0 * std::f32::consts::PI * det.sqrt())
}

/// C3. The responsibilities `r_k(x) = π_k N_k(x) / Σ_j π_j N_j(x)` of a mixture with weights
/// `weights` (slide "Responsibilities: soft labels").
pub fn responsibilities(x: [f32; 2], weights: &[f32], components: &[Gaussian]) -> Vec<f32> {
    let parts: Vec<f32> = weights.iter().zip(components).map(|(w, g)| w * density(x, g)).collect();
    let total: f32 = parts.iter().sum();
    parts.iter().map(|p| p / total).collect()
}

/// C4. The maximum-likelihood Gaussian of a set of points: the mean and the covariance,
/// dividing by `n` (slide "The likelihood of a data set").
pub fn fit_gaussian(points: &[[f32; 2]]) -> Gaussian {
    let n = points.len() as f32;
    let mut mu = [0.0; 2];
    for p in points {
        mu[0] += p[0] / n;
        mu[1] += p[1] / n;
    }
    let mut sigma = [[0.0; 2]; 2];
    for p in points {
        let y = [p[0] - mu[0], p[1] - mu[1]];
        for i in 0..2 {
            for j in 0..2 {
                sigma[i][j] += y[i] * y[j] / n;
            }
        }
    }
    Gaussian { mu, sigma }
}

/// C5. Draw `n` points from a mixture (slide "A mixture is a recipe for data"): pick a
/// component `k` with probability `weights[k]`, then draw from its Gaussian as
/// `x = μ + L z`, with `z` two independent standard normal numbers and `L` the lower
/// triangular matrix with `L Lᵀ = Σ`:
/// `L = [[√a, 0], [b/√a, √(d − b²/a)]]` for `Σ = [[a, b], [b, d]]`.
/// Returns the points and the component that made each one (the hidden labels).
pub fn sample_mixture(weights: &[f32], components: &[Gaussian], n: usize, rng: &mut Rng) -> (Vec<[f32; 2]>, Vec<usize>) {
    let (mut points, mut labels) = (Vec::new(), Vec::new());
    for _ in 0..n {
        // the component: walk along the cumulative weights
        let u = rng.uniform();
        let mut k = components.len() - 1;
        let mut sum = 0.0;
        for (j, w) in weights.iter().enumerate() {
            sum += w;
            if u < sum {
                k = j;
                break;
            }
        }
        let g = &components[k];
        let [[a, b], [_, d]] = g.sigma;
        let l11 = a.sqrt();
        let l21 = b / l11;
        let l22 = (d - l21 * l21).sqrt();
        let (z1, z2) = (rng.normal(), rng.normal());
        points.push([g.mu[0] + l11 * z1, g.mu[1] + l21 * z1 + l22 * z2]);
        labels.push(k);
    }
    (points, labels)
}
