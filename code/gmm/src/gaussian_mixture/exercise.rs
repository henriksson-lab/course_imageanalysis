// Coding exercises for "The multivariate Gaussian and the mixture model". Replace each `todo!()` with your
// code, then check it with `cargo test -p gmm gaussian_mixture::` and look at `cargo run -p gmm -- gaussian-mixture`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: `let [[a, b], [_, d]] = g.sigma;` takes the matrix apart.
pub fn mahalanobis(x: [f32; 2], g: &Gaussian) -> f32 {
    todo!("C1: mahalanobis")
}

/// C2. The density `N(x | μ, Σ) = exp(−d_M² / 2) / (2π sqrt(|Σ|))` in 2D (slide "The math: the
/// density and its ellipses").
///
/// Hint: use `mahalanobis`; `std::f32::consts::PI`.
pub fn density(x: [f32; 2], g: &Gaussian) -> f32 {
    todo!("C2: density")
}

/// C3. The responsibilities `r_k(x) = π_k N_k(x) / Σ_j π_j N_j(x)` of a mixture with weights
/// `weights` (slide "Responsibilities: soft labels").
pub fn responsibilities(x: [f32; 2], weights: &[f32], components: &[Gaussian]) -> Vec<f32> {
    todo!("C3: responsibilities")
}

/// C4. The maximum-likelihood Gaussian of a set of points: the mean and the covariance,
/// dividing by `n` (slide "The likelihood of a data set").
pub fn fit_gaussian(points: &[[f32; 2]]) -> Gaussian {
    todo!("C4: fit_gaussian")
}

/// C5. Draw `n` points from a mixture (slide "A mixture is a recipe for data"): pick a
/// component `k` with probability `weights[k]`, then draw from its Gaussian as
/// `x = μ + L z`, with `z` two independent standard normal numbers and `L` the lower
/// triangular matrix with `L Lᵀ = Σ`:
/// `L = [[√a, 0], [b/√a, √(d − b²/a)]]` for `Σ = [[a, b], [b, d]]`.
/// Returns the points and the component that made each one (the hidden labels).
///
/// Hint: `rng.uniform()` for the component (add up the weights until the sum passes it),
/// `rng.normal()` for `z1` and `z2`.
pub fn sample_mixture(weights: &[f32], components: &[Gaussian], n: usize, rng: &mut Rng) -> (Vec<[f32; 2]>, Vec<usize>) {
    todo!("C5: sample_mixture")
}
