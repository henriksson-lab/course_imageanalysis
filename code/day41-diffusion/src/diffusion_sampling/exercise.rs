// Coding exercises for "Samplers, steps, and judging the samples". Replace each `todo!()` with your code,
// then check it with `cargo test -p day41 diffusion_sampling::` and look at `cargo run -p day41 -- diffusion-sampling`.
#![allow(unused_variables, unused_imports)]

/// C1. Deterministic sampling as Euler steps of the probability-flow equation
/// `dy/dσ = (y − D(y, σ)) / σ`: start at `y` with the noise level `sigmas[0]` and step through
/// the levels `sigmas` (falling, the last one may be 0). Each step from σ to σ′ is
/// `y ← y + (σ′ − σ) · (y − D(y, σ)) / σ`, where `denoise(y, σ)` is the best denoiser `D`.
pub fn euler_sample(y: f32, sigmas: &[f32], denoise: impl Fn(f32, f32) -> f32) -> f32 {
    todo!("C1: euler_sample")
}

/// The mean and standard deviation (dividing by n) of a set of values (in `src/common.rs`).
pub use crate::common::mean_sd;

/// C2. The Fréchet distance of two sets of numbers in 1D: fit a Gaussian to each (mean and
/// standard deviation, dividing by n) and return `(μ1 − μ2)² + (s1 − s2)²`.
pub fn frechet_1d(a: &[f32], b: &[f32]) -> f32 {
    todo!("C2: frechet_1d")
}

/// The distance from `x` to its `k`-th nearest neighbour in `set` (the point itself, at
/// distance 0, counts as the first if it is in the set).
pub fn kth_distance(x: f32, set: &[f32], k: usize) -> f32 {
    let mut d: Vec<f32> = set.iter().map(|s| (s - x).abs()).collect();
    d.sort_by(|a, b| a.total_cmp(b));
    d[k.min(d.len() - 1)]
}

/// C3. Precision and recall with `k`-NN balls. Around every real point put a ball whose
/// radius is the distance to its `k`-th nearest real neighbour (`kth_distance(r, real, k)`);
/// **precision** is the fraction of generated points inside at least one real ball. In the
/// same way with balls around the generated points, **recall** is the fraction of real points
/// inside at least one generated ball. Returns `(precision, recall)`.
///
/// Hint: write the "fraction of `points` inside the balls of `centres`" once, as a closure
/// or a function, and call it twice.
pub fn precision_recall(real: &[f32], generated: &[f32], k: usize) -> (f32, f32) {
    todo!("C3: precision_recall")
}

/// C4. The memorisation check: for every sample, the distance to its nearest training point.
pub fn nearest_distances(samples: &[f32], training: &[f32]) -> Vec<f32> {
    todo!("C4: nearest_distances")
}
