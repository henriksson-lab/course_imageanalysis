// Coding exercises for "Kernel functions and the kernel trick". Replace each `todo!()` with your
// code, then check it with `cargo test -p svm kernel_trick::` and look at `cargo run -p svm -- kernel-trick`.
#![allow(unused_variables, unused_imports)]

/// A kernel function: a similarity of two feature vectors that is a dot product in some
/// lifted feature space.
#[derive(Clone, Copy, Debug)]
pub enum Kernel {
    /// `x · z`: no lifting, the linear SVM.
    Linear,
    /// `(x · z + c)^degree`: all monomials up to `degree`.
    Polynomial { c: f32, degree: i32 },
    /// `exp(−γ ‖x − z‖²)`: Gaussian bumps of length scale `1/√γ`.
    Rbf { gamma: f32 },
}

/// C1. The quadratic map of the slide "Kernel functions: the kernel trick":
/// `φ(x) = (x1², √2 x1 x2, x2²)`, so that `φ(x) · φ(z) = (x · z)²`.
pub fn quadratic_map(x: [f32; 2]) -> [f32; 3] {
    todo!("C1: quadratic_map")
}

/// C2. The kernel function `k(x, z)` for vectors of any length.
///
/// Hint: `match k { Kernel::Linear => ..., Kernel::Polynomial { c, degree } => ..., Kernel::Rbf { gamma } => ... }`;
/// `v.powi(degree)` raises to a whole power. The helper `dot` below is given.
pub fn kernel(k: Kernel, x: &[f32], z: &[f32]) -> f32 {
    todo!("C2: kernel")
}

/// The dot product `x · z`. (Given.)
pub fn dot(x: &[f32], z: &[f32]) -> f32 {
    x.iter().zip(z).map(|(a, b)| a * b).sum()
}

/// C3. The decision value of a kernel SVM, `f(x) = Σ_i α_i y_i k(x_i, x) + b` (slide "The RBF
/// kernel function"). Only the support vectors (`α_i > 0`) contribute.
pub fn decision(points: &[Vec<f32>], labels: &[f32], alpha: &[f32], b: f32, k: Kernel, x: &[f32]) -> f32 {
    todo!("C3: decision")
}

/// C4. Train a kernel SVM by SMO (slide "The math: how it is solved"): change two `α`s at a
/// time, always the pair that violates the optimality conditions most. Start from all `α = 0`.
/// Returns `(α, b)`.
///
/// With `g_k = Σ_l α_l y_l K_lk` (the decision value without `b`) and `F_k = g_k − y_k`:
/// - `i` = the point with the smallest `F` among those whose `α` may move in the direction of
///   its label (`y = +1` and `α < C`, or `y = −1` and `α > 0`);
/// - `j` = the point with the largest `F` among those whose `α` may move against it
///   (`y = −1` and `α < C`, or `y = +1` and `α > 0`);
/// - stop when `F_j − F_i < 2 · tol`: then `b = −(F_i + F_j)/2`.
///
/// Hint: compute the kernel matrix K first, and keep `g` up to date after every change:
/// `g_t += Δα_i y_i K_it + Δα_j y_j K_jt`. For the pair, with `η = K_ii + K_jj − 2 K_ij`:
/// `α_j ← α_j + y_j (F_i − F_j) / η`, clipped to `[lo, hi]`, then `α_i ← α_i + y_i y_j (old α_j − new α_j)`,
/// where `[lo, hi]` = `[max(0, α_j − α_i), min(C, C + α_j − α_i)]` if `y_i ≠ y_j`,
/// and `[max(0, α_i + α_j − C), min(C, α_i + α_j)]` if `y_i = y_j`.
pub fn train_smo(points: &[Vec<f32>], labels: &[f32], k: Kernel, c: f32, tol: f32, max_steps: usize) -> (Vec<f32>, f32) {
    todo!("C4: train_smo")
}
