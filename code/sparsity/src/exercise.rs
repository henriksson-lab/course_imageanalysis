// Coding exercises for "Sparsity and the ℓ1 penalty". Replace each `todo!()` with your code,
// then check it with `cargo test -p sparsity` and look at `cargo run -p sparsity`.
// Vectors and matrices are f64, as in imgcourse::linalg; a matrix is `a[row][column]`.
#![allow(unused_variables, unused_imports)]

/// C1a. The ℓ2 penalty (λ/2)c² for one coefficient: every coefficient shrinks,
/// `c = w / (1 + λ)`, and none becomes 0.
pub fn shrink_l2(w: f64, lambda: f64) -> f64 {
    todo!("C1a: shrink_l2")
}

/// C1b. The ℓ1 penalty λ|c|: the soft threshold. 0 if `|w| ≤ λ`, otherwise `w` moves
/// by λ towards 0.
pub fn soft(w: f64, lambda: f64) -> f64 {
    todo!("C1b: soft")
}

/// C1c. The ℓ0 penalty (λ²/2 per nonzero): the hard threshold. 0 if `|w| ≤ λ`,
/// otherwise `w` unchanged (keep or kill).
pub fn hard(w: f64, lambda: f64) -> f64 {
    todo!("C1c: hard")
}

/// C2. How many coefficients are needed for the share `fraction` (e.g. 0.95) of the energy
/// (the sum of squares): sort the squares from large to small and add them up until the
/// sum reaches `fraction` of the total.
///
/// Hint: `energy.sort_by(|a, b| b.total_cmp(a))` sorts a `Vec<f64>` from large to small.
pub fn count_for_energy(coeffs: &[f64], fraction: f64) -> usize {
    todo!("C2: count_for_energy")
}

/// C3. λ_max of the lasso: the largest correlation `|a_jᵀ y|` of a column of `a` with the
/// data. For every λ at or above it, the lasso solution is all zeros.
///
/// Hint: column `j` is `a[0][j], a[1][j], ...`; the number of columns is `a[0].len()`.
pub fn lambda_max(a: &[Vec<f64>], y: &[f64]) -> f64 {
    todo!("C3: lambda_max")
}

/// C4. The point of the ℓ1 ball `|x_1| + |x_2| + ... ≤ budget` closest to `v` (the
/// "budget" picture). Outside the ball, it is the soft threshold of `v` at the level θ
/// that makes the ℓ1 norm of the result exactly `budget`.
///
/// Hint: if `v` is inside the ball, return it. Otherwise sort the magnitudes `|v_i|` from
/// large to small. If the `k` largest survive, each loses θ = (their sum − budget) / k.
/// Try k = 1, 2, ... and keep the last θ for which the k-th magnitude is still above θ.
/// Then soft-threshold every `v_i` at θ (use your `soft`).
pub fn project_l1_ball(v: &[f64], budget: f64) -> Vec<f64> {
    todo!("C4: project_l1_ball")
}
