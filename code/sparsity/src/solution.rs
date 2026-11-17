// Model solution for the coding exercises of "Sparsity and the ℓ1 penalty".
// Vectors and matrices are f64, as in imgcourse::linalg; a matrix is `a[row][column]`.

/// C1a. The ℓ2 penalty (λ/2)c² for one coefficient: every coefficient shrinks,
/// `c = w / (1 + λ)`, and none becomes 0.
pub fn shrink_l2(w: f64, lambda: f64) -> f64 {
    w / (1.0 + lambda)
}

/// C1b. The ℓ1 penalty λ|c|: the soft threshold. 0 if `|w| ≤ λ`, otherwise `w` moves
/// by λ towards 0.
pub fn soft(w: f64, lambda: f64) -> f64 {
    if w > lambda {
        w - lambda
    } else if w < -lambda {
        w + lambda
    } else {
        0.0
    }
}

/// C1c. The ℓ0 penalty (λ²/2 per nonzero): the hard threshold. 0 if `|w| ≤ λ`,
/// otherwise `w` unchanged (keep or kill).
pub fn hard(w: f64, lambda: f64) -> f64 {
    if w.abs() > lambda {
        w
    } else {
        0.0
    }
}

/// C2. How many coefficients are needed for the share `fraction` (e.g. 0.95) of the energy
/// (the sum of squares): sort the squares from large to small and add them up until the
/// sum reaches `fraction` of the total.
pub fn count_for_energy(coeffs: &[f64], fraction: f64) -> usize {
    let mut energy: Vec<f64> = coeffs.iter().map(|c| c * c).collect();
    energy.sort_by(|a, b| b.total_cmp(a)); // largest first
    let total: f64 = energy.iter().sum();
    let mut sum = 0.0;
    for (i, e) in energy.iter().enumerate() {
        sum += e;
        if sum >= fraction * total {
            return i + 1;
        }
    }
    energy.len()
}

/// C3. λ_max of the lasso: the largest correlation `|a_jᵀ y|` of a column of `a` with the
/// data. For every λ at or above it, the lasso solution is all zeros.
pub fn lambda_max(a: &[Vec<f64>], y: &[f64]) -> f64 {
    let columns = a[0].len();
    let mut best = 0.0f64;
    for j in 0..columns {
        let mut corr = 0.0;
        for i in 0..y.len() {
            corr += a[i][j] * y[i];
        }
        best = best.max(corr.abs());
    }
    best
}

/// C4. The point of the ℓ1 ball `|x_1| + |x_2| + ... ≤ budget` closest to `v` (the
/// "budget" picture). Outside the ball, it is the soft threshold of `v` at the level θ
/// that makes the ℓ1 norm of the result exactly `budget`.
pub fn project_l1_ball(v: &[f64], budget: f64) -> Vec<f64> {
    if v.iter().map(|x| x.abs()).sum::<f64>() <= budget {
        return v.to_vec(); // already inside
    }
    let mut m: Vec<f64> = v.iter().map(|x| x.abs()).collect();
    m.sort_by(|a, b| b.total_cmp(a)); // largest first
    // Keep the k largest: they lose θ = (their sum − budget) / k each. Take the largest k
    // for which the k-th magnitude is still above that θ.
    let mut theta = 0.0;
    let mut sum = 0.0;
    for k in 0..m.len() {
        sum += m[k];
        let t = (sum - budget) / (k + 1) as f64;
        if m[k] > t {
            theta = t;
        } else {
            break;
        }
    }
    v.iter().map(|&x| soft(x, theta)).collect()
}
