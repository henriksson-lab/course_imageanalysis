// Model solution for the coding exercises of "Solving with ℓ1: thresholding iterations and
// pursuit". Vectors and matrices are f64, as in imgcourse::linalg; a matrix is `a[row][column]`.

use imgcourse::linalg;

/// `A x`.
pub fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x).map(|(r, v)| r * v).sum()).collect()
}

/// `Aᵀ r`.
pub fn mat_t_vec(a: &[Vec<f64>], r: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a[0].len()];
    for (row, ri) in a.iter().zip(r) {
        for (o, aij) in out.iter_mut().zip(row) {
            *o += aij * ri;
        }
    }
    out
}

/// The ℓ1 objective `½‖A x − y‖² + λ‖x‖₁`.
pub fn objective(a: &[Vec<f64>], y: &[f64], x: &[f64], lambda: f64) -> f64 {
    let misfit: f64 = mat_vec(a, x).iter().zip(y).map(|(p, d)| (p - d).powi(2)).sum();
    0.5 * misfit + lambda * x.iter().map(|v| v.abs()).sum::<f64>()
}

/// C1a. Soft-threshold every value at `t`: 0 if `|v| ≤ t`, otherwise move `t` towards 0.
pub fn soft_threshold(v: &[f64], t: f64) -> Vec<f64> {
    v.iter().map(|&x| if x > t { x - t } else if x < -t { x + t } else { 0.0 }).collect()
}

/// C1b. One ISTA iteration: a gradient (Landweber) step `z = x + τ Aᵀ(y − A x)`, then the
/// soft threshold of `z` at τλ.
pub fn ista_step(a: &[Vec<f64>], y: &[f64], x: &[f64], lambda: f64, tau: f64) -> Vec<f64> {
    let ax = mat_vec(a, x);
    let r: Vec<f64> = y.iter().zip(&ax).map(|(d, p)| d - p).collect(); // residual
    let g = mat_t_vec(a, &r); // pushed back with Aᵀ
    let z: Vec<f64> = x.iter().zip(&g).map(|(xi, gi)| xi + tau * gi).collect();
    soft_threshold(&z, tau * lambda)
}

/// C2. ISTA: start at 0 and repeat `ista_step` `iterations` times.
pub fn ista(a: &[Vec<f64>], y: &[f64], lambda: f64, tau: f64, iterations: usize) -> Vec<f64> {
    let mut x = vec![0.0; a[0].len()];
    for _ in 0..iterations {
        x = ista_step(a, y, &x, lambda, tau);
    }
    x
}

/// C3. L, the largest gain of `AᵀA` (its largest eigenvalue), by the power method: start
/// from any vector, apply `AᵀA` and rescale to length 1, `iterations` times. The safe ISTA
/// step is τ = 1/L.
pub fn largest_gain(a: &[Vec<f64>], iterations: usize) -> f64 {
    let n = a[0].len();
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut gain = 0.0;
    for _ in 0..iterations {
        let w = mat_t_vec(a, &mat_vec(a, &v));
        gain = w.iter().map(|x| x * x).sum::<f64>().sqrt(); // |AᵀA v| for a unit v
        v = w.iter().map(|x| x / gain).collect();
    }
    gain
}

/// C4. FISTA: ISTA steps taken from an extrapolated point. With `t = 1` at the start:
/// `x_new = ista_step(z)`, `t_new = (1 + √(1 + 4t²)) / 2`,
/// `z = x_new + (t − 1)/t_new · (x_new − x)`.
pub fn fista(a: &[Vec<f64>], y: &[f64], lambda: f64, tau: f64, iterations: usize) -> Vec<f64> {
    let mut x = vec![0.0; a[0].len()];
    let mut z = x.clone();
    let mut t = 1.0f64;
    for _ in 0..iterations {
        let x_new = ista_step(a, y, &z, lambda, tau);
        let t_new = (1.0 + (1.0 + 4.0 * t * t).sqrt()) / 2.0;
        let momentum = (t - 1.0) / t_new;
        z = x_new.iter().zip(&x).map(|(xn, xo)| xn + momentum * (xn - xo)).collect();
        x = x_new;
        t = t_new;
    }
    x
}

/// C5. Orthogonal matching pursuit: pick the column most correlated with the residual
/// (`|a_jᵀ r| / ‖a_j‖`), refit all chosen columns by least squares, update the residual;
/// stop after `max_atoms` columns or when `‖r‖² ≤ tolerance`. Returns the full coefficient
/// vector (zeros for the columns that were not chosen).
pub fn omp(a: &[Vec<f64>], y: &[f64], max_atoms: usize, tolerance: f64) -> Vec<f64> {
    let (m, n) = (a.len(), a[0].len());
    let norms: Vec<f64> = (0..n).map(|j| (0..m).map(|i| a[i][j] * a[i][j]).sum::<f64>().sqrt()).collect();
    let mut chosen: Vec<usize> = Vec::new();
    let mut x = vec![0.0; n];
    let mut r = y.to_vec();
    while chosen.len() < max_atoms && r.iter().map(|v| v * v).sum::<f64>() > tolerance {
        // 1. the column most correlated with the residual
        let corr = mat_t_vec(a, &r);
        let mut best = 0;
        for j in 1..n {
            if corr[j].abs() / norms[j] > corr[best].abs() / norms[best] {
                best = j;
            }
        }
        if chosen.contains(&best) {
            break; // the residual is orthogonal to everything left (numerically)
        }
        chosen.push(best);
        // 2. refit all chosen columns: least squares with the matrix A_S (m × |S|)
        let a_s: Vec<Vec<f64>> = (0..m).map(|i| chosen.iter().map(|&j| a[i][j]).collect()).collect();
        let Some(coef) = linalg::least_squares(&a_s, y) else { break };
        x = vec![0.0; n];
        for (k, &j) in chosen.iter().enumerate() {
            x[j] = coef[k];
        }
        // 3. the new residual
        let ax = mat_vec(a, &x);
        r = y.iter().zip(&ax).map(|(d, p)| d - p).collect();
    }
    x
}
