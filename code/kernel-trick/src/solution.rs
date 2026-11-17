// Model solution for the coding exercises of "Kernel functions and the kernel trick".

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
    [x[0] * x[0], 2.0f32.sqrt() * x[0] * x[1], x[1] * x[1]]
}

/// C2. The kernel function `k(x, z)` for vectors of any length.
pub fn kernel(k: Kernel, x: &[f32], z: &[f32]) -> f32 {
    match k {
        Kernel::Linear => dot(x, z),
        Kernel::Polynomial { c, degree } => (dot(x, z) + c).powi(degree),
        Kernel::Rbf { gamma } => {
            let mut d2 = 0.0;
            for i in 0..x.len() {
                d2 += (x[i] - z[i]) * (x[i] - z[i]);
            }
            (-gamma * d2).exp()
        }
    }
}

/// The dot product `x · z`. (Given.)
pub fn dot(x: &[f32], z: &[f32]) -> f32 {
    x.iter().zip(z).map(|(a, b)| a * b).sum()
}

/// C3. The decision value of a kernel SVM, `f(x) = Σ_i α_i y_i k(x_i, x) + b` (slide "The RBF
/// kernel function"). Only the support vectors (`α_i > 0`) contribute.
pub fn decision(points: &[Vec<f32>], labels: &[f32], alpha: &[f32], b: f32, k: Kernel, x: &[f32]) -> f32 {
    let mut f = b;
    for i in 0..points.len() {
        if alpha[i] > 0.0 {
            f += alpha[i] * labels[i] * kernel(k, &points[i], x);
        }
    }
    f
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
pub fn train_smo(points: &[Vec<f32>], labels: &[f32], k: Kernel, c: f32, tol: f32, max_steps: usize) -> (Vec<f32>, f32) {
    let n = points.len();
    let km: Vec<Vec<f32>> = (0..n).map(|i| (0..n).map(|j| kernel(k, &points[i], &points[j])).collect()).collect();
    let y = labels;
    let mut alpha = vec![0.0f32; n];
    let mut g = vec![0.0f32; n]; // g_k = Σ_l α_l y_l K_lk, all 0 at the start
    let (mut f_i, mut f_j) = (0.0, 0.0);
    for _ in 0..max_steps {
        // choose the most violating pair
        let (mut i, mut j) = (usize::MAX, usize::MAX);
        (f_i, f_j) = (f32::INFINITY, f32::NEG_INFINITY);
        for t in 0..n {
            let f = g[t] - y[t];
            let up = (y[t] > 0.0 && alpha[t] < c) || (y[t] < 0.0 && alpha[t] > 0.0);
            let low = (y[t] < 0.0 && alpha[t] < c) || (y[t] > 0.0 && alpha[t] > 0.0);
            if up && f < f_i {
                (i, f_i) = (t, f);
            }
            if low && f > f_j {
                (j, f_j) = (t, f);
            }
        }
        if i == usize::MAX || j == usize::MAX || f_j - f_i < 2.0 * tol {
            break;
        }
        // the best change of α_j along Σ α y = const (a 1D quadratic), clipped to [lo, hi]
        let eta = (km[i][i] + km[j][j] - 2.0 * km[i][j]).max(1e-12);
        let (lo, hi) = if y[i] != y[j] {
            ((alpha[j] - alpha[i]).max(0.0), (c + alpha[j] - alpha[i]).min(c))
        } else {
            ((alpha[i] + alpha[j] - c).max(0.0), (alpha[i] + alpha[j]).min(c))
        };
        let new_j = (alpha[j] + y[j] * (f_i - f_j) / eta).clamp(lo, hi);
        let new_i = alpha[i] + y[i] * y[j] * (alpha[j] - new_j);
        let (di, dj) = (new_i - alpha[i], new_j - alpha[j]);
        (alpha[i], alpha[j]) = (new_i, new_j);
        for t in 0..n {
            g[t] += di * y[i] * km[i][t] + dj * y[j] * km[j][t];
        }
    }
    (alpha, -(f_i + f_j) / 2.0)
}
