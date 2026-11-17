// Model solution for the coding exercises of "Spectral partitioning and normalized cuts".
//
// A graph with n nodes is its symmetric matrix of affinities `w[p][q]` (0 on the diagonal).

/// C1. Affinities between points: w = exp(−d² / 2σ²) for every pair at distance d, and 0 on
/// the diagonal (no node is joined to itself).
pub fn affinities(points: &[[f64; 2]], sigma: f64) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut w = vec![vec![0.0; n]; n];
    for p in 0..n {
        for q in 0..n {
            if p != q {
                let d2 = (points[p][0] - points[q][0]).powi(2) + (points[p][1] - points[q][1]).powi(2);
                w[p][q] = (-d2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }
    w
}

/// C2a. The degrees d_p = Σ_q w_pq and the graph Laplacian L = D − W.
pub fn laplacian(w: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = w.len();
    let degrees: Vec<f64> = w.iter().map(|row| row.iter().sum()).collect();
    let mut l = vec![vec![0.0; n]; n];
    for p in 0..n {
        for q in 0..n {
            l[p][q] = if p == q { degrees[p] } else { 0.0 } - w[p][q];
        }
    }
    (degrees, l)
}

/// C2b. The quadratic form xᵀ L x.
pub fn quadratic_form(l: &[Vec<f64>], x: &[f64]) -> f64 {
    let n = x.len();
    let mut sum = 0.0;
    for p in 0..n {
        for q in 0..n {
            sum += x[p] * l[p][q] * x[q];
        }
    }
    sum
}

/// C3. The cut between A (`in_a[p]` true) and B (the rest), and the normalized cut
/// Ncut = cut / vol(A) + cut / vol(B), with vol = the sum of the degrees.
pub fn ncut(w: &[Vec<f64>], in_a: &[bool]) -> (f64, f64) {
    let n = w.len();
    let (mut cut, mut vol_a, mut vol_b) = (0.0, 0.0, 0.0);
    for p in 0..n {
        for q in 0..n {
            if in_a[p] {
                vol_a += w[p][q];
                if !in_a[q] {
                    cut += w[p][q];
                }
            } else {
                vol_b += w[p][q];
            }
        }
    }
    (cut, cut / vol_a + cut / vol_b)
}

/// C4. The normalized Fiedler vector: the eigenvector x of L x = μ D x for the second-smallest
/// μ (math slide "A generalized eigenproblem"), by power iteration without solving any
/// equation. With y = D^{1/2} x the problem is N y = μ y, N = I − D^{−1/2} W D^{−1/2}. The
/// matrix M = 2I − N = I + D^{−1/2} W D^{−1/2} has the eigenvalues 2 − μ, so its largest are
/// 2 (the trivial y₁ = D^{1/2} 1) and 2 − μ₂. Power iteration on M, keeping y orthogonal to
/// y₁, finds y₂. Returns μ₂ and x = D^{−1/2} y₂ (scaled so that its largest entry is ±1).
pub fn fiedler(w: &[Vec<f64>], iterations: usize) -> (f64, Vec<f64>) {
    let n = w.len();
    let (degrees, _) = laplacian(w);
    let sqrt_d: Vec<f64> = degrees.iter().map(|d| d.sqrt()).collect();
    // the trivial eigenvector, length 1
    let norm = sqrt_d.iter().map(|v| v * v).sum::<f64>().sqrt();
    let y1: Vec<f64> = sqrt_d.iter().map(|v| v / norm).collect();
    // take out the part along y₁; then scale to length 1
    let remove_y1 = |y: &mut Vec<f64>| {
        let dot: f64 = y.iter().zip(&y1).map(|(a, b)| a * b).sum();
        for (v, u) in y.iter_mut().zip(&y1) {
            *v -= dot * u;
        }
    };
    let normalise = |y: &mut Vec<f64>| {
        let len = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in y.iter_mut() {
            *v /= len;
        }
    };
    // start from any vector that is not y₁: here 1, 2, 3, ...
    let mut y: Vec<f64> = (0..n).map(|p| (p + 1) as f64).collect();
    remove_y1(&mut y);
    normalise(&mut y);
    let mut lambda = 0.0;
    for _ in 0..iterations {
        // z = M y = y + D^{−1/2} W D^{−1/2} y
        let mut z = y.clone();
        for p in 0..n {
            for q in 0..n {
                z[p] += w[p][q] * y[q] / (sqrt_d[p] * sqrt_d[q]);
            }
        }
        remove_y1(&mut z); // (rounding errors would bring y₁ back)
        lambda = z.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>(); // yᵀ M y: the eigenvalue of M
        normalise(&mut z);
        y = z;
    }
    let mu = 2.0 - lambda;
    let mut x: Vec<f64> = y.iter().zip(&sqrt_d).map(|(v, s)| v / s).collect();
    let largest = x.iter().fold(0.0f64, |m, v| if v.abs() > m.abs() { *v } else { m });
    for v in x.iter_mut() {
        *v /= largest.abs();
    }
    (mu, x)
}

/// C5. The sweep (slide "From eigenvector to segmentation"): sort the nodes by `x`, try
/// every split "the k smallest against the rest" (k = 1, ..., n − 1) and keep the one with
/// the smallest Ncut. Returns that split (`true` = in A, the nodes with the small values)
/// and its Ncut.
pub fn sweep(w: &[Vec<f64>], x: &[f64]) -> (Vec<bool>, f64) {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let mut best = (vec![false; n], f64::INFINITY);
    let mut in_a = vec![false; n];
    for k in 0..n - 1 {
        in_a[order[k]] = true;
        let (_, nc) = ncut(w, &in_a);
        if nc < best.1 {
            best = (in_a.clone(), nc);
        }
    }
    best
}
