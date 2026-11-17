// Model solution for the coding exercises of "Endmembers: the geometry of mixtures".

use imgcourse::linalg;

/// A matrix, row by row: `m[row][column]`. Spectra are the columns of M (rows = channels);
/// in Y the pixels are the columns (rows = channels, one column per pixel).
pub type Matrix = Vec<Vec<f32>>;

/// R1. Non-negative least squares for a few components: the abundances a ≥ 0 that minimise
/// ‖y − M a‖². Try every subset of the components ("which ones are not zero"): solve ordinary
/// least squares with only those columns; keep the solutions whose abundances are all ≥ 0;
/// return the one with the smallest residual (the others are 0). The empty subset (all zero)
/// is allowed too.
pub fn nnls(m: &Matrix, y: &[f32]) -> Vec<f32> {
    let (n, k) = (m.len(), m[0].len());
    let yd: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let mut best = vec![0.0f32; k];
    let mut best_err = y.iter().map(|v| v * v).sum::<f32>(); // all zero
    for mask in 1..(1usize << k) {
        let cols: Vec<usize> = (0..k).filter(|&j| mask & (1 << j) != 0).collect();
        let rows: Vec<Vec<f64>> = (0..n).map(|c| cols.iter().map(|&j| m[c][j] as f64).collect()).collect();
        let Some(sol) = linalg::least_squares(&rows, &yd) else { continue };
        if sol.iter().any(|&v| v < 0.0) {
            continue;
        }
        let mut a = vec![0.0f32; k];
        for (i, &j) in cols.iter().enumerate() {
            a[j] = sol[i] as f32;
        }
        let err: f32 = (0..n).map(|c| (y[c] - (0..k).map(|j| m[c][j] * a[j]).sum::<f32>()).powi(2)).sum();
        if err < best_err {
            best_err = err;
            best = a;
        }
    }
    best
}

/// R2. The fractions (barycentric coordinates) of the point `p` in the triangle with corners
/// `e[0]`, `e[1]`, `e[2]`: `p = a0 e0 + a1 e1 + a2 e2` with a0 + a1 + a2 = 1. A negative
/// fraction means the point is outside the triangle.
pub fn fractions(p: [f32; 2], e: [[f32; 2]; 3]) -> [f32; 3] {
    let (u, v, d) = ([e[1][0] - e[0][0], e[1][1] - e[0][1]], [e[2][0] - e[0][0], e[2][1] - e[0][1]], [p[0] - e[0][0], p[1] - e[0][1]]);
    let det = u[0] * v[1] - v[0] * u[1];
    let a1 = (d[0] * v[1] - v[0] * d[1]) / det;
    let a2 = (u[0] * d[1] - d[0] * u[1]) / det;
    [1.0 - a1 - a2, a1, a2]
}

/// R3. The singular values of the pixel matrix Y (channels × pixels), largest first: the
/// square roots of the eigenvalues of `Y Yᵀ` (channels × channels, no centring).
pub fn singular_values(y: &Matrix) -> Vec<f32> {
    let n = y.len();
    let mut yyt = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            yyt[i][j] = y[i].iter().zip(&y[j]).map(|(&a, &b)| a as f64 * b as f64).sum();
        }
    }
    let (values, _) = linalg::sym_eigen(&yyt);
    values.iter().map(|&v| v.max(0.0).sqrt() as f32).collect()
}

/// R4. Vertex finding (the idea behind VCA) on pixels given as vectors: pick the pixel with
/// the largest length; then repeat: remove from every pixel its part along the directions
/// found so far (project onto their orthogonal complement), and pick the pixel whose
/// remaining part is longest. Returns the indices of the `k` pixels picked.
pub fn find_vertices(pixels: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut directions: Vec<Vec<f32>> = vec![];
    let mut picked = vec![];
    for _ in 0..k {
        let mut best = (0, -1.0f32, vec![]);
        for (i, p) in pixels.iter().enumerate() {
            // the part of p perpendicular to every direction found so far
            let mut rest = p.clone();
            for d in &directions {
                let dot: f32 = rest.iter().zip(d).map(|(a, b)| a * b).sum();
                for c in 0..rest.len() {
                    rest[c] -= dot * d[c];
                }
            }
            let len = rest.iter().map(|v| v * v).sum::<f32>().sqrt();
            if len > best.1 {
                best = (i, len, rest);
            }
        }
        picked.push(best.0);
        directions.push(best.2.iter().map(|v| v / best.1).collect());
    }
    picked
}

/// R5a. One multiplicative update of the abundances in NMF (slide "The math: the NMF updates"):
/// `H ← H ⊙ (WᵀY) ⊘ (WᵀW H)`, element by element. W is channels × K, H is K × pixels.
pub fn update_h(y: &Matrix, w: &Matrix, h: &mut Matrix) {
    let (n, k, p) = (w.len(), w[0].len(), y[0].len());
    let mut wtw = vec![vec![0.0f32; k]; k];
    for i in 0..k {
        for j in 0..k {
            wtw[i][j] = (0..n).map(|c| w[c][i] * w[c][j]).sum();
        }
    }
    // one pixel (column of H) at a time: all K entries from the old column
    for q in 0..p {
        let old: Vec<f32> = (0..k).map(|i| h[i][q]).collect();
        for i in 0..k {
            let wty: f32 = (0..n).map(|c| w[c][i] * y[c][q]).sum();
            let wtwh: f32 = (0..k).map(|j| wtw[i][j] * old[j]).sum();
            h[i][q] = old[i] * wty / (wtwh + 1e-12);
        }
    }
}

/// R5b. One multiplicative update of the spectra in NMF: `W ← W ⊙ (Y Hᵀ) ⊘ (W H Hᵀ)`.
pub fn update_w(y: &Matrix, w: &mut Matrix, h: &Matrix) {
    let (n, k, p) = (w.len(), w[0].len(), y[0].len());
    let mut hht = vec![vec![0.0f32; k]; k];
    for i in 0..k {
        for j in 0..k {
            hht[i][j] = (0..p).map(|q| h[i][q] * h[j][q]).sum();
        }
    }
    for c in 0..n {
        let yht: Vec<f32> = (0..k).map(|j| (0..p).map(|q| y[c][q] * h[j][q]).sum()).collect();
        let whht: Vec<f32> = (0..k).map(|j| (0..k).map(|i| w[c][i] * hht[i][j]).sum()).collect();
        for j in 0..k {
            w[c][j] *= yht[j] / (whht[j] + 1e-12);
        }
    }
}
