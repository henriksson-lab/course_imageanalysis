// Model solution for the coding exercises of "From a spectrum to three numbers".

use imgcourse::linalg;

/// R1. The responses of an observer (eye, camera) to a sampled spectrum: one weighted sum
/// per row of `a`, `c[i] = Σ_j a[i][j] · s[j]` (the matrix–vector product c = A s).
pub fn responses(a: &[Vec<f32>], s: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0; a.len()];
    for i in 0..a.len() {
        for j in 0..s.len() {
            c[i] += a[i][j] * s[j];
        }
    }
    c
}

/// R2. A metamer of the spectrum `s` for the observer `a` (3 rows). First remove from `v`
/// everything the observer can see, keeping its part in the null space of `a`:
/// `p = v − Aᵀ u` with `(A Aᵀ) u = A v` (slide "The math: metamers live in the null space").
/// Then step from `s` along `p` as far as the spectrum stays ≥ 0 everywhere, and take the
/// fraction `share` (between 0 and 1) of that step: `s + share · t_max · p`.
pub fn metamer(a: &[Vec<f32>], s: &[f32], v: &[f32], share: f32) -> Vec<f32> {
    let (k, n) = (a.len(), s.len());
    // A Aᵀ (k × k) and A v (k numbers)
    let mut aat = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for l in 0..k {
            aat[i][l] = (0..n).map(|j| a[i][j] as f64 * a[l][j] as f64).sum();
        }
    }
    let av: Vec<f64> = responses(a, v).iter().map(|&x| x as f64).collect();
    let u = linalg::solve(&aat, &av).expect("the rows of A must be independent");
    // p = v − Aᵀ u: the part of v that the observer cannot see
    let mut p = v.to_vec();
    for j in 0..n {
        for i in 0..k {
            p[j] -= (a[i][j] as f64 * u[i]) as f32;
        }
    }
    // the largest step that keeps every bin ≥ 0
    let mut t_max = f32::INFINITY;
    for j in 0..n {
        if p[j] < 0.0 {
            t_max = t_max.min(s[j] / -p[j]);
        }
    }
    if !t_max.is_finite() {
        t_max = 0.0;
    }
    (0..n).map(|j| s[j] + share * t_max * p[j]).collect()
}

/// R3. The chromaticity (x, y) of a colour (X, Y, Z): divide by X + Y + Z.
pub fn chromaticity(xyz: [f32; 3]) -> [f32; 2] {
    let sum = xyz[0] + xyz[1] + xyz[2];
    [xyz[0] / sum, xyz[1] / sum]
}

/// R4. The barycentric coordinates (α, β, γ) of the point `p` in the triangle with corners
/// `r`, `g`, `b`: `p = α r + β g + γ b` with α + β + γ = 1. The point is inside the triangle
/// (in the gamut) when all three are ≥ 0.
pub fn barycentric(p: [f32; 2], r: [f32; 2], g: [f32; 2], b: [f32; 2]) -> [f32; 3] {
    let (a11, a12, a21, a22) = (r[0] - b[0], g[0] - b[0], r[1] - b[1], g[1] - b[1]);
    let (c1, c2) = (p[0] - b[0], p[1] - b[1]);
    let det = a11 * a22 - a12 * a21;
    let alpha = (c1 * a22 - a12 * c2) / det;
    let beta = (a11 * c2 - a21 * c1) / det;
    [alpha, beta, 1.0 - alpha - beta]
}
