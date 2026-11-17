// Model solution for the coding exercises of "Least-squares unmixing".

use imgcourse::{linalg, Image};

/// A mixing matrix, row by row: `m[c][k]` = channel c of the spectrum of component k
/// (N rows = channels, K columns = components).
pub type Matrix = Vec<Vec<f32>>;

/// C1. Least-squares unmixing of one pixel by the normal equations: build `MᵀM` (K × K) and
/// `Mᵀy` (K numbers), then solve `MᵀM a = Mᵀy`.
pub fn unmix_pixel(m: &Matrix, y: &[f32]) -> Vec<f32> {
    let (n, k) = (m.len(), m[0].len());
    let mut mtm = vec![vec![0.0f64; k]; k];
    let mut mty = vec![0.0f64; k];
    for i in 0..k {
        for j in 0..k {
            for c in 0..n {
                mtm[i][j] += m[c][i] as f64 * m[c][j] as f64;
            }
        }
        for c in 0..n {
            mty[i] += m[c][i] as f64 * y[c] as f64;
        }
    }
    let a = linalg::solve(&mtm, &mty).expect("the spectra must be independent");
    a.iter().map(|&v| v as f32).collect()
}

/// C2a. The pseudo-inverse `M⁺ = (MᵀM)⁻¹ Mᵀ` (K × N). Column c of M⁺ is the unmixing of a
/// pixel that is 1 in channel c and 0 elsewhere.
pub fn pseudo_inverse(m: &Matrix) -> Matrix {
    let (n, k) = (m.len(), m[0].len());
    let mut p = vec![vec![0.0; n]; k];
    for c in 0..n {
        let mut e = vec![0.0; n];
        e[c] = 1.0;
        let col = unmix_pixel(m, &e);
        for i in 0..k {
            p[i][c] = col[i];
        }
    }
    p
}

/// C2b. Unmix a whole lambda stack (one image per channel) with one pseudo-inverse: abundance
/// map k is the weighted sum of the channels with the weights of row k of M⁺.
pub fn unmix_image(stack: &[Image<f32>], m: &Matrix) -> Vec<Image<f32>> {
    let p = pseudo_inverse(m);
    let (w, h) = (stack[0].width(), stack[0].height());
    p.iter()
        .map(|row| {
            let mut a = Image::new(w, h, 0.0);
            for (c, ch) in stack.iter().enumerate() {
                for (x, y) in ch.coords() {
                    a[(x, y)] += row[c] * ch[(x, y)];
                }
            }
            a
        })
        .collect()
}

/// C3a. The residual of one pixel: `r = y − M a`.
pub fn residual(m: &Matrix, y: &[f32], a: &[f32]) -> Vec<f32> {
    let mut r = y.to_vec();
    for c in 0..m.len() {
        for k in 0..a.len() {
            r[c] -= m[c][k] * a[k];
        }
    }
    r
}

/// C3b. The residual map in units of the expected photon noise. With Poisson noise the
/// variance of a channel equals its expected count, so the squared residual should be about
/// as large as the fit, up to the factor (N − K)/N (the residual has only N − K free
/// directions). At every pixel, with `fit = M a`:
/// `sqrt( N/(N − K) · Σ_c r_c² / max(Σ_c fit_c, 1) )`. About 1 where the model explains
/// the pixel.
pub fn residual_map(stack: &[Image<f32>], m: &Matrix, abundances: &[Image<f32>]) -> Image<f32> {
    let (n, k) = (stack.len() as f32, abundances.len() as f32);
    stack[0].map_xy(|x, y, _| {
        let pixel: Vec<f32> = stack.iter().map(|ch| ch[(x, y)]).collect();
        let a: Vec<f32> = abundances.iter().map(|ab| ab[(x, y)]).collect();
        let r = residual(m, &pixel, &a);
        let (mut rr, mut fit) = (0.0, 0.0);
        for c in 0..stack.len() {
            rr += r[c] * r[c];
            fit += pixel[c] - r[c];
        }
        (n / (n - k) * rr / fit.max(1.0)).sqrt()
    })
}

/// C4. The condition number κ = σ_max / σ_min of M: the singular values of M are the square
/// roots of the eigenvalues of MᵀM.
pub fn condition_number(m: &Matrix) -> f32 {
    let k = m[0].len();
    let mut mtm = vec![vec![0.0f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            mtm[i][j] = m.iter().map(|row| row[i] as f64 * row[j] as f64).sum();
        }
    }
    let (values, _) = linalg::sym_eigen(&mtm);
    (values[0] / values[k - 1]).sqrt() as f32
}

/// C5. Weighted least squares of one pixel: solve `MᵀWM a = MᵀWy` with `W = diag(w)`
/// (slide "The math: WLS is maximum likelihood").
pub fn unmix_weighted(m: &Matrix, y: &[f32], w: &[f32]) -> Vec<f32> {
    // Weighting is least squares on rows scaled by sqrt(w): (√w m) a ≈ √w y.
    let scaled: Matrix = m.iter().zip(w).map(|(row, &wc)| row.iter().map(|&v| v * wc.sqrt()).collect()).collect();
    let ys: Vec<f32> = y.iter().zip(w).map(|(&v, &wc)| v * wc.sqrt()).collect();
    unmix_pixel(&scaled, &ys)
}
