// Model solution for the coding exercises of "Continuous MRFs: quadratic smoothing and total
// variation".

use imgcourse::Image;

/// The penalty `ρ(d)` on a difference `d` between neighbours.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Penalty {
    /// `d²/2`
    Quadratic,
    /// `|d|` (total variation)
    Tv,
    /// `d²/2` for `|d| ≤ δ`, `δ(|d| − δ/2)` beyond
    Huber(f32),
}

/// C1a. The penalty `ρ(d)` on a difference `d` between neighbours.
pub fn penalty(d: f32, rho: Penalty) -> f32 {
    match rho {
        Penalty::Quadratic => d * d / 2.0,
        Penalty::Tv => d.abs(),
        Penalty::Huber(delta) => {
            if d.abs() <= delta {
                d * d / 2.0
            } else {
                delta * (d.abs() - delta / 2.0)
            }
        }
    }
}

/// C1b. The energy of a 1D signal `x` for the data `y`:
/// `E = ½ Σ_p (x_p − y_p)² + λ Σ_p ρ(x_{p+1} − x_p)`.
pub fn energy_1d(x: &[f32], y: &[f32], lambda: f32, rho: Penalty) -> f32 {
    let mut e = 0.0;
    for p in 0..x.len() {
        e += 0.5 * (x[p] - y[p]).powi(2);
    }
    for p in 0..x.len().saturating_sub(1) {
        e += lambda * penalty(x[p + 1] - x[p], rho);
    }
    e
}

/// C2. Quadratic smoothing of a 1D signal: solve `(I + λL) x = y`, with the path Laplacian
/// `L` (diagonal 1, 2, …, 2, 1, and −1 next to it), by elimination for a tridiagonal system.
pub fn quadratic_1d(y: &[f32], lambda: f32) -> Vec<f32> {
    let n = y.len();
    if n == 1 {
        return y.to_vec();
    }
    // row p: −λ x_{p−1} + diag_p x_p − λ x_{p+1} = y_p
    let diag: Vec<f32> = (0..n).map(|p| if p == 0 || p == n - 1 { 1.0 + lambda } else { 1.0 + 2.0 * lambda }).collect();
    let off = -lambda;
    // forward: remove x_{p−1} from row p (c = new upper entry / diagonal, d = new right side)
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    c[0] = off / diag[0];
    d[0] = y[0] / diag[0];
    for p in 1..n {
        let m = diag[p] - off * c[p - 1];
        c[p] = off / m;
        d[p] = (y[p] - off * d[p - 1]) / m;
    }
    // backward: x_p = d_p − c_p x_{p+1}
    let mut x = vec![0.0; n];
    x[n - 1] = d[n - 1];
    for p in (0..n - 1).rev() {
        x[p] = d[p] - c[p] * x[p + 1];
    }
    x
}

/// C3. Total-variation smoothing of a 1D signal: minimise `½ Σ (x_p − y_p)² + λ Σ |x_{p+1} − x_p|`.
/// Each pair `(p, p+1)` gets a number `z_p` between `−λ` and `λ` (`λ · g_p` of the math slide);
/// then `x_p = y_p − (z_{p−1} − z_p)` (with `z = 0` beyond the ends). Repeat: compute `x`,
/// then move every `z_p` by `¼ (x_{p+1} − x_p)` and clip it to `[−λ, λ]`.
pub fn tv_1d(y: &[f32], lambda: f32, iterations: usize) -> Vec<f32> {
    let n = y.len();
    let mut z = vec![0.0f32; n.saturating_sub(1)];
    let mut x = y.to_vec();
    for _ in 0..iterations {
        for p in 0..n {
            let left = if p > 0 { z[p - 1] } else { 0.0 };
            let right = if p + 1 < n { z[p] } else { 0.0 };
            x[p] = y[p] - (left - right);
        }
        for p in 0..n.saturating_sub(1) {
            z[p] = (z[p] + 0.25 * (x[p + 1] - x[p])).clamp(-lambda, lambda);
        }
    }
    for p in 0..n {
        let left = if p > 0 { z[p - 1] } else { 0.0 };
        let right = if p + 1 < n { z[p] } else { 0.0 };
        x[p] = y[p] - (left - right);
    }
    x
}

/// C4. Quadratic smoothing of an image (4-neighbours): solve `(I + λL) x = y` by Gauss–Seidel
/// sweeps. Each sweep sets, pixel by pixel in place, `x_p = (y_p + λ Σ_q x_q) / (1 + λ · deg_p)`
/// (the sum over the neighbours inside the image, `deg_p` their number), starting from `y`.
pub fn quadratic_2d(y: &Image<f32>, lambda: f32, sweeps: usize) -> Image<f32> {
    let (w, h) = (y.width(), y.height());
    let mut x = y.clone();
    for _ in 0..sweeps {
        for py in 0..h {
            for px in 0..w {
                let mut sum = 0.0;
                let mut deg = 0.0;
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (qx, qy) = (px as isize + dx, py as isize + dy);
                    if x.contains(qx, qy) {
                        sum += x[(qx as usize, qy as usize)];
                        deg += 1.0;
                    }
                }
                x[(px, py)] = (y[(px, py)] + lambda * sum) / (1.0 + lambda * deg);
            }
        }
    }
    x
}

/// C5. Total-variation smoothing of an image, `½ Σ (x_p − y_p)² + λ Σ |x_p − x_q|` over the
/// pairs of 4-neighbours: C3 with one `z` per horizontal pair (`zx`, between `(x, y)` and
/// `(x+1, y)`) and one per vertical pair (`zy`), and step `1/8` instead of `1/4`.
pub fn tv_2d(y: &Image<f32>, lambda: f32, iterations: usize) -> Image<f32> {
    let (w, h) = (y.width(), y.height());
    let mut zx = Image::new(w, h, 0.0f32); // the last column is never used
    let mut zy = Image::new(w, h, 0.0f32); // the last row is never used
    let compute_x = |zx: &Image<f32>, zy: &Image<f32>| {
        Image::from_fn(w, h, |px, py| {
            let left = if px > 0 { zx[(px - 1, py)] } else { 0.0 };
            let right = if px + 1 < w { zx[(px, py)] } else { 0.0 };
            let up = if py > 0 { zy[(px, py - 1)] } else { 0.0 };
            let down = if py + 1 < h { zy[(px, py)] } else { 0.0 };
            y[(px, py)] - (left - right) - (up - down)
        })
    };
    for _ in 0..iterations {
        let x = compute_x(&zx, &zy);
        for py in 0..h {
            for px in 0..w {
                if px + 1 < w {
                    zx[(px, py)] = (zx[(px, py)] + 0.125 * (x[(px + 1, py)] - x[(px, py)])).clamp(-lambda, lambda);
                }
                if py + 1 < h {
                    zy[(px, py)] = (zy[(px, py)] + 0.125 * (x[(px, py + 1)] - x[(px, py)])).clamp(-lambda, lambda);
                }
            }
        }
    }
    compute_x(&zx, &zy)
}
