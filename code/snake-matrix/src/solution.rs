// Model solution for the coding exercises of "The snake as a linear system".

use imgcourse::{linalg, Border, Image};

/// C1. The matrix `A` of the internal forces (slide "All beads at once: a matrix"): force
/// `= −A x`. Row `i` has `2α + 6β` on the diagonal, `−α − 4β` at `i ± 1` and `β` at `i ± 2`,
/// with the columns taken modulo `n` (the curve is closed). Entries that land on the same
/// column (possible for `n < 5`) add up.
pub fn snake_matrix(n: usize, alpha: f64, beta: f64) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; n]; n];
    let stencil = [(-2, beta), (-1, -alpha - 4.0 * beta), (0, 2.0 * alpha + 6.0 * beta), (1, -alpha - 4.0 * beta), (2, beta)];
    for i in 0..n {
        for (offset, value) in stencil {
            let j = (i as isize + offset).rem_euclid(n as isize) as usize;
            a[i][j] += value;
        }
    }
    a
}

/// C2. One explicit step for one coordinate (slide "Why the explicit step explodes"):
/// `x_new = x − τ (A x − f)`, with `f` the image force on every bead.
pub fn explicit_step(a: &[Vec<f64>], x: &[f64], f: &[f64], tau: f64) -> Vec<f64> {
    let n = x.len();
    let mut new = vec![0.0; n];
    for i in 0..n {
        let mut ax = 0.0;
        for j in 0..n {
            ax += a[i][j] * x[j];
        }
        new[i] = x[i] - tau * (ax - f[i]);
    }
    new
}

/// C3. "Factor once" (slide "The semi-implicit step"): the inverse of `A + γI`, `γ = 1/τ`,
/// computed column by column: column `j` solves `(A + γI) c = e_j` (`e_j` = 1 at `j`, else 0).
pub fn semi_implicit_inverse(a: &[Vec<f64>], tau: f64) -> Vec<Vec<f64>> {
    let n = a.len();
    let gamma = 1.0 / tau;
    let mut m = a.to_vec();
    for i in 0..n {
        m[i][i] += gamma;
    }
    let mut inv = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let c = linalg::solve(&m, &e).expect("A + γI is always invertible");
        for i in 0..n {
            inv[i][j] = c[i];
        }
    }
    inv
}

/// C4. One semi-implicit step for one coordinate: `x_new = (A + γI)⁻¹ (γ x + f)`, with the
/// inverse from C3.
pub fn semi_implicit_step(inv: &[Vec<f64>], x: &[f64], f: &[f64], tau: f64) -> Vec<f64> {
    let n = x.len();
    let rhs: Vec<f64> = (0..n).map(|i| x[i] / tau + f[i]).collect();
    let mut new = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            new[i] += inv[i][j] * rhs[j];
        }
    }
    new
}

/// C5. Gradient vector flow (slide "Gradient vector flow"): start from `v = g` (the edge
/// force, `(gx, gy)`) and repeat `iterations` times, for both components,
/// `v ← v + Δt [ μ ∇²v − |g|² (v − g) ]`, with the 5-point Laplacian (neighbours outside the
/// image: copy the edge). Every step reads only the old field.
pub fn gvf(gx: &Image<f32>, gy: &Image<f32>, mu: f32, dt: f32, iterations: usize) -> (Image<f32>, Image<f32>) {
    let mag2 = gx.zip_map(gy, |a, b| a * a + b * b);
    let (mut u, mut v) = (gx.clone(), gy.clone());
    for _ in 0..iterations {
        let update = |field: &Image<f32>, g: &Image<f32>| {
            field.map_xy(|x, y, val| {
                let (xi, yi) = (x as isize, y as isize);
                let lap = field.get_border(xi - 1, yi, Border::Clamp)
                    + field.get_border(xi + 1, yi, Border::Clamp)
                    + field.get_border(xi, yi - 1, Border::Clamp)
                    + field.get_border(xi, yi + 1, Border::Clamp)
                    - 4.0 * val;
                val + dt * (mu * lap - mag2[(x, y)] * (val - g[(x, y)]))
            })
        };
        let (nu, nv) = (update(&u, gx), update(&v, gy));
        u = nu;
        v = nv;
    }
    (u, v)
}

// ---- Helpers from earlier units (filled in, not part of the exercise) ----

/// Gaussian blur of width `sigma` in two passes, copying the edge outside the image.
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    if sigma <= 0.0 {
        return img.clone();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let mut w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= total);
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Clamp);
            }
            sum
        })
    };
    pass(&pass(img, true), false)
}

/// The gradient `(∂/∂x, ∂/∂y)` of an image by central differences, copying the edge.
pub fn gradient(img: &Image<f32>) -> (Image<f32>, Image<f32>) {
    let d = |dx: isize, dy: isize| {
        img.map_xy(|x, y, _| {
            let (x, y) = (x as isize, y as isize);
            (img.get_border(x + dx, y + dy, Border::Clamp) - img.get_border(x - dx, y - dy, Border::Clamp)) / 2.0
        })
    };
    (d(1, 0), d(0, 1))
}

/// The edge map of the snakes unit, `|∇(G_σ * I)|²`, scaled to largest value 1.
pub fn edge_map(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let (gx, gy) = gradient(&gaussian_blur(img, sigma));
    let f = gx.zip_map(&gy, |a, b| a * a + b * b);
    let max = f.max();
    f.map(|v| if max > 0.0 { v / max } else { v })
}

/// The image value at a position between pixels, by bilinear interpolation; outside the
/// image the nearest edge pixel is used.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    let p = |dx: isize, dy: isize| img.get_border(x0 + dx, y0 + dy, Border::Clamp);
    (1.0 - fx) * (1.0 - fy) * p(0, 0) + fx * (1.0 - fy) * p(1, 0) + (1.0 - fx) * fy * p(0, 1) + fx * fy * p(1, 1)
}
