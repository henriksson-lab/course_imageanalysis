// Model solution for the coding exercises of "Better bases: Legendre moments and reconstruction".

use imgcourse::Image;

/// A table of moments: `a[p][q]`, filled for `p + q <= order`.
pub type Moments = Vec<Vec<f64>>;

/// The centre of pixel `i` of `n`, mapped to [-1, 1]: `(2i + 1) / n - 1`. Given.
pub fn centre(i: usize, n: usize) -> f64 {
    (2 * i + 1) as f64 / n as f64 - 1.0
}

/// C1. The Legendre polynomial `P_n(x)` by the three-term recurrence (slide "The math:
/// Legendre polynomials"): `P_0 = 1`, `P_1 = x`, `(k + 1) P_(k+1) = (2k + 1) x P_k - k P_(k-1)`.
pub fn legendre(n: usize, x: f64) -> f64 {
    let (mut prev, mut p) = (1.0, x); // P_0 and P_1
    if n == 0 {
        return prev;
    }
    for k in 1..n {
        let next = ((2 * k + 1) as f64 * x * p - k as f64 * prev) / (k + 1) as f64;
        prev = p;
        p = next;
    }
    p
}

/// C2. Gram–Schmidt on the sampled monomials `1, x, x², ..., x^max_degree` at the points `xs`:
/// returns vectors `t[n]` (one value per point) that are orthonormal on the points,
/// `sum over i of t[m][i] t[n][i] = 1 if m = n, else 0`. These are the discrete orthogonal
/// polynomials of the slide "Pixels: sampled polynomials are not orthogonal".
pub fn gram_schmidt(max_degree: usize, xs: &[f64]) -> Vec<Vec<f64>> {
    let mut t: Vec<Vec<f64>> = Vec::new();
    for n in 0..=max_degree {
        let mut v: Vec<f64> = xs.iter().map(|x| x.powi(n as i32)).collect();
        // subtract the projections onto the vectors already made (twice, for accuracy)
        for _ in 0..2 {
            for b in &t {
                let dot: f64 = v.iter().zip(b).map(|(a, b)| a * b).sum();
                for i in 0..v.len() {
                    v[i] -= dot * b[i];
                }
            }
        }
        let len = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        t.push(v.iter().map(|a| a / len).collect());
    }
    t
}

/// C3. Legendre moments with the image on [-1, 1]² and the polynomials sampled at the pixel
/// centres (slide "Legendre moments"):
/// `a[p][q] = (2p + 1)(2q + 1) / 4 · sum of P_p(x) P_q(y) I(x, y) Δx Δy`, `Δx = 2 / width`,
/// `Δy = 2 / height`.
pub fn legendre_moments(img: &Image<f32>, order: usize) -> Moments {
    let (w, h) = (img.width(), img.height());
    // the polynomials at all column and row centres, computed once
    let px: Vec<Vec<f64>> = (0..=order).map(|p| (0..w).map(|x| legendre(p, centre(x, w))).collect()).collect();
    let py: Vec<Vec<f64>> = (0..=order).map(|q| (0..h).map(|y| legendre(q, centre(y, h))).collect()).collect();
    let area = (2.0 / w as f64) * (2.0 / h as f64);
    let mut a = vec![vec![0.0; order + 1]; order + 1];
    for p in 0..=order {
        for q in 0..=order - p {
            let mut sum = 0.0;
            for (x, y) in img.coords() {
                sum += px[p][x] * py[q][y] * img[(x, y)] as f64;
            }
            a[p][q] = (2 * p + 1) as f64 * (2 * q + 1) as f64 / 4.0 * sum * area;
        }
    }
    a
}

/// C4. The reconstruction from the Legendre moments (slide "Reconstruction is a projection"):
/// `I(x, y) = sum over p + q <= N of a[p][q] P_p(x) P_q(y)`, on a `width × height` grid.
pub fn reconstruct(a: &Moments, width: usize, height: usize) -> Image<f32> {
    let order = a.len() - 1;
    let px: Vec<Vec<f64>> = (0..=order).map(|p| (0..width).map(|x| legendre(p, centre(x, width))).collect()).collect();
    let py: Vec<Vec<f64>> = (0..=order).map(|q| (0..height).map(|y| legendre(q, centre(y, height))).collect()).collect();
    Image::from_fn(width, height, |x, y| {
        let mut sum = 0.0;
        for p in 0..=order {
            for q in 0..=order - p {
                sum += a[p][q] * px[p][x] * py[q][y];
            }
        }
        sum as f32
    })
}
