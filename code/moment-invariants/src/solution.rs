// Model solution for the coding exercises of "Normalisation and moment invariants".

use imgcourse::{Complex, Image};

/// A table of moments: `m[p][q]` is the moment of order `p + q`, filled for `p + q <= order`.
pub type Moments = Vec<Vec<f64>>;

/// The raw moments `m[p][q] = sum of x^p y^q I(x, y)` (pixel units, `y` down), as in the
/// previous unit. Given, not an exercise.
pub fn raw_moments(img: &Image<f32>, order: usize) -> Moments {
    let mut m = vec![vec![0.0; order + 1]; order + 1];
    for (x, y) in img.coords() {
        let v = img[(x, y)] as f64;
        for p in 0..=order {
            for q in 0..=order - p {
                m[p][q] += (x as f64).powi(p as i32) * (y as f64).powi(q as i32) * v;
            }
        }
    }
    m
}

/// The binomial coefficient "n choose k". Given.
pub fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// C1. Central moments from raw moments, without a second pass over the image (slide "The
/// math: central from raw moments"):
/// `mu[p][q] = sum over k <= p, l <= q of C(p, k) C(q, l) (-xbar)^(p-k) (-ybar)^(q-l) m[k][l]`.
pub fn central_from_raw(m: &Moments) -> Moments {
    let order = m.len() - 1;
    let (xbar, ybar) = (m[1][0] / m[0][0], m[0][1] / m[0][0]);
    let mut mu = vec![vec![0.0; order + 1]; order + 1];
    for p in 0..=order {
        for q in 0..=order - p {
            let mut sum = 0.0;
            for k in 0..=p {
                for l in 0..=q {
                    sum += binomial(p, k) * binomial(q, l) * (-xbar).powi((p - k) as i32) * (-ybar).powi((q - l) as i32) * m[k][l];
                }
            }
            mu[p][q] = sum;
        }
    }
    mu
}

/// C2. Normalised moments `eta[p][q] = mu[p][q] / mu00^(1 + (p + q) / 2)`: unchanged when the
/// object is moved or scaled (slide "Scale: normalised moments").
pub fn normalised(mu: &Moments) -> Moments {
    let order = mu.len() - 1;
    let mut eta = vec![vec![0.0; order + 1]; order + 1];
    for p in 0..=order {
        for q in 0..=order - p {
            eta[p][q] = mu[p][q] / mu[0][0].powf(1.0 + (p + q) as f64 / 2.0);
        }
    }
    eta
}

/// C3. The normalised complex moment `c_pq = sum of (x + iy)^p (x - iy)^q I / mu00^(1 + (p+q)/2)`,
/// with `x, y` measured from the centroid and `y` pointing **up** (so `y = -(row - ybar)`),
/// as on the slide "Complex moments: rotation becomes a phase".
pub fn complex_moment(img: &Image<f32>, p: usize, q: usize) -> Complex {
    let m = raw_moments(img, 1);
    let (xbar, ybar) = (m[1][0] / m[0][0], m[0][1] / m[0][0]);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (x, y) in img.coords() {
        let z = Complex::new((x as f64 - xbar) as f32, -(y as f64 - ybar) as f32);
        let mut w = Complex::from(img[(x, y)]);
        for _ in 0..p {
            w = w * z;
        }
        for _ in 0..q {
            w = w * z.conj();
        }
        re += w.re as f64;
        im += w.im as f64;
    }
    let scale = m[0][0].powf(1.0 + (p + q) as f64 / 2.0);
    Complex::new((re / scale) as f32, (im / scale) as f32)
}

/// C4. Hu's seven invariants from the complex moments (table on the slide "The math: complex
/// moments and Hu's invariants"): `phi1 = c11`, `phi2 = |c20|²`, `phi3 = |c30|²`,
/// `phi4 = |c21|²`, `phi5 = Re(c30 c12³)`, `phi6 = Re(c20 c12²)`, `phi7 = Im(c30 c12³)`.
pub fn hu(img: &Image<f32>) -> [f64; 7] {
    let c = |p, q| complex_moment(img, p, q);
    let (c11, c20, c30, c21, c12) = (c(1, 1), c(2, 0), c(3, 0), c(2, 1), c(1, 2));
    let a = c30 * c12 * c12 * c12;
    let b = c20 * c12 * c12;
    [c11.re, c20.norm_sqr(), c30.norm_sqr(), c21.norm_sqr(), a.re, b.re, a.im].map(|v| v as f64)
}
