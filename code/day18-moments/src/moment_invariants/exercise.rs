// Coding exercises for "Normalisation and moment invariants".
// Replace each `todo!()` with your code, then check it with `cargo test -p day18 moment_invariants::`
// and look at `cargo run -p day18 -- moment-invariants`.
#![allow(unused_variables, unused_imports)]

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
///
/// Hint: four nested loops, over p, q (the result) and k, l (the raw moments used).
/// `binomial(p, k)` is given; `(-xbar).powi((p - k) as i32)`.
pub fn central_from_raw(m: &Moments) -> Moments {
    todo!("C1: central_from_raw")
}

/// C2. Normalised moments `eta[p][q] = mu[p][q] / mu00^(1 + (p + q) / 2)`: unchanged when the
/// object is moved or scaled (slide "Scale: normalised moments").
///
/// Hint: `mu[0][0].powf(1.0 + (p + q) as f64 / 2.0)`.
pub fn normalised(mu: &Moments) -> Moments {
    todo!("C2: normalised")
}

/// C3. The normalised complex moment `c_pq = sum of (x + iy)^p (x - iy)^q I / mu00^(1 + (p+q)/2)`,
/// with `x, y` measured from the centroid and `y` pointing **up** (so `y = -(row - ybar)`),
/// as on the slide "Complex moments: rotation becomes a phase".
///
/// Hint: for every pixel make `z = Complex::new(dx, -dy)` (minus: y up) from the distances to
/// the centroid, multiply `I(x, y)` by `z` p times and by `z.conj()` q times, and add up. Add
/// in `f64` (`re` and `im` separately) for accuracy, then divide.
pub fn complex_moment(img: &Image<f32>, p: usize, q: usize) -> Complex {
    todo!("C3: complex_moment")
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
