// Coding exercises for "Better bases: Legendre moments and reconstruction".
// Replace each `todo!()` with your code, then check it with `cargo test -p orthogonal-moments`
// and look at `cargo run -p orthogonal-moments`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// A table of moments: `a[p][q]`, filled for `p + q <= order`.
pub type Moments = Vec<Vec<f64>>;

/// The centre of pixel `i` of `n`, mapped to [-1, 1]: `(2i + 1) / n - 1`. Given.
pub fn centre(i: usize, n: usize) -> f64 {
    (2 * i + 1) as f64 / n as f64 - 1.0
}

/// C1. The Legendre polynomial `P_n(x)` by the three-term recurrence (slide "The math:
/// Legendre polynomials"): `P_0 = 1`, `P_1 = x`, `(k + 1) P_(k+1) = (2k + 1) x P_k - k P_(k-1)`.
///
/// Hint: keep the last two values (`prev` = P_(k-1), `p` = P_k) and step k from 1 to n - 1.
pub fn legendre(n: usize, x: f64) -> f64 {
    todo!("C1: legendre")
}

/// C2. Gram–Schmidt on the sampled monomials `1, x, x², ..., x^max_degree` at the points `xs`:
/// returns vectors `t[n]` (one value per point) that are orthonormal on the points,
/// `sum over i of t[m][i] t[n][i] = 1 if m = n, else 0`. These are the discrete orthogonal
/// polynomials of the slide "Pixels: sampled polynomials are not orthogonal".
///
/// Hint: for each degree, start from the sampled monomial, subtract `(v · b) b` for every
/// vector `b` made so far, then divide by the length. Doing the subtraction twice makes the
/// result accurate also for high degrees.
pub fn gram_schmidt(max_degree: usize, xs: &[f64]) -> Vec<Vec<f64>> {
    todo!("C2: gram_schmidt")
}

/// C3. Legendre moments with the image on [-1, 1]² and the polynomials sampled at the pixel
/// centres (slide "Legendre moments"):
/// `a[p][q] = (2p + 1)(2q + 1) / 4 · sum of P_p(x) P_q(y) I(x, y) Δx Δy`, `Δx = 2 / width`,
/// `Δy = 2 / height`.
///
/// Hint: `centre(x, width)` gives the coordinate of a column in [-1, 1]. Computing all `P_p` at
/// all columns (and rows) once, before the loops, saves a lot of time.
pub fn legendre_moments(img: &Image<f32>, order: usize) -> Moments {
    todo!("C3: legendre_moments")
}

/// C4. The reconstruction from the Legendre moments (slide "Reconstruction is a projection"):
/// `I(x, y) = sum over p + q <= N of a[p][q] P_p(x) P_q(y)`, on a `width × height` grid.
///
/// Hint: `Image::from_fn(width, height, |x, y| ...)` with the double sum inside; `order =
/// a.len() - 1`.
pub fn reconstruct(a: &Moments, width: usize, height: usize) -> Image<f32> {
    todo!("C4: reconstruct")
}
