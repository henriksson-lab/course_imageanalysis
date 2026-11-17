// Coding exercises for "Zernike moments on the unit disc".
// Replace each `todo!()` with your code, then check it with `cargo test -p moments zernike::`
// and look at `cargo run -p moments -- zernike`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Complex, Image};
use std::f64::consts::PI;

/// `k!` as a floating-point number. Given.
pub fn factorial(k: usize) -> f64 {
    (1..=k).fold(1.0, |acc, i| acc * i as f64)
}

/// The polar coordinates `(rho, theta)` of the centre of pixel `(x, y)` in the disc inscribed
/// in a `size × size` image: centre of the image at the origin, radius `size / 2` = 1, `y` up,
/// `theta` counter-clockwise from the `x` axis. Given.
pub fn polar(x: usize, y: usize, size: usize) -> (f64, f64) {
    let half = size as f64 / 2.0;
    let u = (x as f64 + 0.5 - half) / half;
    let v = (half - (y as f64 + 0.5)) / half;
    (u.hypot(v), v.atan2(u))
}

/// C1. The radial polynomial `R_nm(rho)` by the factorial formula (slide "The math: Zernike
/// polynomials are orthogonal"), for `n - |m|` even:
/// `R_nm = sum over k = 0 .. (n - |m|)/2 of (-1)^k (n - k)! / (k! ((n + |m|)/2 - k)! ((n - |m|)/2 - k)!) rho^(n - 2k)`.
///
/// Hint: `factorial` is given. Use `m.unsigned_abs() as usize` for |m|; `(n - m) / 2` is a
/// whole number because n - |m| is even.
pub fn radial(n: usize, m: i32, rho: f64) -> f64 {
    todo!("C1: radial")
}

/// C2. The Zernike moment of a square image on its inscribed disc (slide "Zernike moments"):
/// `Z_nm = (n + 1) / pi · sum over pixels inside the disc of I · R_nm(rho) · e^(-i m theta) · ΔA`,
/// with `ΔA = (2 / size)²`, the area of one pixel when the disc has radius 1.
///
/// Hint: `polar(x, y, size)` gives ρ and θ of a pixel centre. e^(-imθ) = cos(mθ) - i sin(mθ):
/// add up the real and imaginary parts in two `f64` sums, then make a `Complex`.
pub fn zernike_moment(img: &Image<f32>, n: usize, m: i32) -> Complex {
    todo!("C2: zernike_moment")
}

/// C3. The reconstruction from all moments up to order `order` (slide "Zernike moments"):
/// `I = sum over n, m of Z_nm V_nm`. For a real image `Z_n,-m` is the conjugate of `Z_nm`, so
/// the terms with `m > 0` come in pairs: `Z_n0 R_n0 + sum over m > 0 of 2 Re(Z_nm R_nm e^(i m theta))`.
/// Pixels outside the disc are 0.
///
/// Hint: first collect all `(n, m, zernike_moment(img, n, m))` with `m = n % 2, n % 2 + 2, ...,
/// n`, then build the image with `Image::from_fn`, using `polar` again for every pixel.
pub fn reconstruct(img: &Image<f32>, order: usize) -> Image<f32> {
    todo!("C3: reconstruct")
}

/// C4. `R_n0(rho)` without the large alternating terms of the factorial formula, using
/// `R_n0(rho) = P_(n/2)(2 rho² - 1)` (n even) and the Legendre recurrence of the previous unit:
/// `P_0 = 1`, `P_1 = t`, `(k + 1) P_(k+1) = (2k + 1) t P_k - k P_(k-1)`.
///
/// Hint: the same loop as `legendre` in the previous unit, with t = 2ρ² - 1 and n/2 in place of
/// n.
pub fn radial0_recurrence(n: usize, rho: f64) -> f64 {
    todo!("C4: radial0_recurrence")
}
