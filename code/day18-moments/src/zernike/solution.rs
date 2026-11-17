// Model solution for the coding exercises of "Zernike moments on the unit disc".

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
pub fn radial(n: usize, m: i32, rho: f64) -> f64 {
    let m = m.unsigned_abs() as usize;
    let mut sum = 0.0;
    for k in 0..=(n - m) / 2 {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let c = sign * factorial(n - k) / (factorial(k) * factorial((n + m) / 2 - k) * factorial((n - m) / 2 - k));
        sum += c * rho.powi((n - 2 * k) as i32);
    }
    sum
}

/// C2. The Zernike moment of a square image on its inscribed disc (slide "Zernike moments"):
/// `Z_nm = (n + 1) / pi · sum over pixels inside the disc of I · R_nm(rho) · e^(-i m theta) · ΔA`,
/// with `ΔA = (2 / size)²`, the area of one pixel when the disc has radius 1.
pub fn zernike_moment(img: &Image<f32>, n: usize, m: i32) -> Complex {
    let size = img.width();
    let area = (2.0 / size as f64).powi(2);
    let (mut re, mut im) = (0.0, 0.0);
    for (x, y) in img.coords() {
        let (rho, theta) = polar(x, y, size);
        if rho <= 1.0 {
            let w = img[(x, y)] as f64 * radial(n, m, rho) * area;
            re += w * (m as f64 * theta).cos();
            im -= w * (m as f64 * theta).sin();
        }
    }
    let f = (n + 1) as f64 / PI;
    Complex::new((f * re) as f32, (f * im) as f32)
}

/// C3. The reconstruction from all moments up to order `order` (slide "Zernike moments"):
/// `I = sum over n, m of Z_nm V_nm`. For a real image `Z_n,-m` is the conjugate of `Z_nm`, so
/// the terms with `m > 0` come in pairs: `Z_n0 R_n0 + sum over m > 0 of 2 Re(Z_nm R_nm e^(i m theta))`.
/// Pixels outside the disc are 0.
pub fn reconstruct(img: &Image<f32>, order: usize) -> Image<f32> {
    let size = img.width();
    // all (n, m, Z_nm) with m >= 0 and n - m even
    let mut moments = Vec::new();
    for n in 0..=order {
        for m in (n % 2..=n).step_by(2) {
            moments.push((n, m as i32, zernike_moment(img, n, m as i32)));
        }
    }
    Image::from_fn(size, size, |x, y| {
        let (rho, theta) = polar(x, y, size);
        if rho > 1.0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for &(n, m, z) in &moments {
            let r = radial(n, m, rho);
            let (c, s) = ((m as f64 * theta).cos(), (m as f64 * theta).sin());
            // Re(Z e^(i m theta)) = Z.re cos - Z.im sin
            let term = r * (z.re as f64 * c - z.im as f64 * s);
            sum += if m == 0 { term } else { 2.0 * term };
        }
        sum as f32
    })
}

/// C4. `R_n0(rho)` without the large alternating terms of the factorial formula, using
/// `R_n0(rho) = P_(n/2)(2 rho² - 1)` (n even) and the Legendre recurrence of the previous unit:
/// `P_0 = 1`, `P_1 = t`, `(k + 1) P_(k+1) = (2k + 1) t P_k - k P_(k-1)`.
pub fn radial0_recurrence(n: usize, rho: f64) -> f64 {
    let t = 2.0 * rho * rho - 1.0;
    let (mut prev, mut p) = (1.0, t);
    if n == 0 {
        return 1.0;
    }
    for k in 1..n / 2 {
        let next = ((2 * k + 1) as f64 * t * p - k as f64 * prev) / (k + 1) as f64;
        prev = p;
        p = next;
    }
    p
}
