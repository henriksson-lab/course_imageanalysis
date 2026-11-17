// Coding exercises for "Cross-correlation with the FFT". Replace each `todo!()` with your code,
// then check it with `cargo test -p xcorr-fft` and look at `cargo run -p xcorr-fft`.
#![allow(unused_variables, unused_imports)]

use imgcourse::fft::{fft, fft2};
use imgcourse::{Complex, Image};

/// C1. The circular cross-correlation, directly: `c[t] = sum over x of a[(x + t) mod N] · b[x]`
/// for every shift `t = 0 .. N−1` (both signals of length `N`).
///
/// Hint: two loops; `(x + t) % n` wraps around.
pub fn xcorr_direct(a: &[f32], b: &[f32]) -> Vec<f32> {
    todo!("C1: xcorr_direct")
}

/// C2. The same with the FFT: transform both, multiply `A_k · conj(B_k)` at every frequency,
/// transform back and keep the real part (slide "All shifts at once").
///
/// Hint: turn the samples into complex numbers with `Complex::from(v)`; `fft(&x, false)` is the
/// forward transform, `fft(&x, true)` the inverse (it already divides by N); `q.conj()` and `z.re`.
pub fn xcorr_fft(a: &[f32], b: &[f32]) -> Vec<f32> {
    todo!("C2: xcorr_fft")
}

/// C3. The linear cross-correlation (what sliding computes, no wrap-around) with the FFT: put
/// `a` (length `n`) and `b` (length `m`) into arrays of zeros of a power-of-two length
/// `L >= n + m − 1`, correlate circularly, and read out the shifts `t = −(m−1) .. n−1`.
/// `out[t + m − 1]` is the correlation at shift `t` (slide "Circular or linear?").
///
/// Hint: `(n + m - 1).next_power_of_two()`; reuse `xcorr_fft` on the padded arrays. Shift `t`
/// of the circular result is index `t` for `t >= 0` and `len + t` for `t < 0`
/// (`t.rem_euclid(len as isize)`).
pub fn xcorr_linear(a: &[f32], b: &[f32]) -> Vec<f32> {
    todo!("C3: xcorr_linear")
}

/// C4. The circular cross-correlation of two images of the same size, after subtracting each
/// image's mean, with the 2D FFT. `out[(tx, ty)]` is the score of the shift `(tx, ty)`, with
/// indices above half the size meaning negative shifts.
///
/// Hint: as `xcorr_fft`, with `fft2` on `Image<Complex>`: `a.map(|v| Complex::from(v - ma))`,
/// and `zip_map` for the product.
pub fn xcorr2(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    todo!("C4: xcorr2")
}

/// C4. The highest value of a circular correlation surface, as a signed shift: index `i` along
/// an axis of length `n` means `i` if `i <= n/2`, else `i − n`. Returns `(tx, ty, value)`.
///
/// Hint: find the largest pixel, then turn each index into a signed shift.
pub fn peak(c: &Image<f32>) -> (isize, isize, f32) {
    todo!("C4: peak")
}
