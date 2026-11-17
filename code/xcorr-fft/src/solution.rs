// Model solution for the coding exercises of "Cross-correlation with the FFT".

use imgcourse::fft::{fft, fft2};
use imgcourse::{Complex, Image};

/// C1. The circular cross-correlation, directly: `c[t] = sum over x of a[(x + t) mod N] · b[x]`
/// for every shift `t = 0 .. N−1` (both signals of length `N`).
pub fn xcorr_direct(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut c = vec![0.0; n];
    for t in 0..n {
        for x in 0..n {
            c[t] += a[(x + t) % n] * b[x];
        }
    }
    c
}

/// C2. The same with the FFT: transform both, multiply `A_k · conj(B_k)` at every frequency,
/// transform back and keep the real part (slide "All shifts at once").
pub fn xcorr_fft(a: &[f32], b: &[f32]) -> Vec<f32> {
    let fa = fft(&a.iter().map(|&v| Complex::from(v)).collect::<Vec<_>>(), false);
    let fb = fft(&b.iter().map(|&v| Complex::from(v)).collect::<Vec<_>>(), false);
    let product: Vec<Complex> = fa.iter().zip(&fb).map(|(&p, &q)| p * q.conj()).collect();
    fft(&product, true).iter().map(|z| z.re).collect()
}

/// C3. The linear cross-correlation (what sliding computes, no wrap-around) with the FFT: put
/// `a` (length `n`) and `b` (length `m`) into arrays of zeros of a power-of-two length
/// `L >= n + m − 1`, correlate circularly, and read out the shifts `t = −(m−1) .. n−1`.
/// `out[t + m − 1]` is the correlation at shift `t` (slide "Circular or linear?").
pub fn xcorr_linear(a: &[f32], b: &[f32]) -> Vec<f32> {
    let (n, m) = (a.len(), b.len());
    let len = (n + m - 1).next_power_of_two();
    let mut pa = vec![0.0; len];
    let mut pb = vec![0.0; len];
    pa[..n].copy_from_slice(a);
    pb[..m].copy_from_slice(b);
    let c = xcorr_fft(&pa, &pb);
    // negative shifts t sit at the end of the circular result, at index len + t
    (-(m as isize - 1)..n as isize).map(|t| c[t.rem_euclid(len as isize) as usize]).collect()
}

/// C4. The circular cross-correlation of two images of the same size, after subtracting each
/// image's mean, with the 2D FFT. `out[(tx, ty)]` is the score of the shift `(tx, ty)`, with
/// indices above half the size meaning negative shifts.
pub fn xcorr2(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    let (ma, mb) = (a.mean(), b.mean());
    let fa = fft2(&a.map(|v| Complex::from(v - ma)), false);
    let fb = fft2(&b.map(|v| Complex::from(v - mb)), false);
    let product = fa.zip_map(&fb, |p, q| p * q.conj());
    fft2(&product, true).map(|z| z.re)
}

/// C4. The highest value of a circular correlation surface, as a signed shift: index `i` along
/// an axis of length `n` means `i` if `i <= n/2`, else `i − n`. Returns `(tx, ty, value)`.
pub fn peak(c: &Image<f32>) -> (isize, isize, f32) {
    let (mut bx, mut by) = (0, 0);
    for (x, y) in c.coords() {
        if c[(x, y)] > c[(bx, by)] {
            (bx, by) = (x, y);
        }
    }
    let signed = |i: usize, n: usize| if i <= n / 2 { i as isize } else { i as isize - n as isize };
    (signed(bx, c.width()), signed(by, c.height()), c[(bx, by)])
}
