// Model solution for the coding exercises of "Phase correlation".

use imgcourse::fft::{fft, fft2, frequency};
use imgcourse::{Complex, Image};

/// A small number added to the sizes before dividing, so that empty frequencies do not get
/// a full vote (and nothing is divided by zero).
pub const EPS: f32 = 1e-6;

/// C1. Phase correlation of two signals of the same length: transform both, take the
/// cross-power `A_k · conj(B_k)`, divide it by its size (plus `EPS`), transform back and keep
/// the real part (slide "Keep only the phase").
pub fn phase_correlation_1d(a: &[f32], b: &[f32]) -> Vec<f32> {
    let fa = fft(&a.iter().map(|&v| Complex::from(v)).collect::<Vec<_>>(), false);
    let fb = fft(&b.iter().map(|&v| Complex::from(v)).collect::<Vec<_>>(), false);
    let r: Vec<Complex> = fa
        .iter()
        .zip(&fb)
        .map(|(&p, &q)| {
            let c = p * q.conj();
            c / (c.abs() + EPS)
        })
        .collect();
    fft(&r, true).iter().map(|z| z.re).collect()
}

/// C2. The Hann taper of length `n`: `w(i) = ½ (1 − cos(2π i / (n − 1)))`, 0 at both ends and
/// 1 in the middle.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos())).collect()
}

/// Settings of the 2D phase correlation.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Divide the cross-power by its size to the power `p`: 0 is cross-correlation, 1 phase
    /// correlation.
    pub p: f32,
    /// Multiply both images by the 2D Hann taper (after subtracting their means).
    pub taper: bool,
    /// If not 0: weight the normalised cross-power by `exp(−f² / (2 sd²))`, where `f` is the
    /// frequency in cycles per pixel (a low-pass weight).
    pub lowpass_sd: f32,
}

/// C3. Phase correlation of two images of the same size (slides "What it costs: noise" and
/// "Tapering"): subtract each image's mean, taper (optional), FFT both, divide `A_k conj(B_k)`
/// by `|A_k conj(B_k)|^p + EPS`, apply the low-pass weight (optional), inverse FFT, real part.
/// `out[(tx, ty)]` is the score of the shift `(tx, ty)`, wrapped around as in the last unit.
pub fn phase_correlation(a: &Image<f32>, b: &Image<f32>, s: Settings) -> Image<f32> {
    let (w, h) = (a.width(), a.height());
    let (wx, wy) = (hann(w), hann(h));
    let prepare = |img: &Image<f32>| {
        let m = img.mean();
        img.map_xy(|x, y, v| {
            let taper = if s.taper { wx[x] * wy[y] } else { 1.0 };
            Complex::from((v - m) * taper)
        })
    };
    let fa = fft2(&prepare(a), false);
    let fb = fft2(&prepare(b), false);
    let r = fa.zip_map(&fb, |p, q| p * q.conj()).map_xy(|kx, ky, c| {
        let mut v = c / (c.abs().powf(s.p) + EPS);
        if s.lowpass_sd > 0.0 {
            let (fx, fy) = (frequency(kx, w) / w as f32, frequency(ky, h) / h as f32);
            v = v * (-(fx * fx + fy * fy) / (2.0 * s.lowpass_sd * s.lowpass_sd)).exp();
        }
        v
    });
    fft2(&r, true).map(|z| z.re)
}

/// C4. The highest peak of a correlation surface and how much to trust it: returns
/// `(tx, ty, height, ratio)`, the shift (signed, as in the last unit), the peak's value, and
/// its ratio to the second-highest **local maximum** (a pixel larger than its 8 neighbours,
/// wrapping around at the edges) (slide "How much to trust a peak").
pub fn peak_and_ratio(c: &Image<f32>) -> (isize, isize, f32, f32) {
    let (w, h) = (c.width(), c.height());
    let mut maxima: Vec<(f32, usize, usize)> = Vec::new();
    for (x, y) in c.coords() {
        let v = c[(x, y)];
        let mut is_max = true;
        for dy in [h - 1, 0, 1] {
            for dx in [w - 1, 0, 1] {
                if (dx, dy) != (0, 0) && c[((x + dx) % w, (y + dy) % h)] >= v {
                    is_max = false;
                }
            }
        }
        if is_max {
            maxima.push((v, x, y));
        }
    }
    maxima.sort_by(|p, q| q.0.total_cmp(&p.0));
    let (v, x, y) = maxima[0];
    let ratio = if maxima.len() > 1 && maxima[1].0 > 0.0 { v / maxima[1].0 } else { f32::INFINITY };
    let signed = |i: usize, n: usize| if i <= n / 2 { i as isize } else { i as isize - n as isize };
    (signed(x, w), signed(y, h), v, ratio)
}
