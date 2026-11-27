// Coding exercises for "Rotation and scale become shifts: Fourier–Mellin". Replace each
// `todo!()` with your code, then check it with `cargo test -p shapematch fourier_mellin::` and
// look at `cargo run -p shapematch -- fourier-mellin`.
#![allow(unused_variables, unused_imports)]

use imgcourse::fft::{fft, fft2};
use imgcourse::filters::bilinear;
use imgcourse::{Border, Complex, Image};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------------------
// Given (from the FFT registration day and the images unit)

/// The Hann taper of length `n`: `½ (1 − cos(2π i / (n − 1)))`, 0 at both ends (given).
pub fn hann(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos())).collect()
}

/// Phase correlation of two images of the same size (given, from the phase-correlation unit):
/// means subtracted, optional Hann taper, `A_k conj(B_k)` divided by its size.
/// `out[(tx, ty)]` is the score of the shift `t` with `B(x) ≈ A(x + t)`, wrapped around.
pub fn phase_correlation(a: &Image<f32>, b: &Image<f32>, taper: bool) -> Image<f32> {
    let (w, h) = (a.width(), a.height());
    let (wx, wy) = (hann(w), hann(h));
    let prepare = |img: &Image<f32>| {
        let m = img.mean();
        img.map_xy(|x, y, v| Complex::from((v - m) * if taper { wx[x] * wy[y] } else { 1.0 }))
    };
    let (fa, fb) = (fft2(&prepare(a), false), fft2(&prepare(b), false));
    let r = fa.zip_map(&fb, |p, q| {
        let c = p * q.conj();
        c / (c.abs() + 1e-6)
    });
    fft2(&r, true).map(|z| z.re)
}

/// The highest value of a correlation surface (given): `(tx, ty, height)`, the shift signed
/// (indices above half the size are negative shifts) and refined to a fraction of a pixel by a
/// parabola through the peak and its two neighbours in each direction.
pub fn peak(c: &Image<f32>) -> (f32, f32, f32) {
    let (w, h) = (c.width(), c.height());
    let mut best = (0, 0);
    for (x, y) in c.coords() {
        if c[(x, y)] > c[best] {
            best = (x, y);
        }
    }
    let (x, y) = best;
    let parabola = |m: f32, z: f32, p: f32| {
        let d = m - 2.0 * z + p;
        if d < 0.0 { ((m - p) / (2.0 * d)).clamp(-0.5, 0.5) } else { 0.0 }
    };
    let dx = parabola(c[((x + w - 1) % w, y)], c[(x, y)], c[((x + 1) % w, y)]);
    let dy = parabola(c[(x, (y + h - 1) % h)], c[(x, y)], c[(x, (y + 1) % h)]);
    let signed = |i: usize, n: usize| if i <= n / 2 { i as f32 } else { i as f32 - n as f32 };
    (signed(x, w) + dx, signed(y, h) + dy, c[(x, y)])
}

/// A `w × h` view of `src`, rotated by `angle_deg` (clockwise on screen) about the centre,
/// scaled by `scale` and shifted by `(tx, ty)` (given; inverse mapping as in the images unit):
/// `out(x) = src(s0 + R(−angle)(x − c − t) / scale)` with `c = (w/2, h/2)` and `s0 = (cx, cy)`
/// the matching point of the source. Outside the source: mirrored.
#[allow(clippy::too_many_arguments)]
pub fn warp_similarity(src: &Image<f32>, w: usize, h: usize, angle_deg: f32, scale: f32, tx: f32, ty: f32, cx: f32, cy: f32) -> Image<f32> {
    let (s, c) = angle_deg.to_radians().sin_cos();
    Image::from_fn(w, h, |x, y| {
        let dx = (x as f32 - w as f32 / 2.0 - tx) / scale;
        let dy = (y as f32 - h as f32 / 2.0 - ty) / scale;
        bilinear(src, cx + c * dx + s * dy, cy - s * dx + c * dy, Border::Mirror)
    })
}

// ---------------------------------------------------------------------------------------
// Exercises

/// C1. Unwrap `img` around `(cx, cy)` on a polar grid: `n_theta` columns over a full turn and
/// `n_r` rows. Column `j` is the angle `θ = 2π j / n_theta` (clockwise from the +x axis, because
/// y points down), row `i` the radius `r = r_max · i / (n_r − 1)`. The value is the image at
/// `(cx + r cos θ, cy + r sin θ)`, by bilinear interpolation (zero outside the image) (slide
/// "A rotation is a shift of the angle").
///
/// Hint: `Image::from_fn(n_theta, n_r, |j, i| ...)` (x = column = angle, y = row = radius) and
/// `bilinear(img, x, y, Border::Zero)`.
pub fn polar(img: &Image<f32>, cx: f32, cy: f32, r_max: f32, n_r: usize, n_theta: usize) -> Image<f32> {
    todo!("C1: polar")
}

/// C2. Rotational correlation of two polar images (same size; `n_theta` columns, a power of
/// two): `score[t] = Σ_rows weights[i] Σ_j A'(j, i) B'(j + t, i)`, wrapping around in `j`, where
/// `A'`, `B'` are the rows with their own mean subtracted. With one FFT per row:
/// `score = inverse FFT of Σ_i weights[i] · conj(A_i[m]) · B_i[m]`, real part. The peak is at
/// `t` = the rotation of B relative to A in columns (slide "Every angle at once").
///
/// Hint: `pa.row(i)` is row `i` as a slice; `fft(&row_as_complex, false)`; keep a running sum
/// `vec![Complex::ZERO; n]`, then `fft(&sum, true)` and take `.re`.
pub fn rotational_correlation(pa: &Image<f32>, pb: &Image<f32>, weights: &[f32]) -> Vec<f32> {
    todo!("C2: rotational_correlation")
}

/// C3. The magnitude spectrum `|F|` of an image (sides powers of two), with frequency 0 moved to
/// the centre `(w/2, h/2)`: subtract the mean, multiply by the 2D Hann taper if `taper`, FFT,
/// take sizes; if `highpass`, multiply by `(1 − X)(2 − X)` with `X = cos(π fx) cos(π fy)` and
/// `f` the frequency in cycles per pixel (slide "Practicalities").
///
/// Hint: the centred picture at `(x, y)` is the FFT at `((x + w/2) % w, (y + h/2) % h)`; there
/// `fx = (x − w/2) / w`, `fy = (y − h/2) / h`.
pub fn magnitude_spectrum(img: &Image<f32>, taper: bool, highpass: bool) -> Image<f32> {
    todo!("C3: magnitude_spectrum")
}

/// C4. Log-polar resampling of a centred spectrum over **half** a turn (`|F|` of a real image
/// is point-symmetric): column `j` is `θ = π j / n_theta`, row `i` the radius
/// `r = r_min · (r_max / r_min)^(i / (n_r − 1))`, around the centre `(w/2, h/2)`; bilinear
/// (slide "Unknown centre: the magnitude spectrum").
///
/// Hint: as `polar`, with another angle and radius per pixel; `powf`.
pub fn log_polar(mag: &Image<f32>, r_min: f32, r_max: f32, n_r: usize, n_theta: usize) -> Image<f32> {
    todo!("C4: log_polar")
}

/// Settings of the Fourier–Mellin registration.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Hann taper before each FFT.
    pub taper: bool,
    /// High-pass weight on `|F|` (see `magnitude_spectrum`).
    pub highpass: bool,
    /// Test both `θ` and `θ + 180°` by the translation peak (otherwise only `θ`).
    pub both: bool,
    /// Log-polar grid: columns over 180°, rows, smallest and largest radius.
    pub n_theta: usize,
    pub n_r: usize,
    pub r_min: f32,
    pub r_max: f32,
}

/// What Fourier–Mellin found: B ≈ A rotated by `angle` degrees, scaled by `scale` and shifted by
/// `(tx, ty)` (as in `warp_similarity`); `rot_peak` is the height of the phase-correlation peak
/// in the log-polar grid, `shift_peak` that of the final translation step.
#[derive(Clone, Copy, Debug)]
pub struct Estimate {
    pub angle: f32,
    pub scale: f32,
    pub tx: f32,
    pub ty: f32,
    pub rot_peak: f32,
    pub shift_peak: f32,
}

/// C5. Fourier–Mellin registration of two images of the same size (slide "Fourier–Mellin as
/// pseudocode"):
/// 1. `|F|` of both (`magnitude_spectrum`), log-polar (`log_polar`);
/// 2. phase correlation of the log-polar images, **B first**: `phase_correlation(&lp_b, &lp_a, false)`;
///    the peak `(u, v)` gives `angle = 180° · u / n_theta` and `scale = exp(−v · step)` with
///    `step = ln(r_max / r_min) / (n_r − 1)`;
/// 3. for `angle` and (if `both`) `angle + 180°`: undo the rotation and scale on B,
///    `b2 = warp_similarity(b, w, h, −angle, 1 / scale, 0, 0, w/2, h/2)`, phase-correlate A with `b2`
///    (taper as set); its peak `(px, py)` gives the shift `t = −scale · R(angle) (px, py)`, i.e.
///    `tx = −scale (cos·px − sin·py)`, `ty = −scale (sin·px + cos·py)`;
/// 4. keep the candidate with the higher translation peak.
///
/// Hint: report angles above 180° as `angle − 360°`; `angle.to_radians().sin_cos()`.
pub fn fourier_mellin(a: &Image<f32>, b: &Image<f32>, s: Settings) -> Estimate {
    todo!("C5: fourier_mellin")
}
