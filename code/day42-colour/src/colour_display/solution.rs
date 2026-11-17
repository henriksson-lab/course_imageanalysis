// Model solution for the coding exercises of "Colour for display: colour maps, merges and
// decorrelation".

use imgcourse::{linalg, Image};

/// A colour image: `[red, green, blue]` per pixel, encoded sRGB values in [0, 1].
pub type Rgb = Image<[f32; 3]>;

/// Viridis, 9 equally spaced anchor colours (encoded sRGB), from t = 0 to t = 1.
pub const VIRIDIS: [[f32; 3]; 9] = [
    [0.267, 0.004, 0.329], [0.278, 0.176, 0.482], [0.231, 0.322, 0.545], [0.173, 0.447, 0.557], [0.129, 0.565, 0.549],
    [0.153, 0.678, 0.506], [0.365, 0.784, 0.388], [0.667, 0.863, 0.196], [0.992, 0.906, 0.145],
];
/// Jet, the classic rainbow map, 9 equally spaced anchors.
pub const JET: [[f32; 3]; 9] = [
    [0.0, 0.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.5, 1.0], [0.0, 1.0, 1.0], [0.5, 1.0, 0.5],
    [1.0, 1.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0],
];
/// A diverging blue–white–red map.
pub const DIVERGING: [[f32; 3]; 3] = [[0.23, 0.30, 0.75], [1.0, 1.0, 1.0], [0.71, 0.02, 0.15]];

/// sRGB decoding and encoding (from the unit on RGB), in `src/common.rs`.
pub use crate::common::{srgb_decode, srgb_encode};

/// C1a. The position t in [0, 1] of the data value `v` in the display range `lo..hi`:
/// 0 at `lo`, 1 at `hi`, clipped outside.
pub fn position(v: f32, lo: f32, hi: f32) -> f32 {
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// C1b. A colour map: every pixel becomes the colour at its position t, interpolated linearly
/// between the two nearest of the equally spaced `anchors` (anchor 0 at t = 0, the last at t = 1).
pub fn colour_map(img: &Image<f32>, lo: f32, hi: f32, anchors: &[[f32; 3]]) -> Rgb {
    let n = anchors.len();
    img.map(|v| {
        let s = position(v, lo, hi) * (n - 1) as f32;
        let i = (s.floor() as usize).min(n - 2);
        let f = s - i as f32;
        let mut c = [0.0; 3];
        for k in 0..3 {
            c[k] = (1.0 - f) * anchors[i][k] + f * anchors[i + 1][k];
        }
        c
    })
}

/// C2a. The lightness L* (0 to 100) of an encoded sRGB colour: decode, the luminance
/// `Y = 0.2126 R + 0.7152 G + 0.0722 B`, then `L* = 116 f(Y) − 16` (the white has Y = 1).
pub fn lightness(c: [f32; 3]) -> f32 {
    let lin = c.map(srgb_decode);
    let y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
    let d = 6.0f32 / 29.0;
    let f = if y > d * d * d { y.cbrt() } else { y / (3.0 * d * d) + 4.0 / 29.0 };
    116.0 * f - 16.0
}

/// C2b. Is the sequence strictly increasing or strictly decreasing (a monotone lightness
/// profile)?
pub fn is_monotone(values: &[f32]) -> bool {
    let up = values.windows(2).all(|w| w[1] > w[0]);
    let down = values.windows(2).all(|w| w[1] < w[0]);
    up || down
}

/// C3. Merge grey channels into one colour image: channel k (values in [0, 1]) times its
/// colour `colours[k]`, added up, clipped to [0, 1].
pub fn merge(channels: &[Image<f32>], colours: &[[f32; 3]]) -> Rgb {
    let (w, h) = (channels[0].width(), channels[0].height());
    Image::from_fn(w, h, |x, y| {
        let mut c = [0.0f32; 3];
        for (ch, col) in channels.iter().zip(colours) {
            for k in 0..3 {
                c[k] += ch[(x, y)] * col[k];
            }
        }
        c.map(|v| v.clamp(0.0, 1.0))
    })
}

/// C4. The deuteranope's matrix in LMS (slide "The math: a dichromat is a projection"):
/// rows (1, 0, 0), (a, 0, b), (0, 0, 1), with a and b chosen so that the two anchor colours
/// `white` and `blue` (both in LMS) keep their M: `a·L + b·S = M` for both.
pub fn deutan_lms(white: [f32; 3], blue: [f32; 3]) -> [[f32; 3]; 3] {
    // a·wL + b·wS = wM and a·qL + b·qS = qM
    let det = white[0] * blue[2] - white[2] * blue[0];
    let a = (white[1] * blue[2] - white[2] * blue[1]) / det;
    let b = (white[0] * blue[1] - white[1] * blue[0]) / det;
    [[1.0, 0.0, 0.0], [a, 0.0, b], [0.0, 0.0, 1.0]]
}

/// C5a. The factors of a decorrelation stretch (slide "The math: decorrelation stretch"):
/// `(σ̄ / sqrt(λ_k))^amount` with `σ̄² = (λ1 + λ2 + λ3) / 3`, for the three eigenvalues of the
/// colour covariance; `amount` 0 = no stretch, 1 = full stretch.
pub fn stretch_factors(lambda: [f32; 3], amount: f32) -> [f32; 3] {
    let mean_var = (lambda[0] + lambda[1] + lambda[2]) / 3.0;
    lambda.map(|l| (mean_var / l).sqrt().powf(amount))
}

/// C5b. Decorrelation stretch of a colour image: the mean colour μ and the 3 × 3 covariance
/// S of all pixels (divide by n), its eigenvectors U and eigenvalues λ, then every pixel
/// `x' = μ + U diag(factors) Uᵀ (x − μ)`. Values are **not** clipped.
pub fn decorrelation_stretch(img: &Rgb, amount: f32) -> Rgb {
    let n = (img.width() * img.height()) as f64;
    let mut mu = [0.0f64; 3];
    for p in img.pixels() {
        for k in 0..3 {
            mu[k] += p[k] as f64 / n;
        }
    }
    let mut s = vec![vec![0.0f64; 3]; 3];
    for p in img.pixels() {
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += (p[i] as f64 - mu[i]) * (p[j] as f64 - mu[j]) / n;
            }
        }
    }
    let (values, u) = linalg::sym_eigen(&s);
    let f = stretch_factors([values[0] as f32, values[1] as f32, values[2] as f32], amount);
    // the 3 × 3 matrix A = U diag(f) Uᵀ, applied to every pixel
    let mut a = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                a[i][j] += (u[k][i] * u[k][j]) as f32 * f[k];
            }
        }
    }
    img.map(|p| {
        let d = [p[0] - mu[0] as f32, p[1] - mu[1] as f32, p[2] - mu[2] as f32];
        let mut out = [0.0f32; 3];
        for i in 0..3 {
            out[i] = mu[i] as f32 + a[i][0] * d[0] + a[i][1] * d[1] + a[i][2] * d[2];
        }
        out
    })
}
