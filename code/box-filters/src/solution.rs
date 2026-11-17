// Model solution for the coding exercises of "Box filters & local statistics".

use imgcourse::Image;

/// The integral image of the last unit (given): `table[(x, y)]` is the sum of `f(v)` over the
/// pixels `v` in columns `0..x` and rows `0..y`, so row 0 and column 0 are zero.
/// `integral(&img, |v| v)` sums the pixels, `integral(&img, |v| v * v)` their squares.
pub fn integral(img: &Image<f32>, f: impl Fn(f64) -> f64) -> Image<f64> {
    let mut s = Image::new(img.width() + 1, img.height() + 1, 0.0);
    for (x, y) in img.coords() {
        s[(x + 1, y + 1)] = f(img[(x, y)] as f64) + s[(x, y + 1)] + s[(x + 1, y)] - s[(x, y)];
    }
    s
}

/// The sum over columns `x0..=x1`, rows `y0..=y1` from a table (given): four lookups.
pub fn rect_sum(s: &Image<f64>, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    s[(x1 + 1, y1 + 1)] - s[(x0, y1 + 1)] - s[(x1 + 1, y0)] + s[(x0, y0)]
}

/// The window of radius `r` around `c` along an axis of length `n`, clipped to `0..n`:
/// returns the first and last index inside (given).
pub fn clip(c: usize, r: usize, n: usize) -> (usize, usize) {
    (c.saturating_sub(r), (c + r).min(n - 1))
}

/// C1. The box mean of width `2r + 1` along a row, with a running sum: start with the window
/// at position 0, then move one step at a time, adding the pixel that enters and subtracting
/// the one that leaves. At the ends the window is clipped; divide by the number of pixels
/// inside (exercise B1).
pub fn box_mean_1d(f: &[f32], r: usize) -> Vec<f32> {
    let n = f.len();
    let mut sum: f32 = f[..=r.min(n - 1)].iter().sum(); // the window at position 0
    let mut out = vec![0.0; n];
    for i in 0..n {
        if i > 0 {
            if i + r < n {
                sum += f[i + r]; // enters
            }
            if i > r {
                sum -= f[i - r - 1]; // leaves
            }
        }
        let (a, b) = clip(i, r, n);
        out[i] = sum / (b - a + 1) as f32;
    }
    out
}

/// C2. The box mean of width `2r + 1` at every pixel, from the integral image: clip the window
/// to the image and divide by the number of pixels inside (slide "Why the integral image wins
/// anyway").
pub fn box_mean(img: &Image<f32>, r: usize) -> Image<f32> {
    let s = integral(img, |v| v);
    let mut out = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let (x0, x1) = clip(x, r, img.width());
        let (y0, y1) = clip(y, r, img.height());
        let n = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
        out[(x, y)] = (rect_sum(&s, x0, y0, x1, y1) / n) as f32;
    }
    out
}

/// C3. The local mean `m` and local standard deviation `s` in the window of width `2r + 1`
/// (clipped) at every pixel, from two tables, of `I` and of `I²`: `s² = mean of I² − m²`
/// (slide "Local mean and local standard deviation"). A variance that comes out below zero
/// (rounding) counts as 0.
pub fn local_mean_sd(img: &Image<f32>, r: usize) -> (Image<f32>, Image<f32>) {
    let s1 = integral(img, |v| v);
    let s2 = integral(img, |v| v * v);
    let mut mean = Image::new(img.width(), img.height(), 0.0);
    let mut sd = Image::new(img.width(), img.height(), 0.0);
    for (x, y) in img.coords() {
        let (x0, x1) = clip(x, r, img.width());
        let (y0, y1) = clip(y, r, img.height());
        let n = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f64;
        let m = rect_sum(&s1, x0, y0, x1, y1) / n;
        let mean_sq = rect_sum(&s2, x0, y0, x1, y1) / n;
        mean[(x, y)] = m as f32;
        sd[(x, y)] = (mean_sq - m * m).max(0.0).sqrt() as f32;
    }
    (mean, sd)
}

/// C4. Local contrast normalisation `(I − m) / (s + eps)` with the window of width `2r + 1`
/// (slide "Local contrast normalisation").
pub fn normalise(img: &Image<f32>, r: usize, eps: f32) -> Image<f32> {
    let (m, s) = local_mean_sd(img, r);
    img.map_xy(|x, y, v| (v - m[(x, y)]) / (s[(x, y)] + eps))
}

/// C5. `n` passes of the box mean of width `2r + 1` (slide "Repeated boxes look Gaussian").
pub fn repeated_box(img: &Image<f32>, r: usize, n: usize) -> Image<f32> {
    let mut out = img.clone();
    for _ in 0..n {
        out = box_mean(&out, r);
    }
    out
}
