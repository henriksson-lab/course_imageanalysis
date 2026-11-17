// Model solution for the coding exercises of "RGB: gamma, linear maps and white balance".

use imgcourse::{linalg, Image};

/// A colour image: `[red, green, blue]` per pixel, values in [0, 1].
pub type Rgb = Image<[f32; 3]>;

/// C1a. sRGB decoding: a stored value `c` in [0, 1] (the 8-bit code divided by 255) to
/// linear light (slide "The math: the sRGB transfer function").
pub fn srgb_decode(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// C1b. sRGB encoding: linear light in [0, 1] to the stored value; the inverse of `srgb_decode`.
pub fn srgb_encode(v: f32) -> f32 {
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// C2. Downsample an encoded colour image by `factor`: every `factor × factor` block becomes
/// one pixel, the average of its pixels. With `linear` false, average the stored values; with
/// `linear` true, decode → average → encode. (The width and height are divided by `factor`;
/// leftover pixels at the right and bottom are dropped.)
pub fn downsample(img: &Rgb, factor: usize, linear: bool) -> Rgb {
    let n = (factor * factor) as f32;
    Image::from_fn(img.width() / factor, img.height() / factor, |x, y| {
        let mut sum = [0.0f32; 3];
        for dy in 0..factor {
            for dx in 0..factor {
                let p = img[(factor * x + dx, factor * y + dy)];
                for c in 0..3 {
                    sum[c] += if linear { srgb_decode(p[c]) } else { p[c] };
                }
            }
        }
        sum.map(|s| if linear { srgb_encode(s / n) } else { s / n })
    })
}

/// C3. The RGB → XYZ matrix of a display (`m[row][column]`) from the chromaticities (x, y)
/// of its three primaries and of its white (slide "The math: the matrix from primaries and
/// white point"): primary k has XYZ proportional to `p_k = (x/y, 1, (1 − x − y)/y)`; the
/// scales s solve `[p_R p_G p_B] s = w`, with the white `w` scaled to Y = 1; column k of
/// the matrix is `s_k p_k`.
pub fn rgb_to_xyz_matrix(primaries: [[f32; 2]; 3], white: [f32; 2]) -> [[f32; 3]; 3] {
    let vec_of = |c: [f32; 2]| [c[0] / c[1], 1.0, (1.0 - c[0] - c[1]) / c[1]];
    let p = primaries.map(vec_of);
    let w = vec_of(white);
    // the matrix with the vectors p_k as columns
    let a: Vec<Vec<f64>> = (0..3).map(|row| (0..3).map(|k| p[k][row] as f64).collect()).collect();
    let b: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    let s = linalg::solve(&a, &b).expect("the primaries must not lie on one line");
    let mut m = [[0.0f32; 3]; 3];
    for row in 0..3 {
        for k in 0..3 {
            m[row][k] = s[k] as f32 * p[k][row];
        }
    }
    m
}

/// C4. Y′CbCr (JPEG) of one pixel with codes 0..255: luma `Y′ = 0.299 R′ + 0.587 G′ + 0.114 B′`,
/// `Cb = 128 + 0.564 (B′ − Y′)`, `Cr = 128 + 0.713 (R′ − Y′)` (not clipped).
pub fn ycbcr(rgb: [f32; 3]) -> [f32; 3] {
    let y = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
    [y, 128.0 + 0.564 * (rgb[2] - y), 128.0 + 0.713 * (rgb[0] - y)]
}

/// C5. Grey-world white balance of a **linear** colour image: the gains that make the mean
/// of every channel equal to the mean of the green channel, `g_c = mean_G / mean_c`, and the
/// image with every channel multiplied by its gain. Returns `(gains, corrected)`.
pub fn grey_world(img: &Rgb) -> ([f32; 3], Rgb) {
    let mut mean = [0.0f32; 3];
    for p in img.pixels() {
        for c in 0..3 {
            mean[c] += p[c];
        }
    }
    let gains = [mean[1] / mean[0], 1.0, mean[1] / mean[2]];
    let corrected = img.map(|p| [p[0] * gains[0], p[1] * gains[1], p[2] * gains[2]]);
    (gains, corrected)
}
