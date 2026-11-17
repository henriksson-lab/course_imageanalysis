// Model solution for the coding exercises of "How many false spots? Thresholds from noise statistics".

use imgcourse::Image;

/// The median of a list of numbers (the mean of the two middle ones for an even count).
pub fn median(values: &[f32]) -> f32 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let m = v.len() / 2;
    if v.len() % 2 == 1 { v[m] } else { 0.5 * (v[m - 1] + v[m]) }
}

/// C1. The robust noise estimate: the median absolute deviation from the median, divided by
/// 0.6745 (for normal noise this is the standard deviation).
pub fn mad_sigma(values: &[f32]) -> f32 {
    let med = median(values);
    let deviations: Vec<f32> = values.iter().map(|x| (x - med).abs()).collect();
    median(&deviations) / 0.6745
}

/// C2. The noise sd of an image, estimated from a high-pass version of it: the differences
/// of horizontal neighbours, `d = I(x+1, y) - I(x, y)`. Their sd is √2 times the noise sd
/// (the variances of the two pixels add), and a smooth background drops out.
pub fn estimate_noise(img: &Image<f32>) -> f32 {
    let mut d = Vec::new();
    for y in 0..img.height() {
        for x in 0..img.width() - 1 {
            d.push(img[(x + 1, y)] - img[(x, y)]);
        }
    }
    mad_sigma(&d) / 2f32.sqrt()
}

/// C3. The local maxima of a response measured in noise sd (`z`) that are above `k`: pixels
/// larger than all 8 neighbours (the outermost rows and columns are skipped). Returns their
/// positions `(x, y)`. With `k = f32::NEG_INFINITY` it returns all local maxima.
pub fn maxima_above(z: &Image<f32>, k: f32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for y in 1..z.height() - 1 {
        for x in 1..z.width() - 1 {
            let v = z[(x, y)];
            let mut is_max = v > k;
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    if (dx, dy) != (0, 0) && z[((x as isize + dx) as usize, (y as isize + dy) as usize)] >= v {
                        is_max = false;
                    }
                }
            }
            if is_max {
                out.push((x, y));
            }
        }
    }
    out
}

/// C4. The expected number of noise maxima above `u` sd in an image of `area` pixels of white
/// noise smoothed by a Gaussian of width `sigma_f` (the Euler-characteristic formula):
/// `area / (2 sigma_f²) · u e^(−u²/2) / (2π)^(3/2)`.
pub fn euler_expected(area: f32, sigma_f: f32, u: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    area / (2.0 * sigma_f * sigma_f) * u * (-u * u / 2.0).exp() / two_pi.powf(1.5)
}

/// C5. The Anscombe transform `2 sqrt(x + 3/8)` of every pixel: Poisson counts become values
/// whose noise sd is about 1, whatever the brightness.
pub fn anscombe(img: &Image<f32>) -> Image<f32> {
    img.map(|x| 2.0 * (x + 3.0 / 8.0).sqrt())
}
