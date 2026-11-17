// Model solution for the coding exercises of "Ratio imaging and the noise of a ratio".

use crate::ratiometric::sim::colour_map;
use imgcourse::Image;

/// C1. The ratio image (slide "Background before dividing"): subtract the background from
/// both channels, then R = A / D where the sum of the two subtracted channels is at least
/// `min_sum` (the mask), and NaN ("no value") elsewhere.
pub fn ratio_image(a: &Image<f32>, d: &Image<f32>, bg_a: f32, bg_d: f32, min_sum: f32) -> Image<f32> {
    a.zip_map(d, |va, vd| {
        let (sa, sd) = (va - bg_a, vd - bg_d);
        if sa + sd >= min_sum {
            sa / sd
        } else {
            f32::NAN
        }
    })
}

/// C2a. The ratio of a region as a ratio of sums: Σ A / Σ D over the mask
/// (slide "Ratios: bias, tails and the right average").
pub fn ratio_of_sums(a: &Image<f32>, d: &Image<f32>, mask: &Image<bool>) -> f32 {
    let (mut sa, mut sd) = (0.0f64, 0.0f64);
    for p in a.coords().filter(|&p| mask[p]) {
        sa += a[p] as f64;
        sd += d[p] as f64;
    }
    (sa / sd) as f32
}

/// C2b. The mean of the pixel ratios A / D over the mask, skipping pixels with D ≤ 0.
pub fn mean_of_ratios(a: &Image<f32>, d: &Image<f32>, mask: &Image<bool>) -> f32 {
    let (mut sum, mut n) = (0.0f64, 0);
    for p in a.coords().filter(|&p| mask[p]) {
        if d[p] > 0.0 {
            sum += (a[p] / d[p]) as f64;
            n += 1;
        }
    }
    (sum / n as f64) as f32
}

/// C3. From ratio to concentration (slide "Calibration: from ratio to concentration"):
/// c = K_d (R − R_min) / (R_max − R).
pub fn calibrate(r: f32, kd: f32, r_min: f32, r_max: f32) -> f32 {
    kd * (r - r_min) / (r_max - r)
}

/// C4. Binning: every `k × k` block of pixels summed into one pixel (the image shrinks by `k`
/// in each direction; pixels that do not fill a whole block at the right or bottom are dropped).
pub fn bin(img: &Image<f32>, k: usize) -> Image<f32> {
    Image::from_fn(img.width() / k, img.height() / k, |bx, by| {
        let mut s = 0.0;
        for y in 0..k {
            for x in 0..k {
                s += img[(bx * k + x, by * k + y)];
            }
        }
        s
    })
}

/// C5. The intensity-modulated display (slide "Displaying a ratio"): the colour from the
/// ratio (`colour_map((R − r_lo) / (r_hi − r_lo))`, given), the lightness from the intensity:
/// the colour times min(intensity / i_max, 1). Pixels without a ratio (NaN) are black.
pub fn intensity_modulated(ratio: &Image<f32>, intensity: &Image<f32>, r_lo: f32, r_hi: f32, i_max: f32) -> Image<[f32; 3]> {
    ratio.zip_map(intensity, |r, i| {
        if r.is_nan() {
            return [0.0; 3];
        }
        let c = colour_map((r - r_lo) / (r_hi - r_lo));
        let light = (i / i_max).clamp(0.0, 1.0);
        [c[0] * light, c[1] * light, c[2] * light]
    })
}
