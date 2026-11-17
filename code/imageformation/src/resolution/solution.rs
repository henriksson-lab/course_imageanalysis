// Model solution for the coding exercises of "Resolution, sampling and measuring the PSF".

use imgcourse::Image;
use std::f32::consts::PI;

/// The Airy pattern of an ideal lens at distance `r` from its centre, peak 1:
/// `[2 J1(v) / v]²` with `v = 2π NA r / λ` (`r` and `wavelength` in the same unit).
/// Given (not an exercise). J1 is computed from its integral, `(1/π) ∫ cos(τ − v sin τ) dτ`
/// over `0..π`, with Simpson's rule.
pub fn airy(r: f32, na: f32, wavelength: f32) -> f32 {
    let v = 2.0 * PI * na * r.abs() / wavelength;
    if v < 1e-4 {
        return 1.0;
    }
    let n = 64;
    let step = PI / n as f32;
    let mut j1 = 0.0;
    for i in 0..=n {
        let t = i as f32 * step;
        let w = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        j1 += w * (t - v * t.sin()).cos();
    }
    j1 *= step / 3.0 / PI;
    (2.0 * j1 / v).powi(2)
}

/// C1. The profile through two equal points at `−d/2` and `+d/2`, each imaged as the PSF
/// `h` (a function of the distance from the point), at the positions `xs`:
/// `I(x) = h(x − d/2) + h(x + d/2)`.
pub fn two_points(h: impl Fn(f32) -> f32, d: f32, xs: &[f32]) -> Vec<f32> {
    xs.iter().map(|&x| h(x - d / 2.0) + h(x + d / 2.0)).collect()
}

/// C2. The dip between two peaks: `1 − (value in the middle) / (largest value)`, for a profile
/// whose middle sample (index `len / 2`) lies halfway between the two points. 0 when the
/// middle is the highest point (no dip: not resolved).
pub fn dip(profile: &[f32]) -> f32 {
    let middle = profile[profile.len() / 2];
    let peak = profile.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    1.0 - middle / peak
}

/// C3. The full width at half maximum of a profile with one peak, sampled every `dx`:
/// walk left and right from the peak to where the profile drops below half the peak, and
/// place each crossing by linear interpolation between the two samples around it.
pub fn fwhm(profile: &[f32], dx: f32) -> f32 {
    let (ipeak, &peak) = profile.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    let half = peak / 2.0;
    // right crossing: between i − 1 (above half) and i (below)
    let mut right = (profile.len() - 1) as f32;
    for i in ipeak + 1..profile.len() {
        if profile[i] < half {
            let (a, b) = (profile[i - 1], profile[i]);
            right = (i - 1) as f32 + (a - half) / (a - b);
            break;
        }
    }
    // left crossing: between i + 1 (above half) and i (below)
    let mut left = 0.0;
    for i in (0..ipeak).rev() {
        if profile[i] < half {
            let (a, b) = (profile[i + 1], profile[i]);
            left = (i + 1) as f32 - (a - half) / (a - b);
            break;
        }
    }
    (right - left) * dx
}

/// C4. The centre of a bead with sub-pixel accuracy: the centroid (brightness-weighted mean
/// position) of the pixels within `radius` of the brightest pixel (a square window), after
/// subtracting `background` (values that become negative count as 0).
pub fn centroid(img: &Image<f32>, radius: usize, background: f32) -> (f32, f32) {
    let (mut bx, mut by, mut best) = (0, 0, f32::NEG_INFINITY);
    for (x, y) in img.coords() {
        if img[(x, y)] > best {
            (bx, by, best) = (x, y, img[(x, y)]);
        }
    }
    let (mut sw, mut sx, mut sy) = (0.0, 0.0, 0.0);
    for y in by.saturating_sub(radius)..(by + radius + 1).min(img.height()) {
        for x in bx.saturating_sub(radius)..(bx + radius + 1).min(img.width()) {
            let w = (img[(x, y)] - background).max(0.0);
            sw += w;
            sx += w * x as f32;
            sy += w * y as f32;
        }
    }
    (sx / sw, sy / sw)
}

/// C5. Average beads into one PSF. Each bead is sampled on a grid `step` pixels fine,
/// `(2 n + 1) × (2 n + 1)` points centred on its sub-pixel centre (bilinear interpolation,
/// 0 outside the image), with `background` subtracted. The beads are averaged point by point,
/// and the result is normalised to sum 1.
pub fn average_beads(img: &Image<f32>, centres: &[(f32, f32)], n: usize, step: f32, background: f32) -> Image<f32> {
    let size = 2 * n + 1;
    let mut sum = Image::new(size, size, 0.0);
    for &(cx, cy) in centres {
        for (i, j) in sum.coords() {
            let x = cx + (i as f32 - n as f32) * step;
            let y = cy + (j as f32 - n as f32) * step;
            sum[(i, j)] += bilinear(img, x, y) - background;
        }
    }
    let total = sum.sum();
    sum.map(|v| v / total)
}

/// Bilinear interpolation at a position between pixels (the images unit), 0 outside.
/// Given (not an exercise).
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let get = |dx: isize, dy: isize| img.get_border(x0 as isize + dx, y0 as isize + dy, imgcourse::Border::Zero);
    (1.0 - fy) * ((1.0 - fx) * get(0, 0) + fx * get(1, 0)) + fy * ((1.0 - fx) * get(0, 1) + fx * get(1, 1))
}
