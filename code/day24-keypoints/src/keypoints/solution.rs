// Model solution for the coding exercises of "Keypoints: corners and blobs across scale".

use imgcourse::{Border, Image};

/// C1a. The scale σ of a DoG level in original pixels: `σ0 · 2^(octave + level / s)`,
/// with `s` levels per octave (slide "The math: octaves"). `level` may be fractional
/// (a refined keypoint).
pub fn level_sigma(sigma0: f32, s: usize, octave: usize, level: f32) -> f32 {
    sigma0 * 2f32.powf(octave as f32 + level / s as f32)
}

/// C1b. One octave: `img` is taken to be blurred with σ0 already (it is `L_0`). Blur it
/// in steps of `k = 2^(1/s)` to get `s + 3` images `L_0 .. L_{s+2}`, each from the one
/// before, and subtract neighbours: `s + 2` DoGs `D_i = L_{i+1} − L_i`.
/// Returns `(blurred, dogs)`.
pub fn dog_octave(img: &Image<f32>, sigma0: f32, s: usize) -> (Vec<Image<f32>>, Vec<Image<f32>>) {
    let k = 2f32.powf(1.0 / s as f32);
    let mut blurred = vec![img.clone()];
    let mut sigma = sigma0;
    for _ in 0..s + 2 {
        // Blurring with σ_i·sqrt(k² − 1) takes σ_i to σ_{i+1} = k σ_i (variances add).
        let next = gaussian_blur(blurred.last().unwrap(), sigma * (k * k - 1.0).sqrt());
        blurred.push(next);
        sigma *= k;
    }
    let dogs = (0..s + 2).map(|i| blurred[i + 1].zip_map(&blurred[i], |a, b| a - b)).collect();
    (blurred, dogs)
}

/// C2. Candidates: positions `(x, y, level)` where the DoG is larger than all 26
/// neighbours in the 3 × 3 × 3 cube (or smaller than all of them) and `|D| > threshold`.
/// Only the inner levels `1 ..= dogs.len() − 2` are searched, and not the outermost
/// row and column of pixels.
pub fn local_extrema(dogs: &[Image<f32>], threshold: f32) -> Vec<(usize, usize, usize)> {
    let mut found = Vec::new();
    let (w, h) = (dogs[0].width(), dogs[0].height());
    for level in 1..dogs.len() - 1 {
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let v = dogs[level][(x, y)];
                if v.abs() <= threshold {
                    continue;
                }
                let (mut is_max, mut is_min) = (true, true);
                for dl in [level - 1, level, level + 1] {
                    for yy in y - 1..=y + 1 {
                        for xx in x - 1..=x + 1 {
                            if (dl, xx, yy) == (level, x, y) {
                                continue;
                            }
                            let n = dogs[dl][(xx, yy)];
                            is_max &= v > n;
                            is_min &= v < n;
                        }
                    }
                }
                if is_max || is_min {
                    found.push((x, y, level));
                }
            }
        }
    }
    found
}

/// C3. The vertex of the parabola through three samples `dm, d0, dp` at −1, 0, +1:
/// returns `(offset, value)` (slide "Refining a candidate"). If the three samples lie on a
/// line, the offset is 0.
pub fn parabola_peak(dm: f32, d0: f32, dp: f32) -> (f32, f32) {
    let curvature = dm - 2.0 * d0 + dp;
    if curvature == 0.0 {
        return (0.0, d0);
    }
    let offset = (dm - dp) / (2.0 * curvature);
    let slope = (dp - dm) / 2.0;
    (offset, d0 + 0.5 * slope * offset)
}

/// C4. The edge test at pixel `(x, y)` of a DoG level: the 2 × 2 Hessian by finite
/// differences, then keep the point if `det > 0` and `tr² / det < (r + 1)² / r`.
pub fn edge_test(dog: &Image<f32>, x: usize, y: usize, r: f32) -> bool {
    let d = |dx: isize, dy: isize| dog.get_border(x as isize + dx, y as isize + dy, Border::Clamp);
    let dxx = d(1, 0) - 2.0 * d(0, 0) + d(-1, 0);
    let dyy = d(0, 1) - 2.0 * d(0, 0) + d(0, -1);
    let dxy = (d(1, 1) - d(1, -1) - d(-1, 1) + d(-1, -1)) / 4.0;
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    det > 0.0 && tr * tr / det < (r + 1.0) * (r + 1.0) / r
}

/// C5a. The orientation histogram: 36 bins of 10°, bin `b` holds the directions in
/// `[10b, 10b + 10)` degrees. Every vote is `(magnitude, direction in degrees)`; the
/// direction may be any angle (e.g. −30° or 400°) and is wrapped into `[0, 360)`.
pub fn orientation_histogram(votes: &[(f32, f32)]) -> [f32; 36] {
    let mut hist = [0.0; 36];
    for &(magnitude, direction) in votes {
        let bin = (direction.rem_euclid(360.0) / 10.0) as usize % 36;
        hist[bin] += magnitude;
    }
    hist
}

/// C5b. The orientations of a keypoint: every bin that is a peak (larger than both
/// neighbours, which wrap around) and at least 80 % of the highest bin, refined with
/// `parabola_peak`. Returns angles in degrees in `[0, 360)`, in bin order.
pub fn dominant_orientations(hist: &[f32; 36]) -> Vec<f32> {
    let highest = hist.iter().copied().fold(0.0, f32::max);
    let mut angles = Vec::new();
    for b in 0..36 {
        let (left, right) = (hist[(b + 35) % 36], hist[(b + 1) % 36]);
        if hist[b] > left && hist[b] > right && hist[b] >= 0.8 * highest {
            let (offset, _) = parabola_peak(left, hist[b], right);
            // The centre of bin b is at 10b + 5 degrees.
            angles.push(((b as f32 + 0.5 + offset) * 10.0).rem_euclid(360.0));
        }
    }
    angles
}

/// Not an exercise: a Gaussian blur (two passes of a sampled, normalised 1D Gaussian of
/// radius 4σ, mirror border).
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (4.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Mirror);
            }
            sum / total
        })
    };
    pass(&pass(img, true), false)
}
