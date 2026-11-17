// Coding exercises for "Keypoints: corners and blobs across scale". Replace each `todo!()` with your code,
// then check it with `cargo test -p day24 keypoints::` and look at `cargo run -p day24 -- keypoints`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// C1a. The scale σ of a DoG level in original pixels: `σ0 · 2^(octave + level / s)`,
/// with `s` levels per octave (slide "The math: octaves"). `level` may be fractional
/// (a refined keypoint).
///
/// Hint: `2f32.powf(e)` is 2 to the power e.
pub fn level_sigma(sigma0: f32, s: usize, octave: usize, level: f32) -> f32 {
    todo!("C1a: level_sigma")
}

/// C1b. One octave: `img` is taken to be blurred with σ0 already (it is `L_0`). Blur it
/// in steps of `k = 2^(1/s)` to get `s + 3` images `L_0 .. L_{s+2}`, each from the one
/// before, and subtract neighbours: `s + 2` DoGs `D_i = L_{i+1} − L_i`.
/// Returns `(blurred, dogs)`.
///
/// Hint: start with `vec![img.clone()]`; the next image is
/// `gaussian_blur(previous, sigma * (k * k - 1.0).sqrt())`, then `sigma *= k`.
/// A DoG is `blurred[i + 1].zip_map(&blurred[i], |a, b| a - b)`.
pub fn dog_octave(img: &Image<f32>, sigma0: f32, s: usize) -> (Vec<Image<f32>>, Vec<Image<f32>>) {
    todo!("C1b: dog_octave")
}

/// C2. Candidates: positions `(x, y, level)` where the DoG is larger than all 26
/// neighbours in the 3 × 3 × 3 cube (or smaller than all of them) and `|D| > threshold`.
/// Only the inner levels `1 ..= dogs.len() − 2` are searched, and not the outermost
/// row and column of pixels.
///
/// Hint: three loops (level, y, x), then three more over the cube; skip the centre
/// itself. Keep two flags, `is_max` and `is_min`, and compare with `>` and `<` (strictly).
pub fn local_extrema(dogs: &[Image<f32>], threshold: f32) -> Vec<(usize, usize, usize)> {
    todo!("C2: local_extrema")
}

/// C3. The vertex of the parabola through three samples `dm, d0, dp` at −1, 0, +1:
/// returns `(offset, value)` (slide "Refining a candidate"). If the three samples lie on a
/// line, the offset is 0.
///
/// Hint: offset = (dm − dp) / (2 (dm − 2 d0 + dp)); the slope is (dp − dm) / 2 and the
/// value is d0 + ½ · slope · offset (slide "The math: sub-pixel refinement").
pub fn parabola_peak(dm: f32, d0: f32, dp: f32) -> (f32, f32) {
    todo!("C3: parabola_peak")
}

/// C4. The edge test at pixel `(x, y)` of a DoG level: the 2 × 2 Hessian by finite
/// differences, then keep the point if `det > 0` and `tr² / det < (r + 1)² / r`.
///
/// Hint: read neighbours with `dog.get_border(x as isize + dx, y as isize + dy, Border::Clamp)`.
/// Dxx = D(+1, 0) − 2 D(0, 0) + D(−1, 0); Dxy = (D(+1, +1) − D(+1, −1) − D(−1, +1) + D(−1, −1)) / 4.
pub fn edge_test(dog: &Image<f32>, x: usize, y: usize, r: f32) -> bool {
    todo!("C4: edge_test")
}

/// C5a. The orientation histogram: 36 bins of 10°, bin `b` holds the directions in
/// `[10b, 10b + 10)` degrees. Every vote is `(magnitude, direction in degrees)`; the
/// direction may be any angle (e.g. −30° or 400°) and is wrapped into `[0, 360)`.
///
/// Hint: `direction.rem_euclid(360.0)` wraps an angle into [0, 360).
pub fn orientation_histogram(votes: &[(f32, f32)]) -> [f32; 36] {
    todo!("C5a: orientation_histogram")
}

/// C5b. The orientations of a keypoint: every bin that is a peak (larger than both
/// neighbours, which wrap around) and at least 80 % of the highest bin, refined with
/// `parabola_peak`. Returns angles in degrees in `[0, 360)`, in bin order.
///
/// Hint: the neighbours of bin `b` are `(b + 35) % 36` and `(b + 1) % 36`. The centre of
/// bin `b` is at 10b + 5 degrees; the parabola's offset is in bins.
pub fn dominant_orientations(hist: &[f32; 36]) -> Vec<f32> {
    todo!("C5b: dominant_orientations")
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
