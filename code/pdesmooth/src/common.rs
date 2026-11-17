//! Helpers shared by the units of day 10 (not exercises): the Gaussian blur to compare the
//! flows with, and the test card of the smoothers and bilateral/non-local-means programs.

use imgcourse::{Border, Image};

/// A Gaussian blur (two passes of a sampled, normalised 1D Gaussian of radius 4σ, mirror
/// border).
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

/// A test card with known truth: a dark background (40), a bright square (200), discs of
/// radius 2 to 8, a dot lattice of period 6, fine stripes, a ramp and a 1-pixel line
/// (grey values / 255). The line is at x = 120, rows 132-151.
pub fn test_card(size: usize) -> Image<f32> {
    Image::from_fn(size, size, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let mut v = 40.0;
        if (16..64).contains(&x) && (16..64).contains(&y) {
            v = 200.0;
        }
        for (i, r) in [2.0, 3.0, 5.0, 8.0].iter().enumerate() {
            let (cx, cy) = (84.0 + 20.0 * i as f32, 24.0);
            if (xf - cx).powi(2) + (yf - cy).powi(2) <= r * r {
                v = 160.0;
            }
        }
        if (16..64).contains(&x) && (76..124).contains(&y) && x % 6 < 2 && y % 6 < 2 {
            v = 180.0; // a lattice of 2 × 2 dots
        }
        if (76..148).contains(&x) && (76..124).contains(&y) && x % 4 < 2 {
            v = 120.0; // stripes of period 4
        }
        if (132..152).contains(&y) {
            v = if x < 16 { 40.0 } else if x < 96 { 40.0 + 2.0 * (x - 16) as f32 } else { 200.0 };
            if x == 120 {
                v = 40.0; // a 1-pixel line
            }
        }
        v / 255.0
    })
}
