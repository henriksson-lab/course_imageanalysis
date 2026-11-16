//! Test images that need no files.

use crate::{Image, Rng};

/// The course micrograph: a transmission electron micrograph of a mesenchymal stem cell,
/// 256 × 256, grey values in `[0, 1]`. Image: Robert M. Hunt, CC BY 3.0, via Wikimedia Commons.
pub fn cell() -> Image<f32> {
    crate::io::decode_gray(include_bytes!("../data/cell256.png"))
}

/// A white letter F on black, `size × size`: easy to see if an image was mirrored or turned.
pub fn letter_f(size: usize) -> Image<f32> {
    let s = size as f32;
    Image::from_fn(size, size, |x, y| {
        let (u, v) = (x as f32 / s, y as f32 / s);
        let stem = (0.25..0.40).contains(&u) && (0.15..0.85).contains(&v);
        let top = (0.25..0.75).contains(&u) && (0.15..0.30).contains(&v);
        let middle = (0.25..0.62).contains(&u) && (0.45..0.58).contains(&v);
        if stem || top || middle { 1.0 } else { 0.0 }
    })
}

/// A vertical step edge: `left` on the left half, `right` on the right half.
pub fn step(width: usize, height: usize, left: f32, right: f32) -> Image<f32> {
    Image::from_fn(width, height, |x, _| if x < width / 2 { left } else { right })
}

/// A checkerboard of `square × square` blocks, 0 and 1.
pub fn checker(width: usize, height: usize, square: usize) -> Image<f32> {
    Image::from_fn(width, height, |x, y| ((x / square + y / square) % 2) as f32)
}

/// A copy of `img` with Gaussian noise of standard deviation `sigma` added (not clipped).
pub fn add_noise(img: &Image<f32>, sigma: f32, seed: u64) -> Image<f32> {
    let mut rng = Rng::new(seed);
    Image::from_fn(img.width(), img.height(), |x, y| img[(x, y)] + sigma * rng.normal())
}

/// A synthetic fluorescence image of `n` nuclei: bright ellipses of different sizes,
/// brightness and orientation, some touching, on a dark uneven background, blurred and with
/// noise. Returns the image (values roughly 0..1) and the true labels (0 = background,
/// 1..=n = the nuclei; where two overlap, the later one wins). Same `seed`, same image.
pub fn nuclei(size: usize, n: usize, seed: u64) -> (Image<f32>, Image<u32>) {
    let mut rng = Rng::new(seed);
    let s = size as f32;
    struct Ell { cx: f32, cy: f32, a: f32, b: f32, c: f32, sn: f32, bright: f32 }
    let ells: Vec<Ell> = (0..n)
        .map(|_| {
            let a = s * (0.04 + 0.03 * rng.uniform());
            let b = a * (0.6 + 0.4 * rng.uniform());
            let t = std::f32::consts::PI * rng.uniform();
            Ell {
                cx: a + (s - 2.0 * a) * rng.uniform(),
                cy: a + (s - 2.0 * a) * rng.uniform(),
                a,
                b,
                c: t.cos(),
                sn: t.sin(),
                bright: 0.5 + 0.4 * rng.uniform(),
            }
        })
        .collect();
    let labels = Image::from_fn(size, size, |x, y| {
        let mut l = 0;
        for (i, e) in ells.iter().enumerate() {
            let (dx, dy) = (x as f32 - e.cx, y as f32 - e.cy);
            let (u, v) = (e.c * dx + e.sn * dy, -e.sn * dx + e.c * dy);
            if (u / e.a).powi(2) + (v / e.b).powi(2) <= 1.0 {
                l = i as u32 + 1;
            }
        }
        l
    });
    let sharp = Image::from_fn(size, size, |x, y| {
        let bg = 0.08 + 0.06 * (x as f32 / s) + 0.04 * (y as f32 / s);
        match labels[(x, y)] {
            0 => bg,
            l => {
                let e = &ells[l as usize - 1];
                // a little texture inside each nucleus
                bg + e.bright * (0.85 + 0.15 * ((x as f32 * 0.7).sin() * (y as f32 * 0.5).cos()))
            }
        }
    });
    // blur with a small binomial kernel (the optics), then add noise (the camera)
    let w = [1.0, 4.0, 6.0, 4.0, 1.0].map(|v| v / 16.0);
    let pass = |img: &Image<f32>, along_x: bool| {
        Image::from_fn(size, size, |x, y| {
            (0..5)
                .map(|k| {
                    let d = k as isize - 2;
                    let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                    w[k] * img.get_border(xx, yy, crate::Border::Clamp)
                })
                .sum()
        })
    };
    let blurred = pass(&pass(&sharp, true), false);
    let img = Image::from_fn(size, size, |x, y| blurred[(x, y)] + 0.03 * rng.normal());
    (img, labels)
}
