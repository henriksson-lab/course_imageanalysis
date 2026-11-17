//! Noise images, Gaussian filters, the normal tail and drawing for the threshold exercises.
//! Not part of the exercise: read it if you like.

use imgcourse::{Border, Image, Rng};

/// An image of white Gaussian noise: mean `mean`, standard deviation `sd`.
pub fn noise_image(size: usize, mean: f32, sd: f32, seed: u64) -> Image<f32> {
    let mut rng = Rng::new(seed);
    Image::from_fn(size, size, |_, _| mean + sd * rng.normal())
}

/// The weights of a normalised 1D Gaussian of width `sigma`, radius ceil(3 sigma).
pub fn gauss_weights(sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = w.iter().sum();
    w.iter().map(|v| v / s).collect()
}

/// Blur with a normalised Gaussian of width `sigma`, along rows and then columns (mirror outside).
pub fn gaussian_blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let w = gauss_weights(sigma);
    let r = (w.len() / 2) as isize;
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for (i, wi) in w.iter().enumerate() {
                let d = i as isize - r;
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += wi * src.get_border(xx, yy, Border::Mirror);
            }
            sum
        })
    };
    pass(&pass(img, true), false)
}

/// sqrt(sum of squared weights) of the 2D filter "Gaussian `s1` minus Gaussian `s2`" (`s2 = 0`:
/// the Gaussian alone): white noise of sd 1 has this sd after the filter.
pub fn filter_noise_factor(s1: f32, s2: f32) -> f32 {
    let (a, b) = (gauss_weights(s1), if s2 > 0.0 { gauss_weights(s2) } else { vec![0.0] });
    let r = a.len().max(b.len()) / 2;
    let at = |w: &Vec<f32>, i: isize| {
        let c = (w.len() / 2) as isize;
        let j = i + c;
        if j >= 0 && (j as usize) < w.len() { w[j as usize] } else { 0.0 }
    };
    let mut ss = 0.0;
    for dy in -(r as isize)..=r as isize {
        for dx in -(r as isize)..=r as isize {
            let v = at(&a, dx) * at(&a, dy) - at(&b, dx) * at(&b, dy);
            ss += v * v;
        }
    }
    ss.sqrt()
}

/// P(Z > u) for a standard normal Z (an erfc approximation, good to about 1e-7).
pub fn normal_tail(u: f32) -> f32 {
    let x = u as f64 / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = t * (-x * x - 1.265_512_23
        + t * (1.000_023_68 + t * (0.374_091_96 + t * (0.096_784_18 + t * (-0.186_288_06
        + t * (0.278_868_07 + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
        .exp();
    let erfc = if x >= 0.0 { y } else { 2.0 - y };
    (0.5 * erfc) as f32
}

/// The response in grey (−5 to +5 noise sd) with a red cross at every pixel in `marks`.
pub fn draw_marks(z: &Image<f32>, marks: &[(usize, usize)]) -> Image<[f32; 3]> {
    let mut out = z.map(|v| {
        let g = (0.5 + v / 10.0).clamp(0.0, 1.0);
        [g, g, g]
    });
    for &(x, y) in marks {
        for d in -2..=2isize {
            for (xx, yy) in [(x as isize + d, y as isize), (x as isize, y as isize + d)] {
                if out.contains(xx, yy) {
                    out[(xx as usize, yy as usize)] = [1.0, 0.1, 0.1];
                }
            }
        }
    }
    out
}
