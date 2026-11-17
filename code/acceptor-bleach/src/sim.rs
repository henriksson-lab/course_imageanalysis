//! A simulated acceptor-photobleaching experiment: donor and acceptor images before and
//! after bleaching a box, with photon (Poisson) noise.
//! Not part of the exercise: read it if you like.

use imgcourse::{samples, Image, Rng};

/// The settings of one experiment.
#[derive(Clone, Debug)]
pub struct Bleach {
    /// Efficiency of a paired donor.
    pub e: f32,
    /// Fraction of donors with an acceptor.
    pub f: f32,
    /// Number of bleach pulses; each destroys 30 % of the acceptor left in the box.
    pub pulses: u32,
    /// Fraction of the donor lost per pulse (donor bleaching), everywhere.
    pub donor_loss: f32,
    /// Movement of the cells between before and after, in pixels.
    pub drift: (isize, isize),
    /// Typical unquenched donor photons per pixel.
    pub photons: f32,
    pub seed: u64,
}

impl Default for Bleach {
    fn default() -> Self {
        Bleach { e: 0.3, f: 1.0, pulses: 15, donor_loss: 0.0, drift: (0, 0), photons: 600.0, seed: 3 }
    }
}

/// The bleach box: the left half of the image, rows 16 to 111 (it covers several cells).
pub fn bleach_box(size: usize) -> Image<bool> {
    Image::from_fn(size, size, |x, y| x < size / 2 && (16..112).contains(&y))
}

/// The images of one experiment.
pub struct Experiment {
    pub donor_pre: Image<f32>,
    pub donor_post: Image<f32>,
    pub acceptor_pre: Image<f32>,
    pub acceptor_post: Image<f32>,
    /// The cells (label > 0).
    pub cells: Image<bool>,
    /// The bleach box.
    pub bleached: Image<bool>,
    /// The fraction of the acceptor left in the box after bleaching (the truth).
    pub acceptor_left: f32,
}

/// A Poisson-distributed count with mean `lambda` (Knuth's method below 30, a rounded normal
/// approximation above).
pub fn poisson(rng: &mut Rng, lambda: f32) -> f32 {
    if lambda < 30.0 {
        let limit = (-lambda).exp();
        let mut k = 0;
        let mut p = rng.uniform();
        while p > limit {
            k += 1;
            p *= rng.uniform();
        }
        k as f32
    } else {
        (lambda + lambda.sqrt() * rng.normal()).round().max(0.0)
    }
}

/// Simulate an experiment on a 128 × 128 field of cells (background already subtracted).
pub fn experiment(p: &Bleach) -> Experiment {
    let size = 128;
    let (shape, labels) = samples::nuclei(size, 12, p.seed);
    let cells = labels.map(|l| l > 0);
    let bleached = bleach_box(size);
    let mut rng = Rng::new(p.seed + 1);
    let left = 0.7f32.powi(p.pulses as i32);
    let donor_keep = (1.0 - p.donor_loss).powi(p.pulses as i32);
    // amount of donor (and acceptor) per pixel: the cell's brightness pattern
    let amount = |x: usize, y: usize| if cells[(x, y)] { p.photons * shape[(x, y)].max(0.0) / 0.7 } else { 0.0 };
    let pre_mean = |x: usize, y: usize| amount(x, y) * (1.0 - p.f * p.e);
    let post_mean = |x: usize, y: usize| {
        let k = if bleached[(x, y)] { left } else { 1.0 };
        // unpaired donors 1, paired donors with a bleached acceptor 1, with an intact one 1 − E
        amount(x, y) * ((1.0 - p.f) + p.f * ((1.0 - k) + k * (1.0 - p.e))) * donor_keep
    };
    let acc_post_mean = |x: usize, y: usize| amount(x, y) * if bleached[(x, y)] { left } else { 1.0 };
    let shifted = |x: usize, y: usize| {
        let (sx, sy) = (x as isize - p.drift.0, y as isize - p.drift.1);
        if sx >= 0 && sy >= 0 && (sx as usize) < size && (sy as usize) < size {
            Some((sx as usize, sy as usize))
        } else {
            None
        }
    };
    let donor_pre = Image::from_fn(size, size, |x, y| poisson(&mut rng, pre_mean(x, y)));
    let acceptor_pre = Image::from_fn(size, size, |x, y| poisson(&mut rng, amount(x, y)));
    // after: the scene has moved by `drift`
    let donor_post = Image::from_fn(size, size, |x, y| poisson(&mut rng, shifted(x, y).map_or(0.0, |(a, b)| post_mean(a, b))));
    let acceptor_post = Image::from_fn(size, size, |x, y| poisson(&mut rng, shifted(x, y).map_or(0.0, |(a, b)| acc_post_mean(a, b))));
    Experiment { donor_pre, donor_post, acceptor_pre, acceptor_post, cells, bleached, acceptor_left: left }
}
