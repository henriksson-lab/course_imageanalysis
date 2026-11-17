//! Simulated images of a single-chain FRET biosensor in cells (donor and acceptor channel,
//! photon noise, a background), and a colour map.
//! Not part of the exercise: read it if you like.

use imgcourse::{samples, Image, Rng};

/// The sensor: ratio when empty, when saturated, and the dissociation constant (nM).
pub const R_MIN: f32 = 0.5;
pub const R_MAX: f32 = 2.5;
pub const KD: f32 = 200.0;

/// The ratio of the sensor at concentration `c` (nM): R_min + (R_max − R_min) c / (c + K_d).
pub fn true_ratio(c: f32) -> f32 {
    R_MIN + (R_MAX - R_MIN) * c / (c + KD)
}

/// A field of cells with the biosensor.
pub struct Field {
    /// Acceptor and donor channel (photons, background included).
    pub acceptor: Image<f32>,
    pub donor: Image<f32>,
    /// The cells: 0 background, 1, 2, … the cells.
    pub labels: Image<u32>,
    /// The true concentration (nM) at every pixel (0 outside the cells).
    pub concentration: Image<f32>,
}

/// Cells of different thickness and expression; the concentration is 150 nM in most cells, a
/// gradient from 50 to 1000 nM across cell 1, and 2000 nM in cell 2. `photons`: the typical
/// donor signal per pixel; `background`: photons per pixel added to both channels;
/// `bleed`: donor light seen in the acceptor channel (adds to the ratio). Same seed, same field.
pub fn field(photons: f32, background: f32, bleed: f32, seed: u64) -> Field {
    let (shape, labels) = samples::nuclei(128, 8, seed);
    let mut rng = Rng::new(seed + 1);
    let expression: Vec<f32> = (0..8).map(|_| 0.4 + 1.2 * rng.uniform()).collect();
    // where each cell lies (for the gradient across cell 1)
    let (mut x_lo, mut x_hi) = (128.0f32, 0.0f32);
    for (x, y) in labels.coords() {
        if labels[(x, y)] == 1 {
            x_lo = x_lo.min(x as f32);
            x_hi = x_hi.max(x as f32);
        }
    }
    let concentration = labels.map_xy(|x, _, l| match l {
        0 => 0.0,
        1 => 50.0 + 950.0 * (x as f32 - x_lo) / (x_hi - x_lo).max(1.0),
        2 => 2000.0,
        _ => 150.0,
    });
    let mut donor = Image::new(128, 128, 0.0f32);
    let mut acceptor = Image::new(128, 128, 0.0f32);
    for (x, y) in labels.coords() {
        let l = labels[(x, y)] as usize;
        // thickness and amount of sensor: the nucleus image (bright = thick) times the expression
        let amount = if l > 0 { photons * expression[l - 1] * shape[(x, y)].max(0.0) / 0.7 } else { 0.0 };
        let r = true_ratio(concentration[(x, y)]);
        donor[(x, y)] = rng.poisson(amount + background);
        acceptor[(x, y)] = rng.poisson((r + bleed) * amount + background);
    }
    Field { acceptor, donor, labels, concentration }
}

/// A colour map from blue (t = 0) through green (t = 0.5) to red (t = 1).
pub fn colour_map(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let ramp = |c: f32| (1.0 - (4.0 * (t - c)).abs()).clamp(0.0, 1.0);
    [ramp(0.75) + if t > 0.75 { 1.0 - ramp(0.75) } else { 0.0 }, ramp(0.5), ramp(0.25) + if t < 0.25 { 1.0 - ramp(0.25) } else { 0.0 }]
}
