// Model solution for the coding exercises of "Domain shift and adaptation".

pub use super::model::*;

/// C1. Mean and standard deviation of each of the two features over a set of cells (no labels).
pub fn stats(cells: &[Cell]) -> ([f32; 2], [f32; 2]) {
    let n = cells.len() as f32;
    let mut mean = [0.0; 2];
    let mut sd = [0.0; 2];
    for c in cells {
        for k in 0..2 {
            mean[k] += c.f[k] / n;
        }
    }
    for c in cells {
        for k in 0..2 {
            sd[k] += (c.f[k] - mean[k]).powi(2) / n;
        }
    }
    (mean, [sd[0].sqrt(), sd[1].sqrt()])
}

/// C2. Standardise every feature with given statistics: (f − mean) / sd. Labels are kept.
pub fn standardise(cells: &[Cell], mean: [f32; 2], sd: [f32; 2]) -> Vec<Cell> {
    cells.iter().map(|c| Cell { f: [(c.f[0] - mean[0]) / sd[0], (c.f[1] - mean[1]) / sd[1]], class: c.class }).collect()
}

/// C3. A shift score that needs no labels: the largest |mean_B − mean_A| / sd_A over the features.
pub fn shift_score(a: &[Cell], b: &[Cell]) -> f32 {
    let (ma, sa) = stats(a);
    let (mb, _) = stats(b);
    (0..2).map(|k| (mb[k] - ma[k]).abs() / sa[k]).fold(0.0, f32::max)
}

/// C4. Augmentation of the intensity: `copies` copies of every cell, each with its intensity
/// (feature 0) multiplied by a random gain in [0.4, 1.2] and shifted by a random offset in
/// [−0.05, 0.2]. The size (feature 1) and the class stay.
pub fn augment(cells: &[Cell], copies: usize, seed: u64) -> Vec<Cell> {
    let mut rng = imgcourse::Rng::new(seed);
    let mut out = Vec::new();
    for c in cells {
        for _ in 0..copies {
            let g = 0.4 + 0.8 * rng.uniform();
            let o = -0.05 + 0.25 * rng.uniform();
            out.push(Cell { f: [g * c.f[0] + o, c.f[1]], class: c.class });
        }
    }
    out
}
