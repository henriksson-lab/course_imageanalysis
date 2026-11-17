// Coding exercises for "Domain shift and adaptation". Replace each `todo!()` with your code,
// then check it with `cargo test -p finetuning domain_shift::` and look at
// `cargo run -p finetuning -- domain-shift`.
#![allow(unused_variables, unused_imports)]

pub use super::model::*;

/// C1. Mean and standard deviation of each of the two features over a set of cells (no labels).
///
/// Hint: divide by the number of cells (not n − 1), as the normalisation layers of networks do.
pub fn stats(cells: &[Cell]) -> ([f32; 2], [f32; 2]) {
    todo!("C1: stats")
}

/// C2. Standardise every feature with given statistics: (f − mean) / sd. Labels are kept.
pub fn standardise(cells: &[Cell], mean: [f32; 2], sd: [f32; 2]) -> Vec<Cell> {
    todo!("C2: standardise")
}

/// C3. A shift score that needs no labels: the largest |mean_B − mean_A| / sd_A over the features.
pub fn shift_score(a: &[Cell], b: &[Cell]) -> f32 {
    todo!("C3: shift_score")
}

/// C4. Augmentation of the intensity: `copies` copies of every cell, each with its intensity
/// (feature 0) multiplied by a random gain in [0.4, 1.2] and shifted by a random offset in
/// [−0.05, 0.2]. The size (feature 1) and the class stay.
///
/// Hint: `imgcourse::Rng::new(seed)`; draw the gain first, then the offset, with `rng.uniform()`.
pub fn augment(cells: &[Cell], copies: usize, seed: u64) -> Vec<Cell> {
    todo!("C4: augment")
}
