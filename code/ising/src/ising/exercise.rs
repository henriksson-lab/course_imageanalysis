// Coding exercises for "Energy and probability: the Ising model".
// Replace each `todo!()` with your code, then check it with `cargo test -p ising ising::`
// and look at `cargo run -p ising -- ising`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// A lattice of spins, `+1` or `-1`, one per pixel.
pub type Spins = Image<i8>;

/// State number `index` of a `width × height` lattice: pixel `i` (row by row) is `+1` if bit
/// `i` of `index` is set, else `-1`. Indices `0 .. 2^(width·height)` list every state once.
pub fn state_from_index(index: usize, width: usize, height: usize) -> Spins {
    Image::from_fn(width, height, |x, y| if (index >> (y * width + x)) & 1 == 1 { 1 } else { -1 })
}

/// C1. The Ising energy `E = −J · Σ s_p s_q − h · Σ s_p`, the first sum over all pairs of
/// 4-neighbours, each pair once. With `periodic`, the right neighbour of the last column is
/// the first column (and likewise for rows), but only along sides of at least 3 pixels: on a
/// side of 2 the wrap-around pair would be the same pair again.
///
/// Hint: visit every pixel and add its pair with the right neighbour `(x + 1, y)` and the
/// one below `(x, y + 1)`; at the last column or row, use column 0 or row 0 if `periodic`
/// (and the side has at least 3 pixels), else nothing. Spins are `i8`: `s[(x, y)] as f32`.
pub fn energy(s: &Spins, j: f32, h: f32, periodic: bool) -> f32 {
    todo!("C1: energy")
}

/// C2. The energy levels of a `width × height` lattice: every different energy, with the
/// number of states that have it, sorted from the lowest energy up. Lists all `2^n` states.
/// Two energies closer than `1e-4` count as the same level.
///
/// Hint: `for index in 0..(1usize << n)` with `state_from_index`; keep a `Vec<(f32, usize)>`
/// and either add 1 to the count of the matching energy or push a new level. Sort at the end
/// with `levels.sort_by(|a, b| a.0.total_cmp(&b.0))`.
pub fn energy_levels(width: usize, height: usize, j: f32, h: f32, periodic: bool) -> Vec<(f32, usize)> {
    todo!("C2: energy_levels")
}

/// C3. The partition function `Z = Σ (number of states) · e^(−E/T)` over the levels, and the
/// probability of each level, `count · e^(−E/T) / Z` (in the order of `levels`).
pub fn level_probabilities(levels: &[(f32, usize)], t: f64) -> (f64, Vec<f64>) {
    todo!("C3: level_probabilities")
}

/// C4. `Z` of a ring of `n` spins (each spin has a left and a right neighbour, the last one
/// next to the first) with the transfer matrix
/// `M = [[e^((J+h)/T), e^(−J/T)], [e^(−J/T), e^((J−h)/T)]]`: `Z = trace(M^n)`.
/// `n` small matrix products instead of `2^n` states.
///
/// Hint: start from the identity matrix and multiply by `M` `n` times (2 × 2 products by
/// hand); the trace is the sum of the diagonal.
pub fn ring_partition_function(n: usize, j: f64, h: f64, t: f64) -> f64 {
    todo!("C4: ring_partition_function")
}
