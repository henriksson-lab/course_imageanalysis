// Coding exercises for "Images as landscapes: watershed by flooding". Replace each `todo!()`
// with your code, then check it with `cargo test -p watershed ws_landscape::` and look at `cargo run -p watershed -- ws-landscape`.
#![allow(unused_variables, unused_imports)]
// All functions work on a 1D signal (a profile), samples 0, 1, ..., n − 1.

/// The size of the derivative by central differences, |f(i+1) − f(i−1)| / 2, with the end
/// samples repeated (f(−1) = f(0), f(n) = f(n−1)). (A helper, as in exercise B4.)
pub fn abs_derivative(f: &[f32]) -> Vec<f32> {
    let n = f.len();
    (0..n).map(|i| (f[(i + 1).min(n - 1)] - f[i.saturating_sub(1)]).abs() / 2.0).collect()
}

/// C1. The regional minima: runs of equal samples whose neighbours (the sample left of the
/// run and the one right of it, where they exist) are both higher. Returns a label per sample:
/// 1, 2, 3, ... for the minima from left to right, 0 elsewhere.
///
/// Hint: walk along the signal run by run: find where the run of equal values that starts at
/// `start` ends, look at the samples just outside it, then continue after the run.
pub fn regional_minima(f: &[f32]) -> Vec<u32> {
    todo!("C1: regional_minima")
}

/// C2. A drop of water placed on sample `start`: at each step it moves to the lower of its two
/// neighbours, if that is lower than where it is (on a tie, to the left). Returns the sample
/// where it stops.
pub fn drop_of_water(f: &[f32], start: usize) -> usize {
    todo!("C2: drop_of_water")
}

/// C3. Flooding from the labelled `sources` (0 = not a source). For every water level L, the
/// distinct heights in increasing order, grow all lakes at the same speed, in rounds:
/// (a) every unlabelled sample with f ≤ L next to a lake takes its label (the left neighbour's
/// if both are labelled), all samples at once, so that each lake grows by one sample;
/// (b) water runs downhill at once: an unlabelled sample with f < L next to a lake takes its
/// label straight away (sweep from left to right, repeat until nothing changes).
/// Repeat the rounds until nothing changes, then go to the next level. Returns the labels.
///
/// Hint: `let mut levels = f.to_vec(); levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
/// levels.dedup();` gives the levels. In a round, read the labels of the last round and write
/// into a copy, so that a lake grows by only one sample per round.
pub fn flood(f: &[f32], sources: &[u32]) -> Vec<u32> {
    todo!("C3: flood")
}

/// C4. The dams: wherever two neighbouring samples i, i + 1 have different labels, the dam is
/// on the higher of the two (the left one if they are equal), and its height is that sample's
/// height. Returns (position, height) for every dam, from left to right.
pub fn dams(f: &[f32], labels: &[u32]) -> Vec<(usize, f32)> {
    todo!("C4: dams")
}

/// C5. The h-minima transform: fill every basin by h. Start with g = f + h and erode it,
/// never going below f: g(i) = max(f(i), min(g(i−1), g(i), g(i+1))), all samples at once,
/// until nothing changes. Its regional minima are the minima of f deeper than h.
///
/// Hint: compute the new values into a new vector and compare it with the old one
/// (`next == g`) to know when to stop.
pub fn h_minima(f: &[f32], h: f32) -> Vec<f32> {
    todo!("C5: h_minima")
}
