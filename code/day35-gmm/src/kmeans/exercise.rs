// Coding exercises for "Feature spaces and k-means". Replace each `todo!()` with your
// code, then check it with `cargo test -p day35 kmeans::` and look at `cargo run -p day35 -- kmeans`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Rng;

/// The squared distance `‖a − b‖²` of two points with any number of features. (Given.)
pub fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// C1. The assign step (slide "Lloyd's algorithm"): every point goes to its nearest centre
/// (the first one, if two are equally near). Returns the group of every point and
/// `J = Σ_i ‖x_i − m_c(i)‖²`.
///
/// Hint: use `dist2`; keep the index of the nearest centre so far.
pub fn assign(points: &[Vec<f32>], centres: &[Vec<f32>]) -> (Vec<usize>, f32) {
    todo!("C1: assign")
}

/// C2. The update step: every centre moves to the mean of the points assigned to it. A centre
/// without points stays where it is.
///
/// Hint: one sum vector and one count per centre.
pub fn update(points: &[Vec<f32>], groups: &[usize], centres: &[Vec<f32>]) -> Vec<Vec<f32>> {
    todo!("C2: update")
}

/// C3. k-means (Lloyd's algorithm) from the given starting centres: assign, update, repeat,
/// until no point changes its group (or after `max_iterations`). Returns the final centres,
/// the groups and the value of `J` after every assign step (slide "What k-means minimises").
///
/// Hint: start with an empty `groups` vector, so that the first assign step always counts as a change.
/// Push J to the trace before checking whether the groups changed.
pub fn kmeans(points: &[Vec<f32>], start: &[Vec<f32>], max_iterations: usize) -> (Vec<Vec<f32>>, Vec<usize>, Vec<f32>) {
    todo!("C3: kmeans")
}

/// C4a. The k-means++ probabilities (slide "Better starts: k-means++"): every point's squared
/// distance `D²` to its nearest chosen centre, divided by the sum of all of them.
///
/// Hint: `rng.below(n)` gives a whole number in `0..n`, `rng.uniform()` a number in [0, 1).
/// To pick point `i` with probability `p[i]`: add up the `p[i]` until the sum passes `rng.uniform()`.
pub fn kmeanspp_probabilities(points: &[Vec<f32>], chosen: &[Vec<f32>]) -> Vec<f32> {
    todo!("C4a: kmeanspp_probabilities")
}

/// C4b. k-means++ seeding: the first centre is a point chosen at random; every next centre is
/// a point chosen at random with the probabilities of C4a.
pub fn kmeanspp(points: &[Vec<f32>], k: usize, rng: &mut Rng) -> Vec<Vec<f32>> {
    todo!("C4b: kmeanspp")
}
