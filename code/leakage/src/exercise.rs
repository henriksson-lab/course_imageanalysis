// Coding exercises for "Honest validation: leakage, splits and domain shift". Replace each `todo!()` with your code,
// then check it with `cargo test -p leakage` and look at `cargo run -p leakage`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Image, Rng};

/// C1. k nearest neighbours: the `k` training samples closest to `query` (Euclidean
/// distance of the feature vectors) vote; the majority wins. Use an odd `k`.
///
/// Hint: the simplest correct way: compute all distances into a list of
/// `(distance, label)`, sort it with `list.sort_by(|a, b| a.0.total_cmp(&b.0))`, and count the
/// `true` labels among the first `k`. (Keeping only the `k` best while you go is faster.)
pub fn knn_predict(train_x: &[Vec<f32>], train_y: &[bool], query: &[f32], k: usize) -> bool {
    todo!("C1: knn_predict")
}

/// C2. Random blocks: cut a `w × h` image into `block × block` squares and give every
/// square a random fold `0..folds`. With `block = 1` every pixel gets its own random fold.
///
/// Hint: there are `w.div_ceil(block)` blocks across; draw one fold per block with
/// `rng.below(folds)`, then pixel `(x, y)` is in block `(x / block, y / block)`.
pub fn block_folds(w: usize, h: usize, block: usize, folds: usize, seed: u64) -> Image<usize> {
    todo!("C2: block_folds")
}

/// The role of a pixel in one fold of a cross-validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Train,
    Test,
    Buffer,
}

/// C3. The roles of the pixels when `test_fold` is tested: its pixels are Test; any other
/// pixel within `buffer` pixels of a Test pixel (in x and in y) is Buffer, used for
/// nothing; the rest is Train.
///
/// Hint: `folds.map_xy(|x, y, f| ...)` gives one value per pixel; look at all `(dx, dy)`
/// from `-buffer` to `buffer` and use `folds.contains(..)` near the edge.
pub fn roles(folds: &Image<usize>, test_fold: usize, buffer: usize) -> Image<Role> {
    todo!("C3: roles")
}

/// C4. "Peeking at the test set", simulated: `k` candidate methods all have the true
/// accuracy `accuracy`. In each of `runs` studies, test every candidate on `n` objects
/// (each object is right with probability `accuracy`), report the best score, and re-test
/// that candidate on `n` fresh objects. Returns (mean reported, mean fresh).
///
/// Hint: write a small closure `test(&mut rng)` that draws `n` objects with
/// `rng.uniform() < accuracy` and returns the share that is right; call it `k` times per study.
pub fn best_of_k(k: usize, n: usize, accuracy: f32, runs: usize, seed: u64) -> (f32, f32) {
    todo!("C4: best_of_k")
}
