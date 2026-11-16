// Coding exercises for "Principal component analysis". Replace each `todo!()` with your code,
// then check it with `cargo test -p pca` and look at `cargo run -p pca`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// C1. The mean image and the covariance matrix of two-pixel images (points), dividing
/// by the number of images `n` as in the lecture.
pub fn mean_and_covariance(points: &[[f32; 2]]) -> ([f32; 2], [[f32; 2]; 2]) {
    todo!("C1: mean_and_covariance")
}

/// C2. Eigenvalues and unit eigenvectors of a symmetric 2 × 2 matrix, largest first:
/// returns `([λ1, λ2], [u1, u2])`. The eigenvectors are the principal components.
///
/// Hint: the eigenvalues solve λ² − (a + d)λ + (ad − b²) = 0 for `s = [[a, b], [b, d]]`.
/// An eigenvector for λ is `(b, λ − a)`, scaled to length 1 (careful when b = 0).
pub fn eigen2(s: [[f32; 2]; 2]) -> ([f32; 2], [[f32; 2]; 2]) {
    todo!("C2: eigen2")
}

/// C3. Project every point onto the line through `mean` with direction `u` (a unit vector):
/// returns the scores `s = u · (x − mean)` and the mean squared error of the reconstructions
/// `mean + s u`.
///
/// Hint: the error of one point is its squared distance to its reconstruction.
pub fn project(points: &[[f32; 2]], mean: [f32; 2], u: [f32; 2]) -> (Vec<f32>, f32) {
    todo!("C3: project")
}

/// C4a. All `size × size` patches of the image whose top-left corners are `step` pixels
/// apart, each flattened row by row into a vector of `size²` numbers.
///
/// Hint: `img.crop(x0, y0, size, size).pixels().to_vec()` is one patch.
pub fn patches(img: &Image<f32>, size: usize, step: usize) -> Vec<Vec<f32>> {
    todo!("C4a: patches")
}

/// C4b. The mean vector and covariance matrix of vectors of any length (divide by `n`).
pub fn covariance(vectors: &[Vec<f32>]) -> (Vec<f32>, Vec<Vec<f32>>) {
    todo!("C4b: covariance")
}

/// C4c. The largest eigenvalue and its unit eigenvector by power iteration: start from
/// any vector, multiply by `s` and rescale to length 1, `iterations` times.
pub fn power_iteration(s: &[Vec<f32>], iterations: usize) -> (f32, Vec<f32>) {
    todo!("C4c: power_iteration")
}
