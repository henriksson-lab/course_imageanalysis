// Coding exercises for "Pixel classification with a forest". Replace each `todo!()` with your
// code, then check it with `cargo test -p day33 pixelclass::` and look at `cargo run -p day33 -- pixelclass`.
#![allow(unused_variables, unused_imports)]

use imgcourse::{Border, Image};

/// Gaussian smoothing with scale `sigma`, in two passes (from the filtering unit; filled in,
/// not part of these exercises).
pub fn gaussian(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            let mut sum = 0.0;
            for d in -r..=r {
                let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                sum += w[(d + r) as usize] * src.get_border(xx, yy, Border::Mirror);
            }
            sum / total
        })
    };
    pass(&pass(img, true), false)
}

/// C1. Three features of the pixel `(x, y)`, as in B1: the 3 × 3 box mean; the gradient
/// magnitude from central differences, `g_x = (I(x+1, y) − I(x−1, y)) / 2` and likewise
/// `g_y`; and the 4-neighbour Laplacian (sum of the 4 neighbours − 4 · centre). Outside the
/// image, the edge pixel is copied.
///
/// Hint: `img.get_border(x as isize + dx, y as isize + dy, Border::Clamp)` reads a neighbour.
pub fn pixel_features(img: &Image<f32>, x: usize, y: usize) -> [f32; 3] {
    todo!("C1: pixel_features")
}

/// C2. The feature stack: for every scale σ, three channels of the smoothed image
/// `L = gaussian(img, σ)`: intensity `L`, edges `|∇L|`, and blobs `σ² ∇²L` (the gradient and
/// the Laplacian as in C1, computed on `L`). Returns the channels in the order
/// σ1: L, |∇L|, σ²∇²L, then σ2: ..., and so on.
///
/// Hint: smooth with `gaussian(img, sigma)`, then use C1 on the smoothed image at every pixel
/// for its gradient and Laplacian.
pub fn feature_stack(img: &Image<f32>, sigmas: &[f32]) -> Vec<Image<f32>> {
    todo!("C2: feature_stack")
}

/// C3a. Balancing class weights: `w_k = n / (K n_k)` for K classes with `n_k` painted pixels
/// each (n in all), so that every class has the same total weight.
pub fn class_weights(counts: &[usize]) -> Vec<f32> {
    todo!("C3a: class_weights")
}

/// C3b. The class shares of a leaf with `counts` training pixels per class, each pixel of
/// class k counted `weights[k]` times.
pub fn weighted_shares(counts: &[usize], weights: &[f32]) -> Vec<f32> {
    todo!("C3b: weighted_shares")
}

/// C4a. The confusion matrix: `m[t][p]` = the number of pixels of true class t that were
/// predicted as class p.
pub fn confusion(truth: &[usize], predicted: &[usize], n_classes: usize) -> Vec<Vec<usize>> {
    todo!("C4a: confusion")
}

/// C4b. The Dice coefficient of class k from a confusion matrix: `2 TP / (2 TP + FP + FN)`.
///
/// Hint: FP = the rest of column k (predicted k, truly another class); FN = the rest of row k.
pub fn dice(m: &[Vec<usize>], k: usize) -> f32 {
    todo!("C4b: dice")
}
