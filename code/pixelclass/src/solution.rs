// Model solution for the coding exercises of "Pixel classification with a forest".

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
pub fn pixel_features(img: &Image<f32>, x: usize, y: usize) -> [f32; 3] {
    let (x, y) = (x as isize, y as isize);
    let i = |dx: isize, dy: isize| img.get_border(x + dx, y + dy, Border::Clamp);
    let mut sum = 0.0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            sum += i(dx, dy);
        }
    }
    let gx = (i(1, 0) - i(-1, 0)) / 2.0;
    let gy = (i(0, 1) - i(0, -1)) / 2.0;
    let laplacian = i(1, 0) + i(-1, 0) + i(0, 1) + i(0, -1) - 4.0 * i(0, 0);
    [sum / 9.0, (gx * gx + gy * gy).sqrt(), laplacian]
}

/// C2. The feature stack: for every scale σ, three channels of the smoothed image
/// `L = gaussian(img, σ)`: intensity `L`, edges `|∇L|`, and blobs `σ² ∇²L` (the gradient and
/// the Laplacian as in C1, computed on `L`). Returns the channels in the order
/// σ1: L, |∇L|, σ²∇²L, then σ2: ..., and so on.
pub fn feature_stack(img: &Image<f32>, sigmas: &[f32]) -> Vec<Image<f32>> {
    let mut channels = Vec::new();
    for &sigma in sigmas {
        let l = gaussian(img, sigma);
        let mut edges = Image::new(img.width(), img.height(), 0.0);
        let mut blobs = Image::new(img.width(), img.height(), 0.0);
        for (x, y) in l.coords() {
            let [_, g, lap] = pixel_features(&l, x, y);
            edges[(x, y)] = g;
            blobs[(x, y)] = sigma * sigma * lap;
        }
        channels.push(l);
        channels.push(edges);
        channels.push(blobs);
    }
    channels
}

/// C3a. Balancing class weights: `w_k = n / (K n_k)` for K classes with `n_k` painted pixels
/// each (n in all), so that every class has the same total weight.
pub fn class_weights(counts: &[usize]) -> Vec<f32> {
    let n: usize = counts.iter().sum();
    let k = counts.len() as f32;
    counts.iter().map(|&c| n as f32 / (k * c as f32)).collect()
}

/// C3b. The class shares of a leaf with `counts` training pixels per class, each pixel of
/// class k counted `weights[k]` times.
pub fn weighted_shares(counts: &[usize], weights: &[f32]) -> Vec<f32> {
    let total: f32 = counts.iter().zip(weights).map(|(&c, &w)| c as f32 * w).sum();
    counts.iter().zip(weights).map(|(&c, &w)| c as f32 * w / total).collect()
}

/// C4a. The confusion matrix: `m[t][p]` = the number of pixels of true class t that were
/// predicted as class p.
pub fn confusion(truth: &[usize], predicted: &[usize], n_classes: usize) -> Vec<Vec<usize>> {
    let mut m = vec![vec![0; n_classes]; n_classes];
    for (&t, &p) in truth.iter().zip(predicted) {
        m[t][p] += 1;
    }
    m
}

/// C4b. The Dice coefficient of class k from a confusion matrix: `2 TP / (2 TP + FP + FN)`.
pub fn dice(m: &[Vec<usize>], k: usize) -> f32 {
    let tp = m[k][k];
    let fp: usize = (0..m.len()).filter(|&t| t != k).map(|t| m[t][k]).sum(); // predicted k, truly not
    let fn_: usize = (0..m.len()).filter(|&p| p != k).map(|p| m[k][p]).sum(); // truly k, predicted not
    2.0 * tp as f32 / (2 * tp + fp + fn_) as f32
}
