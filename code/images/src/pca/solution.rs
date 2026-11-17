// Model solution for the coding exercises of "Principal component analysis".

use imgcourse::Image;

/// C1. The mean image and the covariance matrix of two-pixel images (points), dividing
/// by the number of images `n` as in the lecture.
pub fn mean_and_covariance(points: &[[f32; 2]]) -> ([f32; 2], [[f32; 2]; 2]) {
    let n = points.len() as f32;
    let mut mean = [0.0; 2];
    for p in points {
        mean[0] += p[0] / n;
        mean[1] += p[1] / n;
    }
    let mut s = [[0.0; 2]; 2];
    for p in points {
        let y = [p[0] - mean[0], p[1] - mean[1]];
        for i in 0..2 {
            for j in 0..2 {
                s[i][j] += y[i] * y[j] / n;
            }
        }
    }
    (mean, s)
}

/// C2. Eigenvalues and unit eigenvectors of a symmetric 2 × 2 matrix, largest first:
/// returns `([λ1, λ2], [u1, u2])`. The eigenvectors are the principal components.
pub fn eigen2(s: [[f32; 2]; 2]) -> ([f32; 2], [[f32; 2]; 2]) {
    let (a, b, d) = (s[0][0], s[0][1], s[1][1]);
    // λ solves λ² − (a + d)λ + (ad − b²) = 0
    let mid = (a + d) / 2.0;
    let r = (((a - d) / 2.0).powi(2) + b * b).sqrt();
    let (l1, l2) = (mid + r, mid - r);
    // (S − λ1 I) u = 0 gives u ∝ (b, λ1 − a); if b = 0 the axes are already the PCs.
    let u1 = if b.abs() > 1e-12 {
        let len = (b * b + (l1 - a).powi(2)).sqrt();
        [b / len, (l1 - a) / len]
    } else if a >= d {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let u2 = [-u1[1], u1[0]]; // perpendicular to u1
    ([l1, l2], [u1, u2])
}

/// C3. Project every point onto the line through `mean` with direction `u` (a unit vector):
/// returns the scores `s = u · (x − mean)` and the mean squared error of the reconstructions
/// `mean + s u`.
pub fn project(points: &[[f32; 2]], mean: [f32; 2], u: [f32; 2]) -> (Vec<f32>, f32) {
    let mut scores = Vec::new();
    let mut err = 0.0;
    for p in points {
        let y = [p[0] - mean[0], p[1] - mean[1]];
        let s = u[0] * y[0] + u[1] * y[1];
        let rebuilt = [mean[0] + s * u[0], mean[1] + s * u[1]];
        err += (p[0] - rebuilt[0]).powi(2) + (p[1] - rebuilt[1]).powi(2);
        scores.push(s);
    }
    (scores, err / points.len() as f32)
}

/// C4a. All `size × size` patches of the image whose top-left corners are `step` pixels
/// apart, each flattened row by row into a vector of `size²` numbers.
pub fn patches(img: &Image<f32>, size: usize, step: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for y0 in (0..=img.height() - size).step_by(step) {
        for x0 in (0..=img.width() - size).step_by(step) {
            out.push(img.crop(x0, y0, size, size).pixels().to_vec());
        }
    }
    out
}

/// C4b. The mean vector and covariance matrix of vectors of any length (divide by `n`).
pub fn covariance(vectors: &[Vec<f32>]) -> (Vec<f32>, Vec<Vec<f32>>) {
    let (n, d) = (vectors.len() as f32, vectors[0].len());
    let mut mean = vec![0.0; d];
    for v in vectors {
        for i in 0..d {
            mean[i] += v[i] / n;
        }
    }
    let mut s = vec![vec![0.0; d]; d];
    for v in vectors {
        for i in 0..d {
            for j in 0..d {
                s[i][j] += (v[i] - mean[i]) * (v[j] - mean[j]) / n;
            }
        }
    }
    (mean, s)
}

/// C4c. The largest eigenvalue and its unit eigenvector by power iteration: start from
/// any vector, multiply by `s` and rescale to length 1, `iterations` times.
pub fn power_iteration(s: &[Vec<f32>], iterations: usize) -> (f32, Vec<f32>) {
    let d = s.len();
    let mut u = vec![1.0 / (d as f32).sqrt(); d];
    let mut lambda = 0.0;
    for _ in 0..iterations {
        let v: Vec<f32> = (0..d).map(|i| (0..d).map(|j| s[i][j] * u[j]).sum()).collect();
        lambda = v.iter().map(|x| x * x).sum::<f32>().sqrt(); // |S u| -> λ as u -> the eigenvector
        u = v.iter().map(|x| x / lambda).collect();
    }
    (lambda, u)
}
