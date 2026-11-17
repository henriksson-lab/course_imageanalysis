// Model solution for the coding exercises of "Two views: matches, epipolar geometry and triangulation".
// Pixels (u, v), u right, v down; a match is (pixel in image 1, pixel in image 2); a line
// (a, b, c) is the set of pixels with a u + b v + c = 0.

use imgcourse::linalg::sym_eigen;
use imgcourse::Rng;

/// 3-vectors and 3 × 3 matrices (row by row: `m[row][column]`) with `mat_vec`, `mat_mul`,
/// `transpose` and `dot`, in `src/common.rs`.
pub use crate::common::{dot, mat_mul, mat_vec, transpose, Mat3, Vec3};

/// A pair of matched pixels: (in image 1, in image 2).
pub type Match = ([f64; 2], [f64; 2]);

/// The similarity transform T (3 × 3) that moves 2D points to mean 0 and an RMS distance of
/// √2 from the origin (Hartley's normalisation).
pub fn normalising_transform(points: &[[f64; 2]]) -> Mat3 {
    let n = points.len() as f64;
    let (mx, my) = (points.iter().map(|p| p[0]).sum::<f64>() / n, points.iter().map(|p| p[1]).sum::<f64>() / n);
    let rms = (points.iter().map(|p| (p[0] - mx).powi(2) + (p[1] - my).powi(2)).sum::<f64>() / n).sqrt();
    let s = 2f64.sqrt() / rms;
    [[s, 0.0, -s * mx], [0.0, s, -s * my], [0.0, 0.0, 1.0]]
}

/// C1a. The epipolar line in image 2 of the pixel `x` in image 1: `F (u, v, 1)`.
pub fn epipolar_line(f: &Mat3, x: [f64; 2]) -> Vec3 {
    mat_vec(f, [x[0], x[1], 1.0])
}

/// C1b. The distance in pixels of the pixel `x` from the line `l = (a, b, c)`:
/// `|a u + b v + c| / sqrt(a² + b²)`.
pub fn line_distance(l: Vec3, x: [f64; 2]) -> f64 {
    (l[0] * x[0] + l[1] * x[1] + l[2]).abs() / (l[0] * l[0] + l[1] * l[1]).sqrt()
}

/// C2. The normalised eight-point algorithm (slide "The math: the eight-point algorithm"):
/// F from n ≥ 8 matches, with x'ᵀ F x = 0, rank 2, up to scale.
pub fn eight_point(matches: &[Match]) -> Mat3 {
    let t1 = normalising_transform(&matches.iter().map(|m| m.0).collect::<Vec<_>>());
    let t2 = normalising_transform(&matches.iter().map(|m| m.1).collect::<Vec<_>>());
    // one row of A per match, from the normalised pixels
    let mut a: Vec<Vec<f64>> = Vec::new();
    for (x1, x2) in matches {
        let p = mat_vec(&t1, [x1[0], x1[1], 1.0]);
        let q = mat_vec(&t2, [x2[0], x2[1], 1.0]);
        a.push(vec![q[0] * p[0], q[0] * p[1], q[0], q[1] * p[0], q[1] * p[1], q[1], p[0], p[1], 1.0]);
    }
    let ata: Vec<Vec<f64>> = (0..9).map(|i| (0..9).map(|j| a.iter().map(|r| r[i] * r[j]).sum()).collect()).collect();
    let (_, vectors) = sym_eigen(&ata);
    let f = &vectors[8]; // the smallest eigenvalue comes last
    let f_hat: Mat3 = [[f[0], f[1], f[2]], [f[3], f[4], f[5]], [f[6], f[7], f[8]]];
    // rank 2: remove the direction v that F̂ shrinks most (smallest eigenvalue of F̂ᵀF̂):
    // F̂ (I − v vᵀ) sets the smallest singular value to 0 and keeps the others.
    let ftf: Vec<Vec<f64>> = (0..3).map(|i| (0..3).map(|j| (0..3).map(|k| f_hat[k][i] * f_hat[k][j]).sum()).collect()).collect();
    let (_, v) = sym_eigen(&ftf);
    let v = [v[2][0], v[2][1], v[2][2]];
    let proj: Mat3 = [0, 1, 2].map(|i| [0, 1, 2].map(|j| if i == j { 1.0 } else { 0.0 } - v[i] * v[j]));
    let f_rank2 = mat_mul(&f_hat, &proj);
    // undo the normalisation: F = T2ᵀ F̂ T1
    mat_mul(&transpose(&t2), &mat_mul(&f_rank2, &t1))
}

/// C3. The number of RANSAC samples needed for probability `p` of at least one sample of `s`
/// correct matches, with a fraction `w` of correct matches: `log(1 − p) / log(1 − w^s)`,
/// rounded up.
pub fn ransac_iterations(p: f64, w: f64, s: i32) -> usize {
    ((1.0 - p).ln() / (1.0 - w.powi(s)).ln()).ceil() as usize
}

/// C4. RANSAC for F: `iterations` times, draw 8 different matches at random, fit F with
/// `eight_point`, and count the matches whose pixel in image 2 lies within `eps` pixels of
/// the epipolar line of its pixel in image 1. Keep the largest set of inliers, refit F on
/// it, and return that F with the indices of the inliers.
pub fn ransac_fundamental(matches: &[Match], eps: f64, iterations: usize, rng: &mut Rng) -> (Mat3, Vec<usize>) {
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..iterations {
        let mut sample: Vec<usize> = Vec::new();
        while sample.len() < 8 {
            let k = rng.below(matches.len());
            if !sample.contains(&k) {
                sample.push(k);
            }
        }
        let chosen: Vec<Match> = sample.iter().map(|&k| matches[k]).collect();
        let f = eight_point(&chosen);
        let inliers: Vec<usize> = (0..matches.len()).filter(|&k| line_distance(epipolar_line(&f, matches[k].0), matches[k].1) <= eps).collect();
        if inliers.len() > best.len() {
            best = inliers;
        }
    }
    let chosen: Vec<Match> = best.iter().map(|&k| matches[k]).collect();
    (eight_point(&chosen), best)
}

/// C5. Midpoint triangulation (slide "The math: triangulation"): the rays c1 + s d1 and
/// c2 + t d2; find s and t that bring the two points closest (a 2 × 2 linear system), and
/// return the middle of the segment between them.
pub fn triangulate_midpoint(c1: Vec3, d1: Vec3, c2: Vec3, d2: Vec3) -> Vec3 {
    // minimise |c1 + s d1 − c2 − t d2|²: the derivatives in s and t give
    // (d1·d1) s − (d1·d2) t = d1·(c2 − c1),   (d1·d2) s − (d2·d2) t = d2·(c2 − c1)
    let w = [c2[0] - c1[0], c2[1] - c1[1], c2[2] - c1[2]];
    let (a, b, c) = (dot(d1, d1), dot(d1, d2), dot(d2, d2));
    let (e, g) = (dot(d1, w), dot(d2, w));
    let det = -a * c + b * b;
    let s = (-e * c + b * g) / det;
    let t = (a * g - b * e) / det;
    [0, 1, 2].map(|k| 0.5 * (c1[k] + s * d1[k] + c2[k] + t * d2[k]))
}
