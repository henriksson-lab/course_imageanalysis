// Model solution for the coding exercises of "Hough transform: lines".
//
// The accumulator is an image: column j is the angle θ_j = j · 180° / n_theta, row k is the
// distance ρ_k = −rho_max + k · d_rho. Points are given relative to the origin of the line
// parameters (the image centre in the program).

use crate::hough_lines::sim::EdgePoint;
use imgcourse::Image;
use std::f32::consts::PI;

/// The angle of column `j`, in radians.
pub fn theta_of(j: usize, n_theta: usize) -> f32 {
    j as f32 * PI / n_theta as f32
}

/// The distance of row `k`.
pub fn rho_of(k: usize, d_rho: f32, rho_max: f32) -> f32 {
    -rho_max + k as f32 * d_rho
}

/// An empty accumulator: `n_theta` columns, and rows for ρ from −rho_max to rho_max.
pub fn empty_accumulator(n_theta: usize, d_rho: f32, rho_max: f32) -> Image<f32> {
    let n_rho = (2.0 * rho_max / d_rho).round() as usize + 1;
    Image::new(n_theta, n_rho, 0.0)
}

/// C1a. Add one vote at angle column `j` for the line through `p`: compute ρ = x cos θ + y sin θ and
/// add 1 to the nearest row, if it is inside the accumulator.
pub fn vote(acc: &mut Image<f32>, j: usize, p: [f32; 2], d_rho: f32, rho_max: f32) {
    let theta = theta_of(j, acc.width());
    let rho = p[0] * theta.cos() + p[1] * theta.sin();
    let k = ((rho + rho_max) / d_rho).round();
    if k >= 0.0 && (k as usize) < acc.height() {
        acc[(j, k as usize)] += 1.0;
    }
}

/// C1b. The Hough accumulator: every point votes once in every angle column, in the row of its ρ.
pub fn accumulate(points: &[[f32; 2]], n_theta: usize, d_rho: f32, rho_max: f32) -> Image<f32> {
    let mut acc = empty_accumulator(n_theta, d_rho, rho_max);
    for &p in points {
        for j in 0..n_theta {
            vote(&mut acc, j, p, d_rho, rho_max);
        }
    }
    acc
}

/// C2. Gradient voting: each edge point votes only in the `2 window + 1` angle columns around its
/// own gradient direction (taken modulo 180°). Columns beyond either end wrap around to the
/// other end. The positions are given separately (relative to the origin), in the same order.
pub fn accumulate_gradient(points: &[[f32; 2]], edges: &[EdgePoint], n_theta: usize, d_rho: f32, rho_max: f32, window: usize) -> Image<f32> {
    let mut acc = empty_accumulator(n_theta, d_rho, rho_max);
    for (&p, e) in points.iter().zip(edges) {
        let j0 = (e.theta.rem_euclid(PI) / PI * n_theta as f32).round() as isize;
        for dj in -(window as isize)..=window as isize {
            let j = (j0 + dj).rem_euclid(n_theta as isize) as usize;
            vote(&mut acc, j, p, d_rho, rho_max);
        }
    }
    acc
}

/// C3. The peaks: bins with at least `threshold` votes that are the largest in the window of
/// ±`window` bins around them (non-maximum suppression). Of equal neighbours only the first in
/// raster order (row by row) is kept. Across the left and right edge the accumulator wraps
/// around with ρ mirrored: column −1 is column `n_theta − 1` at row `n_rho − 1 − k`.
/// Returns `(j, k, votes)`, strongest first.
pub fn peaks(acc: &Image<f32>, threshold: f32, window: usize) -> Vec<(usize, usize, f32)> {
    let (n_theta, n_rho) = (acc.width() as isize, acc.height() as isize);
    let w = window as isize;
    let mut out = Vec::new();
    for (j, k) in acc.coords() {
        let v = acc[(j, k)];
        if v < threshold {
            continue;
        }
        let mut is_peak = true;
        for dk in -w..=w {
            for dj in -w..=w {
                if (dj, dk) == (0, 0) {
                    continue;
                }
                let (mut jj, mut kk) = (j as isize + dj, k as isize + dk);
                if jj < 0 || jj >= n_theta {
                    jj = jj.rem_euclid(n_theta);
                    kk = n_rho - 1 - kk; // the same line with θ ± 180° has −ρ
                }
                if kk < 0 || kk >= n_rho {
                    continue;
                }
                let u = acc[(jj as usize, kk as usize)];
                let earlier = (kk, jj) < (k as isize, j as isize);
                if u > v || (u == v && earlier) {
                    is_peak = false;
                }
            }
        }
        if is_peak {
            out.push((j, k, v));
        }
    }
    out.sort_by(|a, b| b.2.total_cmp(&a.2));
    out
}

/// C4. The segments of the line (θ, ρ): the points within `tol` of the line, sorted along it,
/// split wherever two consecutive points are more than `gap` apart; pieces shorter than
/// `min_len` are dropped. Returns the end points of each piece.
pub fn segments(points: &[[f32; 2]], theta: f32, rho: f32, tol: f32, gap: f32, min_len: f32) -> Vec<([f32; 2], [f32; 2])> {
    let (c, s) = (theta.cos(), theta.sin());
    // position along the line (direction (−sin θ, cos θ)) of every point near it
    let mut along: Vec<(f32, [f32; 2])> = points
        .iter()
        .filter(|p| (p[0] * c + p[1] * s - rho).abs() <= tol)
        .map(|&p| (-p[0] * s + p[1] * c, p))
        .collect();
    along.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=along.len() {
        if i == along.len() || along[i].0 - along[i - 1].0 > gap {
            if i > start && along[i - 1].0 - along[start].0 >= min_len {
                out.push((along[start].1, along[i - 1].1));
            }
            start = i;
        }
    }
    out
}
