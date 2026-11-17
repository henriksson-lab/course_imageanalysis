// Model solution for the coding exercises of "The generalised Hough transform".
// Pixel coordinates (x, y), y down. A gradient direction θ is atan2(gy, gx): 0 points right,
// 90° (π/2) points down. A rotation by φ turns (x, y) into (x cos φ − y sin φ, x sin φ + y cos φ)
// and adds φ to every gradient direction (clockwise on the screen, because y points down).

use crate::ght::sim::EdgePoint;
use imgcourse::Image;
use std::f32::consts::TAU;

/// The R-table: for each direction bin, the vectors from edge points to the reference point.
pub type RTable = Vec<Vec<[f32; 2]>>;

/// C1a. The direction bin of a gradient direction `theta` (radians), with `n_bins` bins of
/// width 360°/n_bins centred on 0°, 360°/n_bins, …: bin 0 holds the directions closest to 0°.
pub fn bin_of(theta: f32, n_bins: usize) -> usize {
    let width = TAU / n_bins as f32;
    ((theta / width).round() as isize).rem_euclid(n_bins as isize) as usize
}

/// C1b. Build the R-table of a template: for every template edge point, store the vector
/// `reference − point` in the bin of its gradient direction.
pub fn build_r_table(template: &[EdgePoint], reference: [f32; 2], n_bins: usize) -> RTable {
    let mut table = vec![Vec::new(); n_bins];
    for e in template {
        table[bin_of(e.theta, n_bins)].push([reference[0] - e.x, reference[1] - e.y]);
    }
    table
}

/// C2. Vote with the R-table (translation only): every edge point looks up the bin of its
/// direction and adds one vote at `point + v` for every vector `v` there (rounded to the
/// nearest pixel, if inside the image).
pub fn vote(table: &RTable, edges: &[EdgePoint], width: usize, height: usize) -> Image<f32> {
    vote_pose(table, edges, width, height, 0.0, 1.0)
}

/// C3. Vote for the pose rotated by `phi` and scaled by `scale`: an edge point with direction θ
/// looks up the bin of `θ − phi` and votes at `point + scale · R(phi) v` for every vector `v`
/// there.
pub fn vote_pose(table: &RTable, edges: &[EdgePoint], width: usize, height: usize, phi: f32, scale: f32) -> Image<f32> {
    let mut acc = Image::new(width, height, 0.0);
    let (c, s) = (phi.cos(), phi.sin());
    for e in edges {
        for v in &table[bin_of(e.theta - phi, table.len())] {
            let (rx, ry) = (scale * (c * v[0] - s * v[1]), scale * (s * v[0] + c * v[1]));
            let (x, y) = ((e.x + rx).round() as isize, (e.y + ry).round() as isize);
            if acc.contains(x, y) {
                acc[(x as usize, y as usize)] += 1.0;
            }
        }
    }
    acc
}
