//! The projector of this unit as a sparse matrix (not an exercise). The rest of the tomography
//! plumbing (coordinates, phantom, projector, back-projection) is in `src/common.rs`.

use crate::common::centre;

/// One row of the matrix A: the pixels a ray touches, as (pixel index y·n + x, weight).
pub type Row = Vec<(usize, f32)>;

/// The projector as a sparse matrix A, one row per ray (angle by angle, bin by bin): the same
/// rays and weights as `projection`, so that `A x` is the sinogram of the image `x` (pixels
/// row by row). The weights are those of `sample` (bilinear), added up per pixel.
pub fn ray_matrix(n: usize, thetas: &[f32]) -> Vec<Row> {
    let c = centre(n);
    let mut rows = Vec::new();
    for &theta in thetas {
        let (cos, sin) = (theta.cos(), theta.sin());
        for k in 0..n {
            let r = k as f32 - c;
            let mut weights: Vec<f32> = Vec::new();
            let mut pixels: Vec<usize> = Vec::new();
            for j in 0..n {
                let s = j as f32 - c;
                let (x, y) = (r * cos - s * sin, r * sin + s * cos);
                let (fx, fy) = (x + c, c - y);
                let (x0, y0) = (fx.floor(), fy.floor());
                let (tx, ty) = (fx - x0, fy - y0);
                for (dx, dy, w) in [(0.0, 0.0, (1.0 - tx) * (1.0 - ty)), (1.0, 0.0, tx * (1.0 - ty)), (0.0, 1.0, (1.0 - tx) * ty), (1.0, 1.0, tx * ty)] {
                    let (px, py) = (x0 + dx, y0 + dy);
                    if px >= 0.0 && py >= 0.0 && (px as usize) < n && (py as usize) < n && w > 0.0 {
                        pixels.push(py as usize * n + px as usize);
                        weights.push(w);
                    }
                }
            }
            // add up the weights of the same pixel
            let mut order: Vec<usize> = (0..pixels.len()).collect();
            order.sort_by_key(|&i| pixels[i]);
            let mut row: Row = Vec::new();
            for i in order {
                match row.last_mut() {
                    Some(last) if last.0 == pixels[i] => last.1 += weights[i],
                    _ => row.push((pixels[i], weights[i])),
                }
            }
            rows.push(row);
        }
    }
    rows
}
