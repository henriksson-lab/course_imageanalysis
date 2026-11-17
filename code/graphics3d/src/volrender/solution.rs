// Model solution for the coding exercises of "Seeing voxels: projections and volume rendering".
// The tasks are called R1, R2, ... because the exercise page already uses C1, C2, ... for the
// demo questions. Voxel (i, j, k) has its centre at the voxel coordinates (i, j, k); in the
// world (millimetres) it sits at (i Δx, j Δy, k Δz).

use imgcourse::Volume;

/// A colour, red, green and blue in [0, 1].
pub type Rgb = [f32; 3];

/// R1. Trilinear interpolation at the voxel coordinates `(x, y, z)`: the weighted mean of the
/// 8 voxels around the point, with weights from the distances along each axis (as bilinear
/// interpolation in the images unit, one dimension more). Voxels outside the volume count as 0.
pub fn trilinear(vol: &Volume<f32>, x: f32, y: f32, z: f32) -> f32 {
    let (x0, y0, z0) = (x.floor(), y.floor(), z.floor());
    let (tx, ty, tz) = (x - x0, y - y0, z - z0);
    let mut sum = 0.0;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let (i, j, k) = (x0 as isize + dx, y0 as isize + dy, z0 as isize + dz);
                if !vol.contains(i, j, k) {
                    continue;
                }
                let wx = if dx == 0 { 1.0 - tx } else { tx };
                let wy = if dy == 0 { 1.0 - ty } else { ty };
                let wz = if dz == 0 { 1.0 - tz } else { tz };
                sum += wx * wy * wz * vol[(i as usize, j as usize, k as usize)];
            }
        }
    }
    sum
}

/// R2. The values along a ray: `n` samples at the world points `start + s · step · dir`,
/// s = 0, 1, ..., n − 1 (`dir` has length 1, all in millimetres). Each point is turned into
/// voxel coordinates by dividing by the voxel `spacing`, then sampled with `trilinear`.
pub fn sample_ray(vol: &Volume<f32>, spacing: [f32; 3], start: [f32; 3], dir: [f32; 3], step: f32, n: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(n);
    for s in 0..n {
        let t = s as f32 * step;
        let p = [start[0] + t * dir[0], start[1] + t * dir[1], start[2] + t * dir[2]];
        values.push(trilinear(vol, p[0] / spacing[0], p[1] / spacing[1], p[2] / spacing[2]));
    }
    values
}

/// R3a. Front-to-back compositing (slide "From the integral to compositing"): start with
/// C = 0 and T = 1; for each sample (colour c, opacity α), from the eye outwards,
/// C ← C + T α c and T ← T (1 − α). Stop early once T < `eps` (use `eps = 0` for no early
/// stop). Returns the colour, the transmittance left, and the number of samples used.
pub fn front_to_back(samples: &[(Rgb, f32)], eps: f32) -> (Rgb, f32, usize) {
    let mut colour = [0.0; 3];
    let mut t = 1.0;
    let mut used = 0;
    for &(c, alpha) in samples {
        if t < eps {
            break;
        }
        for k in 0..3 {
            colour[k] += t * alpha * c[k];
        }
        t *= 1.0 - alpha;
        used += 1;
    }
    (colour, t, used)
}

/// R3b. Back-to-front compositing (the painter's order): start with C = 0 at the far end and
/// walk towards the eye, C ← α c + (1 − α) C.
pub fn back_to_front(samples: &[(Rgb, f32)]) -> Rgb {
    let mut colour = [0.0; 3];
    for &(c, alpha) in samples.iter().rev() {
        for k in 0..3 {
            colour[k] = alpha * c[k] + (1.0 - alpha) * colour[k];
        }
    }
    colour
}

/// R4. Opacity correction: an opacity `alpha0` meant for the reference step, used with a
/// step `ratio` times as long: `1 − (1 − alpha0)^ratio`.
pub fn opacity_correct(alpha0: f32, ratio: f32) -> f32 {
    1.0 - (1.0 - alpha0).powf(ratio)
}

/// R5. The first place along a ray where the values cross the isovalue `c` from below:
/// for the first pair of neighbouring samples with `v[i] < c ≤ v[i + 1]`, the position
/// `i + (c − v[i]) / (v[i + 1] − v[i])` (in samples), by linear interpolation. `None` if the
/// ray never crosses `c`.
pub fn iso_crossing(values: &[f32], c: f32) -> Option<f32> {
    for i in 0..values.len().saturating_sub(1) {
        let (a, b) = (values[i], values[i + 1]);
        if a < c && b >= c {
            return Some(i as f32 + (c - a) / (b - a));
        }
    }
    None
}
