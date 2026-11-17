//! Helpers shared by the units of the day.

use imgcourse::Volume;

/// A Gaussian blur of width `sigma` (voxels) along all three axes (zero outside).
pub fn blur3(vol: &Volume<f32>, sigma: f32) -> Volume<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = w.iter().sum();
    let mut out = vol.clone();
    for axis in 0..3 {
        let src = out.clone();
        out = Volume::from_fn(vol.width(), vol.height(), vol.depth(), |x, y, z| {
            let mut sum = 0.0;
            for d in -r..=r {
                let mut p = [x as isize, y as isize, z as isize];
                p[axis] += d;
                if src.contains(p[0], p[1], p[2]) {
                    sum += w[(d + r) as usize] / s * src[(p[0] as usize, p[1] as usize, p[2] as usize)];
                }
            }
            sum
        });
    }
    out
}
