//! Helpers shared by the units of day 26: bilinear interpolation, a Gaussian blur and the flow
//! colour wheel (given, not part of the exercises).

use imgcourse::{Border, Image};

/// Not an exercise: bilinear interpolation, the edge copied outside the image.
pub fn bilinear(img: &Image<f32>, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let p = |i: f32, j: f32| img.get_border(i as isize, j as isize, Border::Clamp);
    (1.0 - fy) * ((1.0 - fx) * p(x0, y0) + fx * p(x0 + 1.0, y0)) + fy * ((1.0 - fx) * p(x0, y0 + 1.0) + fx * p(x0 + 1.0, y0 + 1.0))
}

/// A Gaussian blur: two passes of a sampled, normalised 1D Gaussian of radius 3σ, mirror border.
pub fn blur(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let w: Vec<f32> = (-r..=r).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let total: f32 = w.iter().sum();
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            (-r..=r)
                .map(|d| {
                    let (xx, yy) = if along_x { (x as isize + d, y as isize) } else { (x as isize, y as isize + d) };
                    w[(d + r) as usize] * src.get_border(xx, yy, Border::Mirror)
                })
                .sum::<f32>()
                / total
        })
    };
    pass(&pass(img, true), false)
}

/// The colour wheel: hue = direction (red = right, counter-clockwise on screen), saturation =
/// speed / `max`, white = still.
pub fn flow_colour(u: f32, v: f32, max: f32) -> [f32; 3] {
    let angle = (-v).atan2(u).to_degrees().rem_euclid(360.0);
    let s = (u.hypot(v) / max).min(1.0);
    // HSV with value 1: hue from the angle, then mix with white by the saturation.
    let h = angle / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as usize {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [1.0 - s + s * r, 1.0 - s + s * g, 1.0 - s + s * b]
}
