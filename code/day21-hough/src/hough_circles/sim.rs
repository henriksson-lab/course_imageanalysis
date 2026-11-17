//! Ellipses and circle drawing for the circle exercises (test scenes, edge points and drawing
//! are in `src/common.rs`). Not part of the exercise: read it if you like.

use imgcourse::Image;

/// An ellipse (or a circle, `rx = ry`) as a polygon of 72 corners, turned by `angle` radians.
pub fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32, angle: f32) -> Vec<[f32; 2]> {
    (0..72)
        .map(|i| {
            let t = i as f32 * std::f32::consts::TAU / 72.0;
            let (u, v) = (rx * t.cos(), ry * t.sin());
            [cx + u * angle.cos() - v * angle.sin(), cy + u * angle.sin() + v * angle.cos()]
        })
        .collect()
}

/// Draw a circle outline in colour `c`.
pub fn draw_circle(img: &mut Image<[f32; 3]>, cx: f32, cy: f32, r: f32, c: [f32; 3]) {
    let n = (8.0 * r).max(16.0) as usize;
    for i in 0..n {
        let t = i as f32 * std::f32::consts::TAU / n as f32;
        let (x, y) = ((cx + r * t.cos()).round() as isize, (cy + r * t.sin()).round() as isize);
        if img.contains(x, y) {
            img[(x as usize, y as usize)] = c;
        }
    }
}
