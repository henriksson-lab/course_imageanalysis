// Coding exercises for "Star-convex polygons". Replace each `todo!()` with your code,
// then check it with `cargo test -p stardist` and look at `cargo run -p stardist`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// The step along a ray, in pixels, when looking for the first point outside the object.
pub const STEP: f32 = 0.01;

/// The direction of ray `k` of `n`: angle 2πk/n, ray 0 to the right. With y pointing down,
/// ray n/4 points down (positive angles turn clockwise on screen).
pub fn direction(k: usize, n: usize) -> (f32, f32) {
    let angle = 2.0 * std::f32::consts::PI * k as f32 / n as f32;
    (angle.cos(), angle.sin())
}

/// The corners of the polygon through the ray tips: tip k is `centre + r_k · direction(k)`.
pub fn polygon_vertices(cx: f32, cy: f32, rays: &[f32]) -> Vec<(f32, f32)> {
    (0..rays.len())
        .map(|k| {
            let (dx, dy) = direction(k, rays.len());
            (cx + rays[k] * dx, cy + rays[k] * dy)
        })
        .collect()
}

/// C1. The `n` rays of the point `(cx, cy)`: along each direction, walk in steps of `STEP`
/// as long as the point is still in the same object (the pixel it falls in, `(x.round(),
/// y.round())`, has the label of the start pixel). The ray length is the last distance
/// that was still inside. Leaving the image counts as leaving the object.
///
/// Hint: compute the distance as `i as f32 * STEP` for i = 1, 2, ... rather than adding
/// `STEP` up, and use `labels.contains(px, py)` before reading a pixel.
pub fn rays(labels: &Image<u32>, cx: f32, cy: f32, n: usize) -> Vec<f32> {
    todo!("C1: rays")
}

/// C2. The area of the polygon through the ray tips, by the triangle rule of the lecture:
/// `A = ½ sin(2π/n) Σ_k r_k r_{k+1}`, with `r_n = r_0`.
pub fn polygon_area(rays: &[f32]) -> f32 {
    todo!("C2: polygon_area")
}

/// C3. Is the point `(px, py)` inside the polygon with these corners? The crossing rule:
/// count the edges that a horizontal half-line from the point to the right crosses; an
/// odd count means inside.
///
/// Hint: the edge from `(x1, y1)` to `(x2, y2)` crosses the line `y = py` if one end is
/// above and the other is not (`(y1 > py) != (y2 > py)`), at `x = x1 + (py − y1)(x2 − x1)/(y2 − y1)`.
pub fn inside_polygon(px: f32, py: f32, vertices: &[(f32, f32)]) -> bool {
    todo!("C3: inside_polygon")
}

/// C4. The IoU of two polygons, counted in pixels: a pixel `(x, y)` (whole numbers) belongs
/// to a polygon if its centre is inside. Only pixels in the box around both polygons need
/// to be counted.
pub fn polygon_iou(a: &[(f32, f32)], b: &[(f32, f32)]) -> f32 {
    todo!("C4: polygon_iou")
}

/// C5. Greedy polygon NMS. Candidates with a score below `prob_threshold` are ignored. The
/// others are visited from the highest score down: a candidate is kept unless its IoU with
/// an already kept candidate is above `nms_threshold`. Returns the kept indices, best first.
/// `iou(i, j)` gives the IoU of candidates `i` and `j`.
///
/// Hint: sort the indices with
/// `order.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap())`.
pub fn nms(scores: &[f32], prob_threshold: f32, nms_threshold: f32, iou: impl Fn(usize, usize) -> f32) -> Vec<usize> {
    todo!("C5: nms")
}
