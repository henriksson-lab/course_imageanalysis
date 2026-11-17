// Model solution for the coding exercises of "Star-convex polygons".

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
pub fn rays(labels: &Image<u32>, cx: f32, cy: f32, n: usize) -> Vec<f32> {
    let own = labels[(cx.round() as usize, cy.round() as usize)];
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let (dx, dy) = direction(k, n);
        let mut i = 1;
        loop {
            let r = i as f32 * STEP;
            let (px, py) = ((cx + r * dx).round() as isize, (cy + r * dy).round() as isize);
            if !labels.contains(px, py) || labels[(px as usize, py as usize)] != own {
                break;
            }
            i += 1;
        }
        out.push((i - 1) as f32 * STEP); // the last step that was inside
    }
    out
}

/// C2. The area of the polygon through the ray tips, by the triangle rule of the lecture:
/// `A = ½ sin(2π/n) Σ_k r_k r_{k+1}`, with `r_n = r_0`.
pub fn polygon_area(rays: &[f32]) -> f32 {
    let n = rays.len();
    let mut sum = 0.0;
    for k in 0..n {
        sum += rays[k] * rays[(k + 1) % n];
    }
    0.5 * (2.0 * std::f32::consts::PI / n as f32).sin() * sum
}

/// C3. Is the point `(px, py)` inside the polygon with these corners? The crossing rule:
/// count the edges that a horizontal half-line from the point to the right crosses; an
/// odd count means inside.
pub fn inside_polygon(px: f32, py: f32, vertices: &[(f32, f32)]) -> bool {
    let n = vertices.len();
    let mut inside = false;
    for k in 0..n {
        let (x1, y1) = vertices[k];
        let (x2, y2) = vertices[(k + 1) % n];
        if (y1 > py) != (y2 > py) {
            let x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1);
            if x_cross > px {
                inside = !inside;
            }
        }
    }
    inside
}

/// C4. The IoU of two polygons, counted in pixels: a pixel `(x, y)` (whole numbers) belongs
/// to a polygon if its centre is inside. Only pixels in the box around both polygons need
/// to be counted.
pub fn polygon_iou(a: &[(f32, f32)], b: &[(f32, f32)]) -> f32 {
    let all = a.iter().chain(b);
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &(x, y) in all {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let (mut both, mut either) = (0, 0);
    for y in y0.floor() as i32..=y1.ceil() as i32 {
        for x in x0.floor() as i32..=x1.ceil() as i32 {
            let in_a = inside_polygon(x as f32, y as f32, a);
            let in_b = inside_polygon(x as f32, y as f32, b);
            if in_a && in_b {
                both += 1;
            }
            if in_a || in_b {
                either += 1;
            }
        }
    }
    if either == 0 { 0.0 } else { both as f32 / either as f32 }
}

/// C5. Greedy polygon NMS. Candidates with a score below `prob_threshold` are ignored. The
/// others are visited from the highest score down: a candidate is kept unless its IoU with
/// an already kept candidate is above `nms_threshold`. Returns the kept indices, best first.
/// `iou(i, j)` gives the IoU of candidates `i` and `j`.
pub fn nms(scores: &[f32], prob_threshold: f32, nms_threshold: f32, iou: impl Fn(usize, usize) -> f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).filter(|&i| scores[i] >= prob_threshold).collect();
    order.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());
    let mut kept: Vec<usize> = Vec::new();
    for i in order {
        if kept.iter().all(|&k| iou(k, i) <= nms_threshold) {
            kept.push(i);
        }
    }
    kept
}
