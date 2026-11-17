// Model solution for the coding exercises of "Shape features".

use imgcourse::Image;

/// A point (x, y) in pixel coordinates. Pixel (x, y) covers the square [x, x+1] × [y, y+1].
pub type Point = (f32, f32);

/// C1. The candidates for the convex hull of the pixel corners: for every row with object
/// pixels, the two left corners of its first object pixel and the two right corners of its
/// last one.
pub fn hull_candidates(mask: &Image<bool>) -> Vec<Point> {
    let mut points = Vec::new();
    for y in 0..mask.height() {
        let xs: Vec<usize> = (0..mask.width()).filter(|&x| mask[(x, y)]).collect();
        if let (Some(&first), Some(&last)) = (xs.first(), xs.last()) {
            let (x0, x1, y0, y1) = (first as f32, last as f32 + 1.0, y as f32, y as f32 + 1.0);
            points.extend([(x0, y0), (x0, y1), (x1, y0), (x1, y1)]);
        }
    }
    points
}

/// The cross product (a − o) × (b − o): positive if o → a → b turns one way, negative if it
/// turns the other way, 0 if the three points are on a line.
pub fn cross(o: Point, a: Point, b: Point) -> f32 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// C2. The convex hull (Andrew's monotone chain). Sort the points by x (then y); walk left to
/// right keeping only turns with a positive cross product (one half of the hull), then right
/// to left (the other half). Returns the vertices in order, starting at the smallest point,
/// without repeating it and without points in the middle of an edge.
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut p = points.to_vec();
    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let mut hull: Vec<Point> = Vec::new();
    // first half, left to right
    for &q in &p {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0 {
            hull.pop();
        }
        hull.push(q);
    }
    // second half, right to left; never pop into the first half
    let first_half = hull.len() + 1;
    for &q in p.iter().rev().skip(1) {
        while hull.len() >= first_half && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0 {
            hull.pop();
        }
        hull.push(q);
    }
    hull.pop(); // the last point is the first one again
    hull
}

/// C3. The area of a polygon (shoelace formula), positive whatever the order of the vertices.
pub fn polygon_area(poly: &[Point]) -> f32 {
    let n = poly.len();
    let mut sum = 0.0;
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        sum += a.0 * b.1 - b.0 * a.1;
    }
    (sum / 2.0).abs()
}

/// C4. The min and max Feret diameters of a convex polygon: the min is the smallest, over all
/// edges, of the largest distance of a vertex from the line through that edge; the max is the
/// largest distance between two vertices. Returns (min, max).
pub fn feret(hull: &[Point]) -> (f32, f32) {
    let n = hull.len();
    let mut min_width = f32::INFINITY;
    for i in 0..n {
        let (a, b) = (hull[i], hull[(i + 1) % n]);
        let length = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        // distance of p from the line a–b: |cross| / length of the edge
        let farthest = hull.iter().map(|&p| cross(a, b, p).abs() / length).fold(0.0, f32::max);
        min_width = min_width.min(farthest);
    }
    let mut max_dist: f32 = 0.0;
    for &p in hull {
        for &q in hull {
            max_dist = max_dist.max(((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt());
        }
    }
    (min_width, max_dist)
}

/// C5. The exact Euclidean distance transform, by brute force: for every object pixel, the
/// distance between its centre and the nearest background pixel centre. Outside the image
/// counts as background (the nearest such pixel is just across the image edge). 0 on the
/// background.
pub fn distance_transform(mask: &Image<bool>) -> Image<f32> {
    let (w, h) = (mask.width(), mask.height());
    let background: Vec<(usize, usize)> = mask.coords().filter(|&p| !mask[p]).collect();
    mask.map_xy(|x, y, inside| {
        if !inside {
            return 0.0;
        }
        // just outside the image: left, top, right, bottom
        let mut best = (x + 1).min(y + 1).min(w - x).min(h - y) as f32;
        for &(bx, by) in &background {
            let d = ((x as f32 - bx as f32).powi(2) + (y as f32 - by as f32).powi(2)).sqrt();
            best = best.min(d);
        }
        best
    })
}
