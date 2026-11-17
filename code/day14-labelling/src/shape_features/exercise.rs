// Coding exercises for "Shape features". Replace each `todo!()` with your code,
// then check it with `cargo test -p day14 shape_features::` and look at `cargo run -p day14 -- shape-features`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// A point (x, y) in pixel coordinates. Pixel (x, y) covers the square [x, x+1] × [y, y+1].
pub type Point = (f32, f32);

/// C1. The candidates for the convex hull of the pixel corners: for every row with object
/// pixels, the two left corners of its first object pixel and the two right corners of its
/// last one.
///
/// Hint: pixel (x, y) has the corners (x, y), (x, y + 1), (x + 1, y), (x + 1, y + 1).
pub fn hull_candidates(mask: &Image<bool>) -> Vec<Point> {
    todo!("C1: hull_candidates")
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
///
/// Hint: `p.sort_by(|a, b| a.partial_cmp(b).unwrap())` sorts by x, then y. Use the
/// hull as a stack: before pushing a point, pop while the last two points and the new one
/// make a cross product ≤ 0. For the second half, never pop below the length the first half had
/// plus one. Remove the last point at the end: it is the first one again.
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    todo!("C2: convex_hull")
}

/// C3. The area of a polygon (shoelace formula), positive whatever the order of the vertices.
///
/// Hint: sum x_i·y_(i+1) − x_(i+1)·y_i, with i + 1 taken modulo the number of vertices.
pub fn polygon_area(poly: &[Point]) -> f32 {
    todo!("C3: polygon_area")
}

/// C4. The min and max Feret diameters of a convex polygon: the min is the smallest, over all
/// edges, of the largest distance of a vertex from the line through that edge; the max is the
/// largest distance between two vertices. Returns (min, max).
///
/// Hint: the distance of p from the line through a and b is |cross(a, b, p)| / |b − a|.
pub fn feret(hull: &[Point]) -> (f32, f32) {
    todo!("C4: feret")
}

/// C5. The exact Euclidean distance transform, by brute force: for every object pixel, the
/// distance between its centre and the nearest background pixel centre. Outside the image
/// counts as background (the nearest such pixel is just across the image edge). 0 on the
/// background.
///
/// Hint: first collect the background pixels in a list; `mask.map_xy` builds the result.
pub fn distance_transform(mask: &Image<bool>) -> Image<f32> {
    todo!("C5: distance_transform")
}
