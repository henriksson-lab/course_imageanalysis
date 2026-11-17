// Coding exercises for "Level sets and marching squares". Replace each `todo!()` with your code,
// then check it with `cargo test -p day47 isocontour::` and look at `cargo run -p day47 -- isocontour`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Image;

/// A point (x, y) in grid coordinates: grid point (x, y) is the centre of pixel (x, y).
pub type Point = (f32, f32);

/// A piece of contour from the first point to the second, with the bright side (I > c) on the
/// right when walking along it (x to the right, y down, as on the screen).
pub type Segment = (Point, Point);

/// C1. The case index of a cell: bit k is set if corner k is above c (value > c).
/// Corners: 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.
///
/// Hint: `1 << k` is 2^k.
pub fn case_index(corners: [f32; 4], c: f32) -> usize {
    todo!("C1: case_index")
}

/// C2. Where the contour crosses the edge from a value `ia` to a value `ib`: the fraction
/// t = (c − ia) / (ib − ia) of the way from a to b (linear interpolation).
pub fn crossing(ia: f32, ib: f32, c: f32) -> f32 {
    todo!("C2: crossing")
}

/// C3. The asymptotic decider: the value of the bilinear interpolant at its saddle point,
/// f* = (I0 I2 − I1 I3) / (I0 + I2 − I1 − I3), with the corners in the order of C1.
pub fn saddle_value(corners: [f32; 4]) -> f32 {
    todo!("C3: saddle_value")
}

/// C4. Marching squares: the contour {I = c} of an image as oriented segments, one or two per
/// cell of 2 × 2 pixels. Saddle cells (cases 5 and 10) join the two bright corners if
/// `decider` is true and the saddle value is above c; otherwise they separate them.
///
/// Hint: for the cell with top-left corner (x, y), list the corner positions
/// [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)] and their values. Edge k runs from corner k to
/// corner (k + 1) % 4; it is crossed if exactly one of its corners is above c, at the point
/// a + t (b − a) with t from `crossing`. Two crossed edges: one segment. Four (a saddle): corner k
/// lies between edges (k + 3) % 4 and k; "separate" cuts off each corner above c with a segment
/// between its two edges, "join" each corner below c.
/// Orientation: for a segment p → q that starts on edge e, the corner a = corner e is on the right
/// if (q.0 − p.0)(a.1 − p.1) − (q.1 − p.1)(a.0 − p.0) > 0. It must be on the right exactly when it
/// is above c; otherwise store the segment as (q, p).
pub fn marching_squares(img: &Image<f32>, c: f32, decider: bool) -> Vec<Segment> {
    todo!("C4: marching_squares")
}

/// C5a. The total length of the segments.
pub fn contour_length(segments: &[Segment]) -> f32 {
    todo!("C5: contour_length")
}

/// C5b. The enclosed area by the shoelace formula, ½ Σ (x_i y_(i+1) − x_(i+1) y_i), summed over
/// the oriented segments: the area above c, with holes counting negative. (The contours must be
/// closed: pad the image with a dark frame.)
pub fn enclosed_area(segments: &[Segment]) -> f32 {
    todo!("C5: enclosed_area")
}
