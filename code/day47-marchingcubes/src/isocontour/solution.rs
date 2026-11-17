// Model solution for the coding exercises of "Level sets and marching squares".

use imgcourse::Image;

/// A point (x, y) in grid coordinates: grid point (x, y) is the centre of pixel (x, y).
pub type Point = (f32, f32);

/// A piece of contour from the first point to the second, with the bright side (I > c) on the
/// right when walking along it (x to the right, y down, as on the screen).
pub type Segment = (Point, Point);

/// C1. The case index of a cell: bit k is set if corner k is above c (value > c).
/// Corners: 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.
pub fn case_index(corners: [f32; 4], c: f32) -> usize {
    let mut index = 0;
    for k in 0..4 {
        if corners[k] > c {
            index += 1 << k;
        }
    }
    index
}

/// C2. Where the contour crosses the edge from a value `ia` to a value `ib`: the fraction
/// t = (c − ia) / (ib − ia) of the way from a to b (linear interpolation).
pub fn crossing(ia: f32, ib: f32, c: f32) -> f32 {
    (c - ia) / (ib - ia)
}

/// C3. The asymptotic decider: the value of the bilinear interpolant at its saddle point,
/// f* = (I0 I2 − I1 I3) / (I0 + I2 − I1 − I3), with the corners in the order of C1.
pub fn saddle_value(corners: [f32; 4]) -> f32 {
    let [i0, i1, i2, i3] = corners;
    (i0 * i2 - i1 * i3) / (i0 + i2 - i1 - i3)
}

/// C4. Marching squares: the contour {I = c} of an image as oriented segments, one or two per
/// cell of 2 × 2 pixels. Saddle cells (cases 5 and 10) join the two bright corners if
/// `decider` is true and the saddle value is above c; otherwise they separate them.
pub fn marching_squares(img: &Image<f32>, c: f32, decider: bool) -> Vec<Segment> {
    let mut segments = vec![];
    for y in 0..img.height() - 1 {
        for x in 0..img.width() - 1 {
            let pos = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)];
            let v = pos.map(|(px, py)| img[(px, py)]);
            let above = v.map(|value| value > c);
            // the crossing point on edge k, from corner k to corner k + 1
            let point = |k: usize| -> Point {
                let (a, b) = (pos[k], pos[(k + 1) % 4]);
                let t = crossing(v[k], v[(k + 1) % 4], c);
                (a.0 as f32 + t * (b.0 as f32 - a.0 as f32), a.1 as f32 + t * (b.1 as f32 - a.1 as f32))
            };
            let crossed: Vec<usize> = (0..4).filter(|&k| above[k] != above[(k + 1) % 4]).collect();
            // pairs of crossed edges to connect
            let pairs: Vec<(usize, usize)> = if crossed.len() == 2 {
                vec![(crossed[0], crossed[1])]
            } else if crossed.len() == 4 {
                // saddle: cut off the corners that are above (separate) or below (join);
                // corner k lies between edge k − 1 and edge k
                let join = decider && saddle_value(v) > c;
                (0..4).filter(|&k| above[k] != join).map(|k| ((k + 3) % 4, k)).collect()
            } else {
                vec![]
            };
            for (e1, e2) in pairs {
                let (p, q) = (point(e1), point(e2));
                // orient: corner e1 (the start of edge e1) is on the right of p → q iff the cross
                // product is positive; it must be on the right exactly when it is above c
                let a = (pos[e1].0 as f32, pos[e1].1 as f32);
                let cross = (q.0 - p.0) * (a.1 - p.1) - (q.1 - p.1) * (a.0 - p.0);
                segments.push(if (cross > 0.0) == above[e1] { (p, q) } else { (q, p) });
            }
        }
    }
    segments
}

/// C5a. The total length of the segments.
pub fn contour_length(segments: &[Segment]) -> f32 {
    segments.iter().map(|&((x0, y0), (x1, y1))| ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()).sum()
}

/// C5b. The enclosed area by the shoelace formula, ½ Σ (x_i y_(i+1) − x_(i+1) y_i), summed over
/// the oriented segments: the area above c, with holes counting negative. (The contours must be
/// closed: pad the image with a dark frame.)
pub fn enclosed_area(segments: &[Segment]) -> f32 {
    0.5 * segments.iter().map(|&((x0, y0), (x1, y1))| x0 * y1 - x1 * y0).sum::<f32>()
}
