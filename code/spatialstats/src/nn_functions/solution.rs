// Model solution for the coding exercises of "Nearest neighbours and empty space: G, F and J".

pub use crate::common::{csr, dist, hard_core, poisson, thomas, Point};

/// The distance from `p` to the border of the window `[0, a] × [0, b]`. Given.
pub fn border_distance(p: Point, a: f64, b: f64) -> f64 {
    p.0.min(a - p.0).min(p.1).min(b - p.1)
}

/// Test locations on a grid with spacing `step`, the first at `(step/2, step/2)`. Given.
pub fn grid(a: f64, b: f64, step: f64) -> Vec<Point> {
    let (nx, ny) = ((a / step) as usize, (b / step) as usize);
    let mut pts = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            pts.push(((i as f64 + 0.5) * step, (j as f64 + 0.5) * step));
        }
    }
    pts
}

/// C1. For every point, the distance to its nearest neighbour (another point of the pattern).
pub fn nn_distances(points: &[Point]) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; points.len()];
    for i in 0..points.len() {
        for j in 0..points.len() {
            if i != j {
                d[i] = d[i].min(dist(points[i], points[j]));
            }
        }
    }
    d
}

/// C2. The estimate of G(r): the share of points whose nearest neighbour is within `r`.
/// With `border`, only the points at least `r` from the border of the window count (the
/// reduced-sample estimator of the slide "The math: the reduced-sample estimator").
/// Returns NaN if no point is kept.
pub fn g_hat(points: &[Point], a: f64, b: f64, r: f64, border: bool) -> f64 {
    let d = nn_distances(points);
    let (mut kept, mut hits) = (0, 0);
    for (i, &p) in points.iter().enumerate() {
        if border && border_distance(p, a, b) < r {
            continue;
        }
        kept += 1;
        if d[i] <= r {
            hits += 1;
        }
    }
    if kept == 0 { f64::NAN } else { hits as f64 / kept as f64 }
}

/// C3. The estimate of F(r): the share of the test locations whose nearest point is within `r`;
/// with `border`, only the test locations at least `r` from the border count. NaN if none is kept.
pub fn f_hat(points: &[Point], tests: &[Point], a: f64, b: f64, r: f64, border: bool) -> f64 {
    let (mut kept, mut hits) = (0, 0);
    for &u in tests {
        if border && border_distance(u, a, b) < r {
            continue;
        }
        kept += 1;
        let nearest = points.iter().map(|&p| dist(u, p)).fold(f64::INFINITY, f64::min);
        if nearest <= r {
            hits += 1;
        }
    }
    if kept == 0 { f64::NAN } else { hits as f64 / kept as f64 }
}

/// C4. The estimate of J(r) = (1 - G(r)) / (1 - F(r)), both with the border rule, F from test
/// locations on a grid with spacing `step`.
pub fn j_hat(points: &[Point], a: f64, b: f64, r: f64, step: f64) -> f64 {
    let g = g_hat(points, a, b, r, true);
    let f = f_hat(points, &grid(a, b, step), a, b, r, true);
    (1.0 - g) / (1.0 - f)
}
