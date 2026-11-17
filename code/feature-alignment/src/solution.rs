// Model solution for the coding exercises of "From matches to a transform".

use imgcourse::{linalg, Complex, Rng};

/// A point `(x, y)`.
pub type Point = (f32, f32);

/// A matched pair: a point of image 1 and its match in image 2.
pub type Pair = (Point, Point);

/// An affine transform, the top two rows of the 3 × 3 matrix: `x' = m[0][0] x + m[0][1] y + m[0][2]`,
/// `y' = m[1][0] x + m[1][1] y + m[1][2]`. Translations and similarities are special cases.
pub type Affine = [[f32; 3]; 2];

/// Not an exercise: apply a transform to a point.
pub fn apply(m: &Affine, (x, y): Point) -> Point {
    (m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2])
}

/// C1. The least-squares translation: the mean of the differences `q − p` over all pairs.
pub fn fit_translation(pairs: &[Pair]) -> Point {
    let (mut tx, mut ty) = (0.0, 0.0);
    for &((x, y), (u, v)) in pairs {
        tx += u - x;
        ty += v - y;
    }
    (tx / pairs.len() as f32, ty / pairs.len() as f32)
}

/// C2. The similarity (rotation, scale, translation) that maps `p1 → q1` and `p2 → q2`
/// exactly. In complex numbers a similarity is `z ↦ a z + b`: `a = (q2 − q1) / (p2 − p1)`,
/// `b = q1 − a p1`. As a matrix: `[[Re a, −Im a, Re b], [Im a, Re a, Im b]]`.
pub fn similarity_from_two(p1: Point, q1: Point, p2: Point, q2: Point) -> Affine {
    let z = |(x, y): Point| Complex::new(x, y);
    // Dividing by a complex number d is multiplying by conj(d) / |d|².
    let d = z(p2) - z(p1);
    let a = (z(q2) - z(q1)) * d.conj() * (1.0 / d.norm_sqr());
    let b = z(q1) - a * z(p1);
    [[a.re, -a.im, b.re], [a.im, a.re, b.im]]
}

/// C3. The least-squares affine transform of three or more pairs (exact for three). Each
/// row of the matrix is its own small problem with the same 3 × 3 normal equations
/// `BᵀB m = Bᵀx'` (rows of `B`: `[x, y, 1]`); solve them with `linalg::solve`.
/// Returns `None` if the points of image 1 lie on a line.
pub fn fit_affine(pairs: &[Pair]) -> Option<Affine> {
    let mut btb = vec![vec![0.0f64; 3]; 3];
    let mut rhs = [vec![0.0f64; 3], vec![0.0f64; 3]];
    for &((x, y), (u, v)) in pairs {
        let row = [x as f64, y as f64, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                btb[i][j] += row[i] * row[j];
            }
            rhs[0][i] += row[i] * u as f64;
            rhs[1][i] += row[i] * v as f64;
        }
    }
    let first = linalg::solve(&btb, &rhs[0])?;
    let second = linalg::solve(&btb, &rhs[1])?;
    Some([[first[0] as f32, first[1] as f32, first[2] as f32], [second[0] as f32, second[1] as f32, second[2] as f32]])
}

/// C4a. The inliers of a transform: the indices of the pairs whose point of image 1 is
/// mapped to within `eps` px of its match.
pub fn inliers(m: &Affine, pairs: &[Pair], eps: f32) -> Vec<usize> {
    let mut found = Vec::new();
    for (i, &(p, q)) in pairs.iter().enumerate() {
        let (u, v) = apply(m, p);
        if (u - q.0).hypot(v - q.1) < eps {
            found.push(i);
        }
    }
    found
}

/// C4b. The number of RANSAC iterations for inlier fraction `w`, sample size `s` and
/// success probability `p`: `N = log(1 − p) / log(1 − w^s)`, rounded up.
pub fn ransac_iterations(w: f32, s: u32, p: f32) -> usize {
    ((1.0 - p).ln() / (1.0 - w.powi(s as i32)).ln()).ceil() as usize
}

/// C5. RANSAC for an affine transform: `iterations` times, pick 3 different pairs at random,
/// fit them exactly (`fit_affine`), count the inliers; keep the transform with the most.
/// Then refit by least squares on its inliers and count again. Returns the transform and
/// its inliers, or `None` if no sample could be fitted.
pub fn ransac_affine(pairs: &[Pair], eps: f32, iterations: usize, rng: &mut Rng) -> Option<(Affine, Vec<usize>)> {
    let mut best: Option<(Affine, Vec<usize>)> = None;
    for _ in 0..iterations {
        let i = rng.below(pairs.len());
        let j = rng.below(pairs.len());
        let k = rng.below(pairs.len());
        if i == j || j == k || i == k {
            continue;
        }
        let Some(m) = fit_affine(&[pairs[i], pairs[j], pairs[k]]) else { continue };
        let found = inliers(&m, pairs, eps);
        if best.as_ref().is_none_or(|(_, b)| found.len() > b.len()) {
            best = Some((m, found));
        }
    }
    let (m, found) = best?;
    // Refit on all inliers of the best sample.
    let chosen: Vec<Pair> = found.iter().map(|&i| pairs[i]).collect();
    let refit = fit_affine(&chosen).unwrap_or(m);
    Some((refit, inliers(&refit, pairs, eps)))
}
