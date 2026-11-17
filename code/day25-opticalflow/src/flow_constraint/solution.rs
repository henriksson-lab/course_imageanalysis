// Model solution for the coding exercises of "Brightness constancy and the aperture problem".

/// C1. The derivatives of a 1D profile between two frames: `ix[x]` is the central difference
/// `(f1[x+1] − f1[x−1]) / 2` of frame 1 (0 at the two ends), `it[x] = f2[x] − f1[x]`.
pub fn derivatives_1d(f1: &[f32], f2: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = f1.len();
    let mut ix = vec![0.0; n];
    for x in 1..n - 1 {
        ix[x] = (f1[x + 1] - f1[x - 1]) / 2.0;
    }
    let it = (0..n).map(|x| f2[x] - f1[x]).collect();
    (ix, it)
}

/// C2. The speed seen by one pixel, `u = −It / Ix`, or `None` where the slope is (almost) 0
/// (`|Ix| < 1e-6`).
pub fn pixel_speed(ix: f32, it: f32) -> Option<f32> {
    if ix.abs() < 1e-6 {
        None
    } else {
        Some(-it / ix)
    }
}

/// C3. The least-squares speed of a window: `u = −Σ Ix It / Σ Ix²` over the pixels
/// `centre − half ..= centre + half` (those that lie inside the profile). `None` if
/// `Σ Ix² < 1e-6` (a flat window).
pub fn window_speed(ix: &[f32], it: &[f32], centre: usize, half: usize) -> Option<f32> {
    let (mut sxt, mut sxx) = (0.0, 0.0);
    let last = (centre + half).min(ix.len() - 1);
    for x in centre.saturating_sub(half)..=last {
        sxt += ix[x] * it[x];
        sxx += ix[x] * ix[x];
    }
    if sxx < 1e-6 {
        None
    } else {
        Some(-sxt / sxx)
    }
}

/// C4. The normal flow of one pixel: the point of its constraint line `Ix u + Iy v + It = 0`
/// closest to the origin, `−It ∇I / |∇I|²`. `None` if `|∇I|² < 1e-6`.
pub fn normal_flow(ix: f32, iy: f32, it: f32) -> Option<(f32, f32)> {
    let g2 = ix * ix + iy * iy;
    if g2 < 1e-6 {
        None
    } else {
        Some((-it * ix / g2, -it * iy / g2))
    }
}

/// C5. Where the constraint lines of two pixels cross: solve
/// `Ix1 u + Iy1 v = −It1`, `Ix2 u + Iy2 v = −It2` (each pixel given as `[Ix, Iy, It]`).
/// `None` if the gradients are (almost) parallel, `|det| < 1e-6`: the aperture problem.
pub fn intersect(a: [f32; 3], b: [f32; 3]) -> Option<(f32, f32)> {
    let det = a[0] * b[1] - a[1] * b[0];
    if det.abs() < 1e-6 {
        return None;
    }
    // Cramer's rule with the right-hand side (−It1, −It2).
    let u = (-a[2] * b[1] + b[2] * a[1]) / det;
    let v = (-b[2] * a[0] + a[2] * b[0]) / det;
    Some((u, v))
}
