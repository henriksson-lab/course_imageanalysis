// Model solution for the coding exercises of "Sub-pixel localisation".

use imgcourse::Image;

/// C1. The centroid of a patch after subtracting `background` from every pixel:
/// `x̂ = Σ x (I − b) / Σ (I − b)`, and the same for `y`. Returns `[x̂, ŷ]` in the patch's
/// pixel coordinates.
pub fn centroid(patch: &Image<f32>, background: f32) -> [f32; 2] {
    let (mut sx, mut sy, mut sum) = (0.0, 0.0, 0.0);
    for (x, y) in patch.coords() {
        let v = patch[(x, y)] - background;
        sx += x as f32 * v;
        sy += y as f32 * v;
        sum += v;
    }
    [sx / sum, sy / sum]
}

/// C2a. The vertex of the parabola through `(−1, cm)`, `(0, c0)`, `(1, cp)`: the offset of the
/// peak from the middle sample.
pub fn parabola_vertex(cm: f32, c0: f32, cp: f32) -> f32 {
    let denom = 2.0 * (cm - 2.0 * c0 + cp);
    if denom == 0.0 {
        return 0.0; // three equal values: no peak, stay in the middle
    }
    (cm - cp) / denom
}

/// C2b. The same through the logarithms of the three values: exact for a Gaussian.
/// Values that are not positive are raised to a tiny positive number first.
pub fn log_parabola_vertex(cm: f32, c0: f32, cp: f32) -> f32 {
    let ln = |c: f32| c.max(1e-6).ln();
    parabola_vertex(ln(cm), ln(c0), ln(cp))
}

/// C2c. The three-point estimate in 2D: find the brightest pixel, subtract `background` from it
/// and its four neighbours, and apply the (log-)parabola along x and along y. At the patch edge
/// the brightest pixel itself is returned.
pub fn three_point(patch: &Image<f32>, background: f32, log: bool) -> [f32; 2] {
    let (bx, by) = patch.coords().max_by(|&a, &b| patch[a].total_cmp(&patch[b])).unwrap();
    if bx == 0 || by == 0 || bx + 1 == patch.width() || by + 1 == patch.height() {
        return [bx as f32, by as f32];
    }
    let v = |x: usize, y: usize| patch[(x, y)] - background;
    let vertex = if log { log_parabola_vertex } else { parabola_vertex };
    let dx = vertex(v(bx - 1, by), v(bx, by), v(bx + 1, by));
    let dy = vertex(v(bx, by - 1), v(bx, by), v(bx, by + 1));
    [bx as f32 + dx, by as f32 + dy]
}

/// The spot shape of peak 1 at pixel `(x, y)`: `exp(−((x − x0)² + (y − y0)²) / 2s²)`.
pub fn shape(x: usize, y: usize, x0: f32, y0: f32, s: f32) -> f32 {
    let d2 = (x as f32 - x0).powi(2) + (y as f32 - y0).powi(2);
    (-d2 / (2.0 * s * s)).exp()
}

/// C3. With the position `(x0, y0)` and width `s` known, the amplitude `A` and background `B` of
/// the model `B + A · shape` that fit the patch best (least squares): solve the 2 × 2 normal
/// equations `[Σg² Σg; Σg n] [A; B] = [ΣgI; ΣI]`. Returns `(A, B)`.
pub fn fit_amplitude_background(patch: &Image<f32>, x0: f32, y0: f32, s: f32) -> (f32, f32) {
    let (mut sgg, mut sg, mut sgi, mut si) = (0.0, 0.0, 0.0, 0.0);
    for (x, y) in patch.coords() {
        let g = shape(x, y, x0, y0, s);
        let i = patch[(x, y)];
        sgg += g * g;
        sg += g;
        sgi += g * i;
        si += i;
    }
    let n = (patch.width() * patch.height()) as f32;
    let det = sgg * n - sg * sg;
    let a = (n * sgi - sg * si) / det;
    let b = (sgg * si - sg * sgi) / det;
    (a, b)
}

/// C4. One Gauss–Newton step for the position, with `A`, `B` and `s` fixed, for each axis
/// separately: `Δx0 = Σ Jx r / Σ Jx²` with the residual `r = I − (B + A g)` and the derivative
/// `Jx = ∂(model)/∂x0 = A g (x − x0) / s²` (and the same for y). An axis with `Σ J² = 0` (a
/// patch one pixel high) does not move. Returns `[Δx0, Δy0]`.
pub fn position_step(patch: &Image<f32>, x0: f32, y0: f32, s: f32, a: f32, b: f32) -> [f32; 2] {
    let (mut jxr, mut jxx, mut jyr, mut jyy) = (0.0, 0.0, 0.0, 0.0);
    for (x, y) in patch.coords() {
        let g = shape(x, y, x0, y0, s);
        let r = patch[(x, y)] - (b + a * g);
        let jx = a * g * (x as f32 - x0) / (s * s);
        let jy = a * g * (y as f32 - y0) / (s * s);
        jxr += jx * r;
        jxx += jx * jx;
        jyr += jy * r;
        jyy += jy * jy;
    }
    let step = |num: f32, den: f32| if den > 0.0 { num / den } else { 0.0 };
    [step(jxr, jxx), step(jyr, jyy)]
}

/// The result of a Gaussian fit.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub x: f32,
    pub y: f32,
    pub amplitude: f32,
    pub background: f32,
}

/// C5. Fit position, amplitude and background of one spot of known width `s`, starting at
/// `start`: `iterations` times, first C3 (A and B at the current position), then C4 (move the
/// position, by at most 1 pixel per step, so that a bad start cannot throw it far away).
pub fn fit_spot(patch: &Image<f32>, start: [f32; 2], s: f32, iterations: usize) -> Fit {
    let [mut x0, mut y0] = start;
    for _ in 0..iterations {
        let (a, b) = fit_amplitude_background(patch, x0, y0, s);
        let [dx, dy] = position_step(patch, x0, y0, s, a, b);
        x0 += dx.clamp(-1.0, 1.0);
        y0 += dy.clamp(-1.0, 1.0);
    }
    let (a, b) = fit_amplitude_background(patch, x0, y0, s);
    Fit { x: x0, y: y0, amplitude: a, background: b }
}
