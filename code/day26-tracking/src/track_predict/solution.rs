// Model solution for the coding exercises of "Tracking: motion models and the Kalman filter".

/// A 1D estimate: the best guess `x` and its variance `p` (uncertainty squared).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub x: f32,
    pub p: f32,
}

/// C1a. Predict with the random-walk model: the guess stays, the variance grows by `q`.
pub fn predict(e: Estimate, q: f32) -> Estimate {
    Estimate { x: e.x, p: e.p + q }
}

/// C1b. Update a prediction with a measurement `z` of variance `r`: returns the new estimate
/// and the gain K (slide "The Kalman filter: predict, then update").
pub fn update(pred: Estimate, z: f32, r: f32) -> (Estimate, f32) {
    let k = pred.p / (pred.p + r);
    let x = pred.x + k * (z - pred.x);
    let p = (1.0 - k) * pred.p;
    (Estimate { x, p }, k)
}

/// C2. Run the random-walk filter over a sequence of measurements, starting from `first`.
/// Every frame: predict, then update if there is a measurement (`Some(z)`); a missed
/// detection (`None`) keeps the prediction. Returns the estimate after every frame.
pub fn run_filter(first: Estimate, zs: &[Option<f32>], q: f32, r: f32) -> Vec<Estimate> {
    let mut e = first;
    let mut out = Vec::new();
    for z in zs {
        e = predict(e, q);
        if let Some(z) = z {
            e = update(e, *z, r).0;
        }
        out.push(e);
    }
    out
}

/// A constant-velocity estimate: state `[position, velocity]` and its 2 × 2 covariance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CvEstimate {
    pub x: [f32; 2],
    pub p: [[f32; 2]; 2],
}

/// C3a. Predict with the constant-velocity model over `dt`: position += velocity · dt, and
/// P⁻ = F P Fᵀ + Q with F = [[1, dt], [0, 1]] and Q = diag(0, `q_v`): noise on the velocity
/// only (math slide "The matrix form").
pub fn cv_predict(e: CvEstimate, dt: f32, q_v: f32) -> CvEstimate {
    let [[a, b], [_, d]] = e.p; // symmetric: p[1][0] = p[0][1] = b
    let x = [e.x[0] + dt * e.x[1], e.x[1]];
    // F P Fᵀ written out
    let pxx = a + 2.0 * dt * b + dt * dt * d;
    let pxv = b + dt * d;
    let pvv = d + q_v;
    CvEstimate { x, p: [[pxx, pxv], [pxv, pvv]] }
}

/// C3b. Update with a measured position `z` of variance `r`. Only the position is measured
/// (H = (1 0)), so S = P⁻ₓₓ + r and the gain is K = (P⁻ₓₓ, P⁻ₓᵥ) / S. Returns the new
/// estimate and K.
pub fn cv_update(pred: CvEstimate, z: f32, r: f32) -> (CvEstimate, [f32; 2]) {
    let [[a, b], [_, d]] = pred.p;
    let s = a + r;
    let k = [a / s, b / s];
    let y = z - pred.x[0]; // the innovation
    let x = [pred.x[0] + k[0] * y, pred.x[1] + k[1] * y];
    // P = (I − K H) P⁻
    let p = [[(1.0 - k[0]) * a, (1.0 - k[0]) * b], [(1.0 - k[0]) * b, d - k[1] * b]];
    (CvEstimate { x, p }, k)
}

/// C4. The squared Mahalanobis distance d² = yᵀ S⁻¹ y of a 2D innovation `y` with covariance
/// `s` (symmetric 2 × 2): the distance in standard deviations, along the shape of the ellipse.
pub fn mahalanobis2(y: [f32; 2], s: [[f32; 2]; 2]) -> f32 {
    let [[a, b], [_, d]] = s;
    let det = a * d - b * b;
    // S⁻¹ = [[d, −b], [−b, a]] / det
    (d * y[0] * y[0] - 2.0 * b * y[0] * y[1] + a * y[1] * y[1]) / det
}
