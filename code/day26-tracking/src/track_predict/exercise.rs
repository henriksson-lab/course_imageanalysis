// Coding exercises for "Tracking: motion models and the Kalman filter". Replace each `todo!()`
// with your code, then check it with `cargo test -p day26 track_predict::` and look at `cargo run -p day26 -- track-predict`.
#![allow(unused_variables, unused_imports)]

/// A 1D estimate: the best guess `x` and its variance `p` (uncertainty squared).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub x: f32,
    pub p: f32,
}

/// C1a. Predict with the random-walk model: the guess stays, the variance grows by `q`.
pub fn predict(e: Estimate, q: f32) -> Estimate {
    todo!("C1a: predict")
}

/// C1b. Update a prediction with a measurement `z` of variance `r`: returns the new estimate
/// and the gain K (slide "The Kalman filter: predict, then update").
///
/// Hint: K = P⁻ / (P⁻ + R); x = x⁻ + K (z − x⁻); P = (1 − K) P⁻.
pub fn update(pred: Estimate, z: f32, r: f32) -> (Estimate, f32) {
    todo!("C1b: update")
}

/// C2. Run the random-walk filter over a sequence of measurements, starting from `first`.
/// Every frame: predict, then update if there is a measurement (`Some(z)`); a missed
/// detection (`None`) keeps the prediction. Returns the estimate after every frame.
///
/// Hint: `if let Some(z) = z { ... }` runs only when there is a measurement.
pub fn run_filter(first: Estimate, zs: &[Option<f32>], q: f32, r: f32) -> Vec<Estimate> {
    todo!("C2: run_filter")
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
///
/// Hint: P is symmetric, `[[a, b], [b, d]]`. Multiply F P Fᵀ out by hand once: it is three
/// numbers (position variance, covariance, velocity variance).
pub fn cv_predict(e: CvEstimate, dt: f32, q_v: f32) -> CvEstimate {
    todo!("C3a: cv_predict")
}

/// C3b. Update with a measured position `z` of variance `r`. Only the position is measured
/// (H = (1 0)), so S = P⁻ₓₓ + r and the gain is K = (P⁻ₓₓ, P⁻ₓᵥ) / S. Returns the new
/// estimate and K.
///
/// Hint: with y = z − x⁻ (position): x += K y for both components;
/// P = (I − K H) P⁻, where K H = [[K₀, 0], [K₁, 0]].
pub fn cv_update(pred: CvEstimate, z: f32, r: f32) -> (CvEstimate, [f32; 2]) {
    todo!("C3b: cv_update")
}

/// C4. The squared Mahalanobis distance d² = yᵀ S⁻¹ y of a 2D innovation `y` with covariance
/// `s` (symmetric 2 × 2): the distance in standard deviations, along the shape of the ellipse.
///
/// Hint: the inverse of [[a, b], [b, d]] is [[d, −b], [−b, a]] / (a d − b²).
pub fn mahalanobis2(y: [f32; 2], s: [[f32; 2]; 2]) -> f32 {
    todo!("C4: mahalanobis2")
}
