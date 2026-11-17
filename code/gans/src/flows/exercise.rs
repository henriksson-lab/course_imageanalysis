// Coding exercises for "Normalising flows". Replace each `todo!()` with your code,
// then check it with `cargo test -p gans flows::` and look at `cargo run -p gans -- flows`.
#![allow(unused_variables, unused_imports, unused_mut)]

use crate::common::gauss;
use imgcourse::Rng;

/// An affine coupling layer in 2D (direction: data → noise). It changes coordinate `d` (0 or 1)
/// and keeps the other one, o = 1 − d:
///   y_d = x_d · exp(s(x_o)) + t(x_o),   y_o = x_o.
/// In a trained flow, s and t are small networks; here they are fixed functions.
#[derive(Clone, Copy)]
pub struct Coupling {
    pub d: usize,
    pub s: fn(f64) -> f64,
    pub t: fn(f64) -> f64,
}

/// C1. One layer forward (data → noise): returns y and log |det ∂y/∂x| = s(x_o).
pub fn coupling_forward(layer: &Coupling, x: [f64; 2]) -> ([f64; 2], f64) {
    todo!("C1: coupling_forward")
}

/// C2. One layer backwards (noise → data): x_o = y_o, x_d = (y_d − t(y_o)) · exp(−s(y_o)).
pub fn coupling_inverse(layer: &Coupling, y: [f64; 2]) -> [f64; 2] {
    todo!("C2: coupling_inverse")
}

/// C3. The log-density of a flow (layers applied in order, data → noise) at x, by the change of
/// variables: log p(x) = log N(z; 0, I) + Σ log-determinants, with log N(z) = −|z|²/2 − log 2π.
pub fn log_density(flow: &[Coupling], x: [f64; 2]) -> f64 {
    todo!("C3: log_density")
}

/// C4. A sample of the flow's distribution: z ~ N(0, I), then the layers backwards, last first.
pub fn sample(flow: &[Coupling], rng: &mut Rng) -> [f64; 2] {
    todo!("C4: sample")
}
