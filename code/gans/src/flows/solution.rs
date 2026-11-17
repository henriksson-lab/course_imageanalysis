// Model solution for the coding exercises of "Normalising flows".

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
    let o = 1 - layer.d;
    let s = (layer.s)(x[o]);
    let mut y = x;
    y[layer.d] = x[layer.d] * s.exp() + (layer.t)(x[o]);
    (y, s)
}

/// C2. One layer backwards (noise → data): x_o = y_o, x_d = (y_d − t(y_o)) · exp(−s(y_o)).
pub fn coupling_inverse(layer: &Coupling, y: [f64; 2]) -> [f64; 2] {
    let o = 1 - layer.d;
    let mut x = y;
    x[layer.d] = (y[layer.d] - (layer.t)(y[o])) * (-(layer.s)(y[o])).exp();
    x
}

/// C3. The log-density of a flow (layers applied in order, data → noise) at x, by the change of
/// variables: log p(x) = log N(z; 0, I) + Σ log-determinants, with log N(z) = −|z|²/2 − log 2π.
pub fn log_density(flow: &[Coupling], x: [f64; 2]) -> f64 {
    let mut v = x;
    let mut logdet = 0.0;
    for layer in flow {
        let (y, s) = coupling_forward(layer, v);
        v = y;
        logdet += s;
    }
    -0.5 * (v[0] * v[0] + v[1] * v[1]) - (2.0 * std::f64::consts::PI).ln() + logdet
}

/// C4. A sample of the flow's distribution: z ~ N(0, I), then the layers backwards, last first.
pub fn sample(flow: &[Coupling], rng: &mut Rng) -> [f64; 2] {
    let mut v = [gauss(rng), gauss(rng)];
    for layer in flow.iter().rev() {
        v = coupling_inverse(layer, v);
    }
    v
}
