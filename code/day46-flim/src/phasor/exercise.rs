// Coding exercises for "Frequency domain and phasors". Replace each `todo!()` with your code,
// then check it with `cargo test -p day46 phasor::` and look at `cargo run -p day46 -- phasor`.
#![allow(unused_variables, unused_imports)]

use imgcourse::Complex;
use std::f32::consts::PI;

/// C1. The phasor (g, s) of a histogram over one laser period: the first Fourier coefficient,
/// divided by the total count. Bin j is taken at its centre tⱼ = (j + ½)·T/n, and ω = 2π/T:
/// g = Σ yⱼ cos ωtⱼ / Σ yⱼ,  s = Σ yⱼ sin ωtⱼ / Σ yⱼ.
pub fn phasor(counts: &[f32], period: f32) -> (f32, f32) {
    todo!("C1: phasor")
}

/// C2a. The phasor of a single exponential of lifetime `tau` at angular frequency `omega`
/// (rad/ns): a point on the universal semicircle, g = 1/(1 + ω²τ²), s = ωτ·g.
pub fn single_phasor(tau: f32, omega: f32) -> (f32, f32) {
    todo!("C2: single_phasor")
}

/// C2b. The phase lifetime τφ = tan φ / ω = s / (g ω) and the modulation lifetime
/// τm = √(1/m² − 1) / ω with m² = g² + s². Equal for one exponential; τφ < τm for a mixture.
pub fn phase_and_modulation_lifetimes(g: f32, s: f32, omega: f32) -> (f32, f32) {
    todo!("C2: phase_and_modulation_lifetimes")
}

/// C3. Calibration: the IRF multiplies every phasor by one complex number. A reference of known
/// lifetime `ref_tau`, measured at `ref_measured`, tells which: the corrected phasor is
/// P · P_ref,true / P_ref,measured (complex numbers P = g + i s).
///
/// Hint: `Complex::new(g, s)`; there is no complex division, but z / w = z · conj(w) / |w|²,
/// i.e. `z * w.conj() / w.norm_sqr()`. The result is `(c.re, c.im)`.
pub fn calibrate(p: (f32, f32), ref_measured: (f32, f32), ref_tau: f32, omega: f32) -> (f32, f32) {
    todo!("C3: calibrate")
}

/// C4. The lever rule: how far along the line from `a` to `b` the point `p` lies, as a fraction
/// (0 at a, 1 at b), by projecting p onto the line. For a mixture of two species this is the
/// photon fraction of species b.
///
/// Hint: the dot product of p − a with b − a, divided by the squared length of b − a.
pub fn fraction_along(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    todo!("C4: fraction_along")
}
