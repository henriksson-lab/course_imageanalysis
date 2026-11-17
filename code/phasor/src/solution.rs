// Model solution for the coding exercises of "Frequency domain and phasors".

use imgcourse::Complex;
use std::f32::consts::PI;

/// C1. The phasor (g, s) of a histogram over one laser period: the first Fourier coefficient,
/// divided by the total count. Bin j is taken at its centre tⱼ = (j + ½)·T/n, and ω = 2π/T:
/// g = Σ yⱼ cos ωtⱼ / Σ yⱼ,  s = Σ yⱼ sin ωtⱼ / Σ yⱼ.
pub fn phasor(counts: &[f32], period: f32) -> (f32, f32) {
    let omega = 2.0 * PI / period;
    let (mut g, mut s, mut total) = (0.0, 0.0, 0.0);
    for (j, &y) in counts.iter().enumerate() {
        let t = (j as f32 + 0.5) * period / counts.len() as f32;
        g += y * (omega * t).cos();
        s += y * (omega * t).sin();
        total += y;
    }
    (g / total, s / total)
}

/// C2a. The phasor of a single exponential of lifetime `tau` at angular frequency `omega`
/// (rad/ns): a point on the universal semicircle, g = 1/(1 + ω²τ²), s = ωτ·g.
pub fn single_phasor(tau: f32, omega: f32) -> (f32, f32) {
    let wt = omega * tau;
    let g = 1.0 / (1.0 + wt * wt);
    (g, wt * g)
}

/// C2b. The phase lifetime τφ = tan φ / ω = s / (g ω) and the modulation lifetime
/// τm = √(1/m² − 1) / ω with m² = g² + s². Equal for one exponential; τφ < τm for a mixture.
pub fn phase_and_modulation_lifetimes(g: f32, s: f32, omega: f32) -> (f32, f32) {
    let m2 = g * g + s * s;
    (s / (g * omega), (1.0 / m2 - 1.0).sqrt() / omega)
}

/// C3. Calibration: the IRF multiplies every phasor by one complex number. A reference of known
/// lifetime `ref_tau`, measured at `ref_measured`, tells which: the corrected phasor is
/// P · P_ref,true / P_ref,measured (complex numbers P = g + i s).
pub fn calibrate(p: (f32, f32), ref_measured: (f32, f32), ref_tau: f32, omega: f32) -> (f32, f32) {
    let (gt, st) = single_phasor(ref_tau, omega);
    let p = Complex::new(p.0, p.1);
    let r_true = Complex::new(gt, st);
    let r_meas = Complex::new(ref_measured.0, ref_measured.1);
    // division by a complex number: multiply by its conjugate, divide by its squared length
    let c = p * r_true * r_meas.conj() / r_meas.norm_sqr();
    (c.re, c.im)
}

/// C4. The lever rule: how far along the line from `a` to `b` the point `p` lies, as a fraction
/// (0 at a, 1 at b), by projecting p onto the line. For a mixture of two species this is the
/// photon fraction of species b.
pub fn fraction_along(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    ((p.0 - a.0) * dx + (p.1 - a.1) * dy) / (dx * dx + dy * dy)
}
