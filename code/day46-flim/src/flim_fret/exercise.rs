// Coding exercises for "FLIM-FRET and metabolic imaging". Replace each `todo!()` with your code,
// then check it with `cargo test -p day46 flim_fret::` and look at `cargo run -p day46 -- flim-fret`.
#![allow(unused_variables, unused_imports)]

/// C1a. The FRET efficiency from the donor's lifetime with (`tau_da`) and without (`tau_d`)
/// the acceptor: E = 1 − τDA / τD.
pub fn fret_efficiency(tau_da: f32, tau_d: f32) -> f32 {
    todo!("C1: fret_efficiency")
}

/// C1b. The apparent donor–acceptor distance for efficiency `e` and Förster distance `r0`:
/// r = R0 (1/E − 1)^(1/6).
pub fn distance(e: f32, r0: f32) -> f32 {
    todo!("C1: distance")
}

/// C2a. The photon fraction of the FRET state when a fraction `alpha` of the donors interact:
/// f = α τDA / (α τDA + (1 − α) τD). Quenched donors are dimmer, so f < α.
pub fn photon_fraction(alpha: f32, tau_da: f32, tau_d: f32) -> f32 {
    todo!("C2: photon_fraction")
}

/// C2b. The inverse: the fraction of interacting donors α from the photon fraction f,
/// α = (f / τDA) / (f / τDA + (1 − f) / τD).
pub fn interacting_fraction(f: f32, tau_da: f32, tau_d: f32) -> f32 {
    todo!("C2: interacting_fraction")
}

/// C3. FRET from the phasor line: the straight line from the donor's point `donor` (on the
/// semicircle) through the phasor `p` of a pixel (or of a whole cloud) meets the semicircle
/// again at the FRET state. Returns (τDA, f): the FRET state's lifetime, τ = s / (g ω) at that
/// point, and the photon fraction f of the FRET state, how far along the line p lies.
///
/// Hint: write the line as donor + t·d with d = p − donor, and put it into the circle
/// (g − ½)² + s² = ¼. The quadratic in t has the root t = 0 (the donor, already on the circle);
/// the other root is t = −2 ((g_D − ½) d_g + s_D d_s) / |d|². The pixel itself is at t = 1, so it
/// lies the fraction 1/t of the way from the donor to the FRET state.
pub fn fret_from_line(donor: (f32, f32), p: (f32, f32), omega: f32) -> (f32, f32) {
    todo!("C3: fret_from_line")
}

/// C4. Three species with phasors `a`, `b`, `c`: the photon fractions of a pixel at `p`, its
/// barycentric coordinates. Each fraction is the signed area of the triangle formed by p and the
/// other two corners, divided by the area of the whole triangle. They sum to 1, and are all
/// ≥ 0 exactly when p lies inside the triangle.
///
/// Hint: twice the signed area of the triangle (u, v, w) is
/// (v.0 − u.0)(w.1 − u.1) − (w.0 − u.0)(v.1 − u.1). The fraction of `a` uses the triangle (p, b, c).
pub fn barycentric(p: (f32, f32), a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> [f32; 3] {
    todo!("C4: barycentric")
}
