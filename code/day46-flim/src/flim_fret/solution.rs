// Model solution for the coding exercises of "FLIM-FRET and metabolic imaging".

/// C1a. The FRET efficiency from the donor's lifetime with (`tau_da`) and without (`tau_d`)
/// the acceptor: E = 1 − τDA / τD.
pub fn fret_efficiency(tau_da: f32, tau_d: f32) -> f32 {
    1.0 - tau_da / tau_d
}

/// C1b. The apparent donor–acceptor distance for efficiency `e` and Förster distance `r0`:
/// r = R0 (1/E − 1)^(1/6).
pub fn distance(e: f32, r0: f32) -> f32 {
    r0 * (1.0 / e - 1.0).powf(1.0 / 6.0)
}

/// C2a. The photon fraction of the FRET state when a fraction `alpha` of the donors interact:
/// f = α τDA / (α τDA + (1 − α) τD). Quenched donors are dimmer, so f < α.
pub fn photon_fraction(alpha: f32, tau_da: f32, tau_d: f32) -> f32 {
    alpha * tau_da / (alpha * tau_da + (1.0 - alpha) * tau_d)
}

/// C2b. The inverse: the fraction of interacting donors α from the photon fraction f,
/// α = (f / τDA) / (f / τDA + (1 − f) / τD).
pub fn interacting_fraction(f: f32, tau_da: f32, tau_d: f32) -> f32 {
    (f / tau_da) / (f / tau_da + (1.0 - f) / tau_d)
}

/// C3. FRET from the phasor line: the straight line from the donor's point `donor` (on the
/// semicircle) through the phasor `p` of a pixel (or of a whole cloud) meets the semicircle
/// again at the FRET state. Returns (τDA, f): the FRET state's lifetime, τ = s / (g ω) at that
/// point, and the photon fraction f of the FRET state, how far along the line p lies.
pub fn fret_from_line(donor: (f32, f32), p: (f32, f32), omega: f32) -> (f32, f32) {
    // points on the line: donor + t (p − donor); on the circle |q − (½, 0)|² = ¼.
    // t = 0 is the donor itself; the other solution of the quadratic is t = −2 (donor − c)·d / |d|².
    let (dg, ds) = (p.0 - donor.0, p.1 - donor.1);
    let t = -2.0 * ((donor.0 - 0.5) * dg + donor.1 * ds) / (dg * dg + ds * ds);
    let (g, s) = (donor.0 + t * dg, donor.1 + t * ds);
    (s / (g * omega), 1.0 / t)
}

/// C4. Three species with phasors `a`, `b`, `c`: the photon fractions of a pixel at `p`, its
/// barycentric coordinates. Each fraction is the signed area of the triangle formed by p and the
/// other two corners, divided by the area of the whole triangle. They sum to 1, and are all
/// ≥ 0 exactly when p lies inside the triangle.
pub fn barycentric(p: (f32, f32), a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> [f32; 3] {
    // twice the signed area of the triangle (u, v, w)
    let area = |u: (f32, f32), v: (f32, f32), w: (f32, f32)| (v.0 - u.0) * (w.1 - u.1) - (w.0 - u.0) * (v.1 - u.1);
    let whole = area(a, b, c);
    [area(p, b, c) / whole, area(a, p, c) / whole, area(a, b, p) / whole]
}
