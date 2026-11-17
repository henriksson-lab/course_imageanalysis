//! Parametric spectra of a cyan donor and a yellow acceptor (shapes only resemble real
//! fluorescent proteins), random directions, and a plain text plot.
//! Not part of the exercise: read it if you like.

use imgcourse::Rng;

/// A split Gaussian: 1 at `peak`, standard deviation `left` below the peak and `right` above.
pub fn split_gaussian(l: f32, peak: f32, left: f32, right: f32) -> f32 {
    let s = if l < peak { left } else { right };
    (-0.5 * ((l - peak) / s).powi(2)).exp()
}

/// The wavelengths of the spectra: 300 to 900 nm in steps of 0.5 nm (the midpoints).
pub fn wavelengths() -> Vec<f32> {
    (0..1200).map(|i| 300.25 + 0.5 * i as f32).collect()
}

/// The donor's emission spectrum (relative): peak 476 nm, long red tail.
pub fn donor_emission(l: f32) -> f32 {
    split_gaussian(l, 476.0, 13.0, 26.0)
}

/// The acceptor's extinction coefficient (M⁻¹ cm⁻¹): peak 83 000 at 514 nm (+ `shift` nm),
/// long blue tail.
pub fn acceptor_extinction(l: f32, shift: f32) -> f32 {
    83000.0 * split_gaussian(l, 514.0 + shift, 11.0 * 2.2, 11.0)
}

/// The donor's quantum yield.
pub const Q_DONOR: f32 = 0.40;
/// The refractive index of the medium.
pub const N_MEDIUM: f32 = 1.4;

/// A random direction in 3D (a unit vector, uniform over the sphere).
pub fn random_direction(rng: &mut Rng) -> [f32; 3] {
    let z = 2.0 * rng.uniform() - 1.0;
    let t = std::f32::consts::TAU * rng.uniform();
    let s = (1.0 - z * z).sqrt();
    [s * t.cos(), s * t.sin(), z]
}

/// A plain text plot of y against x, `rows` lines high, values from `lo` to `hi`.
pub fn text_plot(xs: &[f32], ys: &[f32], lo: f32, hi: f32, rows: usize) -> String {
    let mut out = String::new();
    for row in (0..rows).rev() {
        let level = lo + (hi - lo) * (row as f32 + 0.5) / rows as f32;
        out.push_str(&format!("{:>5.2} |", level));
        for &y in ys {
            out.push(if (y - level).abs() <= 0.5 * (hi - lo) / rows as f32 { '*' } else { ' ' });
        }
        out.push('\n');
    }
    out.push_str(&format!("      +{}\n", "-".repeat(ys.len())));
    out.push_str(&format!("       {:<w$}{:>5}\n", xs[0], xs[xs.len() - 1], w = ys.len() - 5));
    out
}
