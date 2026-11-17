// Coding exercises for "FRET in pictures". Replace each `todo!()` with your code,
// then check it with `cargo test -p fret fret_physics::` and look at `cargo run -p fret -- fret-physics`.
#![allow(unused_variables, unused_imports)]

use crate::fret_physics::sim::random_direction;
use imgcourse::Rng;

/// C1a. The FRET efficiency at distance `r` for Förster radius `r0` (slide "The efficiency
/// curve"): E = 1 / (1 + (r/r0)⁶).
pub fn efficiency(r: f32, r0: f32) -> f32 {
    todo!("C1a: efficiency")
}

/// C1b. The apparent distance of an efficiency (slide "From efficiency to apparent distance"):
/// r = r0 (1/E − 1)^(1/6).
pub fn apparent_distance(e: f32, r0: f32) -> f32 {
    todo!("C1b: apparent_distance")
}

/// C2a. The overlap integral J = Σ F_D(λ) ε_A(λ) λ⁴ Δλ (slide "The math: the Förster radius
/// and κ²"), with the donor emission F_D scaled to unit area (Σ F_D Δλ = 1). The three lists
/// are sampled at the same, equally spaced wavelengths (nm); Δλ is their spacing.
/// The result is in M⁻¹ cm⁻¹ nm⁴.
///
/// Hint: sums of numbers around 10¹⁵ are safer in `f64`.
pub fn overlap_integral(lambda: &[f32], donor_em: &[f32], acc_eps: &[f32]) -> f64 {
    todo!("C2a: overlap_integral")
}

/// C2b. The Förster radius in nm: R0 = 0.0211 (κ² n⁻⁴ Q_D J)^(1/6), with J in M⁻¹ cm⁻¹ nm⁴.
pub fn forster_radius(q_donor: f32, kappa2: f32, n: f32, j: f64) -> f32 {
    todo!("C2b: forster_radius")
}

/// C3. The orientation factor κ² = (μD·μA − 3 (μD·r̂)(μA·r̂))² for unit vectors: the
/// donor's dipole `mu_d`, the acceptor's dipole `mu_a`, and the direction `r_hat` from donor
/// to acceptor.
pub fn kappa2(mu_d: [f32; 3], mu_a: [f32; 3], r_hat: [f32; 3]) -> f32 {
    todo!("C3: kappa2")
}

/// C4a. The mean of κ² over `n` pairs of independent random dipole directions, the
/// donor–acceptor direction fixed along x. Should come out close to 2/3.
///
/// Hint: `random_direction(rng)` (given) is a random unit vector.
pub fn mean_kappa2(n: usize, rng: &mut Rng) -> f32 {
    todo!("C4a: mean_kappa2")
}

/// C4b. The mean efficiency of a population of molecules at the given `distances`
/// (slide "A mixture of distances"). `r0` is the Förster radius for κ² = 2/3. If `frozen`,
/// every molecule has its own random, fixed dipole directions: its κ² gives its own Förster
/// radius r0 (κ²/(2/3))^(1/6). Otherwise every molecule uses r0.
pub fn mean_efficiency(distances: &[f32], r0: f32, frozen: bool, rng: &mut Rng) -> f32 {
    todo!("C4b: mean_efficiency")
}
