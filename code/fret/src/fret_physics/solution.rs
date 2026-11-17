// Model solution for the coding exercises of "FRET in pictures".

use crate::fret_physics::sim::random_direction;
use imgcourse::Rng;

/// C1a. The FRET efficiency at distance `r` for Förster radius `r0` (slide "The efficiency
/// curve"): E = 1 / (1 + (r/r0)⁶).
pub fn efficiency(r: f32, r0: f32) -> f32 {
    1.0 / (1.0 + (r / r0).powi(6))
}

/// C1b. The apparent distance of an efficiency (slide "From efficiency to apparent distance"):
/// r = r0 (1/E − 1)^(1/6).
pub fn apparent_distance(e: f32, r0: f32) -> f32 {
    r0 * (1.0 / e - 1.0).powf(1.0 / 6.0)
}

/// C2a. The overlap integral J = Σ F_D(λ) ε_A(λ) λ⁴ Δλ (slide "The math: the Förster radius
/// and κ²"), with the donor emission F_D scaled to unit area (Σ F_D Δλ = 1). The three lists
/// are sampled at the same, equally spaced wavelengths (nm); Δλ is their spacing.
/// The result is in M⁻¹ cm⁻¹ nm⁴.
pub fn overlap_integral(lambda: &[f32], donor_em: &[f32], acc_eps: &[f32]) -> f64 {
    let step = (lambda[1] - lambda[0]) as f64;
    let (mut area, mut j) = (0.0f64, 0.0f64);
    for i in 0..lambda.len() {
        let l = lambda[i] as f64;
        area += donor_em[i] as f64 * step;
        j += donor_em[i] as f64 * acc_eps[i] as f64 * l.powi(4) * step;
    }
    j / area
}

/// C2b. The Förster radius in nm: R0 = 0.0211 (κ² n⁻⁴ Q_D J)^(1/6), with J in M⁻¹ cm⁻¹ nm⁴.
pub fn forster_radius(q_donor: f32, kappa2: f32, n: f32, j: f64) -> f32 {
    0.0211 * ((kappa2 as f64 * q_donor as f64 * j / (n as f64).powi(4)).powf(1.0 / 6.0)) as f32
}

/// C3. The orientation factor κ² = (μD·μA − 3 (μD·r̂)(μA·r̂))² for unit vectors: the
/// donor's dipole `mu_d`, the acceptor's dipole `mu_a`, and the direction `r_hat` from donor
/// to acceptor.
pub fn kappa2(mu_d: [f32; 3], mu_a: [f32; 3], r_hat: [f32; 3]) -> f32 {
    let dot = |u: [f32; 3], v: [f32; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let k = dot(mu_d, mu_a) - 3.0 * dot(mu_d, r_hat) * dot(mu_a, r_hat);
    k * k
}

/// C4a. The mean of κ² over `n` pairs of independent random dipole directions, the
/// donor–acceptor direction fixed along x. Should come out close to 2/3.
pub fn mean_kappa2(n: usize, rng: &mut Rng) -> f32 {
    let mut sum = 0.0;
    for _ in 0..n {
        sum += kappa2(random_direction(rng), random_direction(rng), [1.0, 0.0, 0.0]);
    }
    sum / n as f32
}

/// C4b. The mean efficiency of a population of molecules at the given `distances`
/// (slide "A mixture of distances"). `r0` is the Förster radius for κ² = 2/3. If `frozen`,
/// every molecule has its own random, fixed dipole directions: its κ² gives its own Förster
/// radius r0 (κ²/(2/3))^(1/6). Otherwise every molecule uses r0.
pub fn mean_efficiency(distances: &[f32], r0: f32, frozen: bool, rng: &mut Rng) -> f32 {
    let mut sum = 0.0;
    for &r in distances {
        let r0_here = if frozen {
            let k2 = kappa2(random_direction(rng), random_direction(rng), [1.0, 0.0, 0.0]);
            r0 * (k2 / (2.0 / 3.0)).powf(1.0 / 6.0)
        } else {
            r0
        };
        sum += efficiency(r, r0_here);
    }
    sum / distances.len() as f32
}
