// The FRET ruler in numbers: the efficiency curve, R0 from the spectra, κ², populations.
//   cargo run -p fret -- fret-physics                      (your code)
//   cargo run -p fret --features solution -- fret-physics  (the model solution)

use crate::fret_physics::sim;
use crate::fret_physics::*;
use imgcourse::Rng;

pub fn main() {
    // R0 of the cyan–yellow pair, and of acceptors shifted to the red or blue
    let l = sim::wavelengths();
    let donor: Vec<f32> = l.iter().map(|&x| sim::donor_emission(x)).collect();
    println!("Förster radius (κ² = 2/3, Q_D = {}, n = {}):", sim::Q_DONOR, sim::N_MEDIUM);
    let mut r0 = 0.0;
    for shift in [-20.0, 0.0, 20.0, 40.0, 60.0, 80.0] {
        let acc: Vec<f32> = l.iter().map(|&x| sim::acceptor_extinction(x, shift)).collect();
        let j = overlap_integral(&l, &donor, &acc);
        let r = forster_radius(sim::Q_DONOR, 2.0 / 3.0, sim::N_MEDIUM, j);
        if shift == 0.0 {
            r0 = r;
        }
        println!("  acceptor shifted {shift:>4} nm: J = {j:.3e} M⁻¹cm⁻¹nm⁴, R0 = {r:.2} nm, E at 4.95 nm = {:.3}", efficiency(4.95, r));
    }

    // The efficiency curve and its window
    let rs: Vec<f32> = (0..60).map(|i| 0.2 * i as f32).collect();
    let es: Vec<f32> = rs.iter().map(|&r| efficiency(r, r0)).collect();
    println!("\nE(r) for R0 = {r0:.2} nm, r from 0 to 12 nm:\n{}", sim::text_plot(&rs, &es, 0.0, 1.0, 10));
    for e in [0.05, 0.25, 0.30, 0.35, 0.90, 0.95, 0.99] {
        println!("  E = {e:.2}: apparent distance {:.2} nm", apparent_distance(e, r0));
    }

    // κ²
    let mut rng = Rng::new(42);
    println!("\nmean κ² of 100 000 random dipole pairs: {:.3}", mean_kappa2(100_000, &mut rng));

    // Populations
    let two = [0.5 * r0, r0];
    let e_two = mean_efficiency(&two, r0, false, &mut rng);
    println!("\nhalf at 0.5 R0, half at R0: mean E = {e_two:.3}, apparent distance {:.3} R0 (mean distance 0.75 R0)", apparent_distance(e_two, r0) / r0);
    let spread: Vec<f32> = (0..2000).map(|_| r0 + 1.2 * rng.normal()).collect();
    let mean_r = spread.iter().sum::<f32>() / spread.len() as f32;
    for frozen in [false, true] {
        let e = mean_efficiency(&spread, r0, frozen, &mut Rng::new(7));
        println!("2000 molecules, r = R0 ± 1.2 nm (mean {mean_r:.2} nm), {}: mean E = {e:.3}, apparent distance {:.2} nm",
            if frozen { "frozen random orientations" } else { "κ² = 2/3 for all" }, apparent_distance(e, r0));
    }
    let at_r0 = vec![r0; 2000];
    let e = mean_efficiency(&at_r0, r0, true, &mut Rng::new(7));
    println!("2000 molecules all at R0, frozen random orientations: mean E = {e:.3}, apparent distance {:.2} nm", apparent_distance(e, r0));
}
