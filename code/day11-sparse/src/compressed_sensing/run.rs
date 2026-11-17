// Sparse signals measured with fewer numbers than unknowns: zero filling with regular and
// random Fourier coefficients, recovery by ℓ1, and a phase diagram. Results in
// out/compressed-sensing/.
//   cargo run -p day11 -- compressed-sensing                      (your code)
//   cargo run -p day11 --features solution -- compressed-sensing  (the model solution)

use crate::compressed_sensing::*;
use crate::common::{plot, Style};
use imgcourse::*;

pub fn main() {
    let dir = "out/compressed-sensing";

    // The by-hand example R1.
    let a = vec![vec![1.0, 1.0, 1.0, 1.0], vec![1.0, -1.0, 2.0, 0.0]];
    println!("R1: minimum norm {:.3?}, ℓ1 recovery {:.3?}", min_norm(&a, &[3.0, 6.0]), recover(&a, &[3.0, 6.0], 2000));
    println!("R8: mutual coherence {:.3}", coherence(&a));

    // Eight spikes of random sign in 256 samples.
    let n = 256;
    let mut rng = Rng::new(5);
    let mut x = vec![0.0; n];
    for _ in 0..8 {
        let mut j = rng.below(n);
        while x[j] != 0.0 {
            j = rng.below(n);
        }
        x[j] = if rng.uniform() < 0.5 { 1.0 } else { -1.0 } * (1.0 + rng.uniform() as f64);
    }

    // 64 of the 256 Fourier coefficients: every 4th, or 32 random frequencies together with
    // their negative partners (k and n − k), so that the zero-filled signal is real.
    let regular: Vec<bool> = (0..n).map(|k| k % 4 == 0).collect();
    let mut random = vec![false; n];
    let mut picked = 0;
    while picked < 32 {
        let k = 1 + rng.below(n / 2 - 1);
        if !random[k] {
            random[k] = true;
            random[n - k] = true;
            picked += 1;
        }
    }
    let grey = [0.6, 0.6, 0.6];
    let blue = [0.1, 0.3, 0.9];
    let black = [0.0, 0.0, 0.0];
    for (name, keep) in [("regular", &regular), ("random", &random)] {
        let m = keep.iter().filter(|k| **k).count();
        // Scaled by n/m, so that a spike that comes back comes back at its own height.
        let z: Vec<f64> = zero_filled(&x, keep).iter().map(|v| v * n as f64 / m as f64).collect();
        let off: Vec<f64> = (0..n).filter(|&i| x[i] == 0.0).map(|i| z[i].abs()).collect();
        let junk = (off.iter().map(|v| v * v).sum::<f64>() / off.len() as f64).sqrt();
        let biggest = off.iter().fold(0.0f64, |a, b| a.max(*b));
        println!("zero filling, {name} ({m} of {n} coefficients): away from the spikes RMS {junk:.3}, largest {biggest:.3}");
        save_rgb(&plot(&[(&z, grey, Style::Line), (&x, black, Style::Dots)], 2, 200), format!("{dir}/zero-filled-{name}.png"));

        // The same measurements as a real matrix: for each kept frequency 0 < k < n/2 a cosine
        // and a sine row (orthonormal), plus the constant row for k = 0. Then recover by ℓ1.
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for k in 0..n / 2 {
            if keep[k] {
                let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
                if k == 0 {
                    rows.push(vec![1.0 / (n as f64).sqrt(); n]);
                } else {
                    let s = (2.0 / n as f64).sqrt();
                    rows.push((0..n).map(|i| s * (w * i as f64).cos()).collect());
                    rows.push((0..n).map(|i| s * (w * i as f64).sin()).collect());
                }
            }
        }
        let xr = recover(&rows, &mat_vec(&rows, &x), 2000);
        let err = xr.iter().zip(&x).map(|(p, q)| (p - q).powi(2)).sum::<f64>().sqrt() / x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nz = xr.iter().filter(|v| v.abs() > 1e-6).count();
        println!("  ℓ1 recovery from {} measurements: {nz} nonzeros, relative error {err:.3}", rows.len());
        save_rgb(&plot(&[(&x, black, Style::Stems), (&xr, blue, Style::Dots)], 2, 200), format!("{dir}/recovered-{name}.png"));
    }

    // The phase diagram, N = 64: δ = M/N across, ρ = K/M down (small ρ at the bottom).
    let n = 64;
    let deltas: Vec<usize> = (1..8).map(|i| 8 * i).collect(); // M = 8, 16, ..., 56
    let rhos: Vec<f64> = (1..=9).map(|i| 0.1 * i as f64).collect();
    let trials = 8;
    let cell = 32;
    let mut pic = Image::new(deltas.len() * cell, rhos.len() * cell, 0.0f32);
    println!("phase diagram, N = {n}, {trials} trials per cell: fraction recovered");
    println!("   ρ \\ δ  {}", deltas.iter().map(|m| format!("{:5.2}", *m as f64 / n as f64)).collect::<String>());
    for (ri, rho) in rhos.iter().enumerate().rev() {
        let mut line = format!("   {rho:.1}   ");
        for (di, &m) in deltas.iter().enumerate() {
            let k = ((rho * m as f64).round() as usize).max(1);
            let rate = success_rate(n, m, k, trials, (100 * m + k) as u64);
            line += &format!("{rate:5.2}");
            for dy in 0..cell {
                for dx in 0..cell {
                    pic[(di * cell + dx, (rhos.len() - 1 - ri) * cell + dy)] = rate as f32;
                }
            }
        }
        println!("{line}");
    }
    save_gray(&pic, format!("{dir}/phase-diagram.png"));
    println!("wrote the results to {dir}/ (black: truth, grey: zero-filled, blue dots: recovered; diagram: white = always recovered)");
}
