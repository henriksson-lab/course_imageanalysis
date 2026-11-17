// Lists every state of small Ising lattices: energy levels, Z and probabilities, a picture of
// all 512 states of the 3 × 3 lattice (out/ising/), the cost of listing, and the transfer matrix.
//   cargo run -p day28 -- ising                      (your code)
//   cargo run -p day28 --features solution -- ising  (the model solution)

use imgcourse::*;
use crate::ising::*;
use std::time::Instant;

pub fn main() {
    let dir = "out/ising";

    // The 2 × 2 lattice of B1 at several temperatures.
    let levels = energy_levels(2, 2, 1.0, 0.0, false);
    println!("2 × 2 lattice, J = 1, h = 0: levels (E, number of states) {levels:?}");
    println!("    T        Z     p(E=−4)  p(E=0)  p(E=4)");
    for t in [0.5, 1.0, 2.0, 4.0, 10.0, 100.0] {
        let (z, p) = level_probabilities(&levels, t);
        println!("{t:6.1} {z:10.2}   {:.3}   {:.3}   {:.3}", p[0], p[1], p[2]);
    }

    // The 3 × 3 torus of E2.
    let levels = energy_levels(3, 3, 1.0, 0.0, true);
    let (z, p) = level_probabilities(&levels, 2.0);
    println!("\n3 × 3 periodic, T = 2: Z = {z:.1}");
    for ((e, n), p) in levels.iter().zip(&p) {
        println!("  E = {e:5.1}: {n:3} states, probability {p:.3}");
    }

    // All 512 states of the 3 × 3 torus, sorted by energy, as tiles (orange +1, blue −1).
    let mut states: Vec<(f32, Spins)> = (0..512).map(|i| state_from_index(i, 3, 3)).map(|s| (energy(&s, 1.0, 0.0, true), s)).collect();
    states.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (tile, gap, cols) = (4, 2, 32);
    let mut sheet = Image::new(cols * (3 * tile + gap) + gap, 16 * (3 * tile + gap) + gap, [1.0f32; 3]);
    for (k, (_, s)) in states.iter().enumerate() {
        let (x0, y0) = (gap + (k % cols) * (3 * tile + gap), gap + (k / cols) * (3 * tile + gap));
        for y in 0..3 * tile {
            for x in 0..3 * tile {
                sheet[(x0 + x, y0 + y)] = if s[(x / tile, y / tile)] > 0 { [0.93, 0.55, 0.13] } else { [0.20, 0.42, 0.80] };
            }
        }
    }
    save_rgb(&sheet, format!("{dir}/states-3x3.png"));

    // The cost of listing every state.
    println!("\nlisting all states (J = 1, h = 0, free boundary):");
    for (w, h) in [(3, 3), (4, 4), (5, 4)] {
        let start = Instant::now();
        let levels = energy_levels(w, h, 1.0, 0.0, false);
        let secs = start.elapsed().as_secs_f64();
        let n = w * h;
        println!("  {w} × {h}: 2^{n} = {} states, {} levels, {secs:.3} s ({:.0} ns per state)", 1u64 << n, levels.len(), 1e9 * secs / (1u64 << n) as f64);
    }

    // A ring: the transfer matrix against listing all states.
    println!("\nring of n spins, J = 1, h = 0.3, T = 1.5: Z by the transfer matrix and by listing");
    for n in [3, 6, 10, 14, 18] {
        let start = Instant::now();
        let z_tm = ring_partition_function(n, 1.0, 0.3, 1.5);
        let t_tm = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let (z_list, _) = level_probabilities(&energy_levels(n, 1, 1.0, 0.3, true), 1.5);
        let t_list = start.elapsed().as_secs_f64();
        println!("  n = {n:2}: {z_tm:14.2} ({:.1} µs)   {z_list:14.2} ({:.1} ms)", 1e6 * t_tm, 1e3 * t_list);
    }
    println!("\nwrote {dir}/states-3x3.png (the 512 states of the 3 × 3 torus, lowest energy first)");
}
