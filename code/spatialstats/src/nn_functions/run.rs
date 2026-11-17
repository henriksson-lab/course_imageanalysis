// G, F and J of four simulated patterns, and the edge effect in a small window; numbers only.
//   cargo run -p spatialstats -- nn-functions                      (your code)
//   cargo run -p spatialstats --features solution -- nn-functions  (the model solution)

use imgcourse::Rng;
use crate::nn_functions::*;

pub fn main() {
    let (a, b) = (200.0, 200.0);
    let mut rng = Rng::new(2);
    let random = csr(150, a, b, &mut rng);
    let clusters = thomas(0.001, 5.0, 4.0, a, b, &mut rng);
    let regular = hard_core(150, 10.0, a, b, &mut rng);
    // tight pairs: a random point and a partner 3 µm away in a random direction
    let mut pairs = Vec::new();
    for p in csr(75, a - 6.0, b - 6.0, &mut rng) {
        let t = 2.0 * std::f64::consts::PI * rng.uniform() as f64;
        pairs.push((p.0 + 3.0, p.1 + 3.0));
        pairs.push((p.0 + 3.0 + 3.0 * t.cos(), p.1 + 3.0 + 3.0 * t.sin()));
    }
    let tests = grid(a, b, 2.0);
    let radii = [2.0, 4.0, 6.0, 8.0, 10.0];
    println!("border-corrected estimates in a {a} × {b} µm window (F from a 2 µm grid of test locations)");
    for (name, pts) in [("CSR", &random), ("clusters", &clusters), ("hard core 10 µm", &regular), ("pairs 3 µm apart", &pairs)] {
        let rho = pts.len() as f64 / (a * b);
        let mean_nn = nn_distances(pts).iter().sum::<f64>() / pts.len() as f64;
        println!();
        println!("{name}: {} points, Clark–Evans R = {:.2}", pts.len(), mean_nn / (0.5 / rho.sqrt()));
        println!("{:>6} {:>7} {:>7} {:>7} {:>7}", "r", "CSR", "G", "F", "J");
        for r in radii {
            let theory = 1.0 - (-rho * std::f64::consts::PI * r * r).exp();
            let (g, f) = (g_hat(pts, a, b, r, true), f_hat(pts, &tests, a, b, r, true));
            println!("{r:6.0} {theory:7.3} {g:7.3} {f:7.3} {:7.2}", j_hat(pts, a, b, r, 2.0));
        }
    }

    // the edge effect: 1000 random patterns of 14 points in a 60 × 60 µm window, at r = 10 µm
    let (s, n, r) = (60.0, 14, 10.0);
    let (mut none, mut border, mut used) = (0.0, 0.0, 0);
    for _ in 0..1000 {
        let pts = csr(n, s, s, &mut rng);
        none += g_hat(&pts, s, s, r, false);
        let g = g_hat(&pts, s, s, r, true);
        if !g.is_nan() {
            border += g;
            used += 1;
        }
    }
    let theory = 1.0 - (-(n as f64 / (s * s)) * std::f64::consts::PI * r * r).exp();
    println!();
    println!("{n} random points in {s} × {s} µm, mean of 1000 patterns at r = {r} µm:");
    println!("  theory {theory:.3}, no correction {:.3}, border rule {:.3} ({} patterns had no point 10 µm from the border)", none / 1000.0, border / used as f64, 1000 - used);
}
