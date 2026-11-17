// K, L and the pair correlation of four simulated patterns, and the bias of K without an edge
// correction; numbers only.
//   cargo run -p day19 -- ripley-k                      (your code)
//   cargo run -p day19 --features solution -- ripley-k  (the model solution)

use imgcourse::Rng;
use crate::ripley_k::*;
use std::f64::consts::PI;

/// Colonies of cells that keep a distance: colony centres at random; around each, up to
/// `per_colony` cells placed uniformly within `radius` of the centre, each rejected if closer
/// than `core` to any earlier cell.
fn colonies(n_colonies: usize, per_colony: usize, radius: f64, core: f64, a: f64, b: f64, rng: &mut Rng) -> Vec<Point> {
    let mut pts: Vec<Point> = Vec::new();
    for c in csr(n_colonies, a, b, rng) {
        let mut placed = 0;
        for _ in 0..50 * per_colony {
            if placed == per_colony {
                break;
            }
            let (r, t) = (radius * (rng.uniform() as f64).sqrt(), 2.0 * PI * rng.uniform() as f64);
            let p = (c.0 + r * t.cos(), c.1 + r * t.sin());
            let inside = (0.0..a).contains(&p.0) && (0.0..b).contains(&p.1);
            if inside && pts.iter().all(|&q| dist(p, q) >= core) {
                pts.push(p);
                placed += 1;
            }
        }
    }
    pts
}

pub fn main() {
    let (a, b) = (200.0, 200.0);
    let mut rng = Rng::new(4);
    let patterns = [
        ("CSR", csr(150, a, b, &mut rng)),
        ("clusters σ = 4", thomas(0.0015, 5.0, 4.0, a, b, &mut rng)),
        ("hard core 8", hard_core(150, 8.0, a, b, &mut rng)),
        ("colonies, core 6", colonies(15, 12, 12.0, 6.0, a, b, &mut rng)),
    ];
    let radii = [2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0];
    println!("translation correction; g with the kernel half-width h = 1.5 µm; window {a} × {b} µm");
    println!("{:>6} {}", "r", radii.map(|r| format!("{r:>6}")).join(""));
    for (name, pts) in &patterns {
        println!("{name} ({} points)", pts.len());
        let l: Vec<String> = radii.iter().map(|&r| format!("{:6.2}", l_minus_r(k_hat(pts, a, b, r, true), r))).collect();
        let g: Vec<String> = radii.iter().map(|&r| format!("{:6.2}", pair_correlation(pts, a, b, r, 1.5))).collect();
        println!("{:>6} {}", "L - r", l.join(""));
        println!("{:>6} {}", "g", g.join(""));
    }

    // the edge effect: 200 CSR patterns of 100 points, mean relative error of K against πr²
    println!();
    println!("200 CSR patterns of 100 points in {a} × {b} µm, mean relative error of K:");
    let check = [10.0, 20.0, 50.0];
    let (mut none, mut trans) = ([0.0; 3], [0.0; 3]);
    for _ in 0..200 {
        let pts = csr(100, a, b, &mut rng);
        for (k, &r) in check.iter().enumerate() {
            none[k] += (k_hat(&pts, a, b, r, false) / (PI * r * r) - 1.0) / 200.0;
            trans[k] += (k_hat(&pts, a, b, r, true) / (PI * r * r) - 1.0) / 200.0;
        }
    }
    for (k, r) in check.iter().enumerate() {
        println!("  r = {r:>2} µm: no correction {:+5.1} %, translation {:+5.1} %", 100.0 * none[k], 100.0 * trans[k]);
    }
}
