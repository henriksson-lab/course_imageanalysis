// Two cell types: cross-L under two null models, the mark connection function, and the
// inhomogeneous L of a gradient; numbers only.
//   cargo run -p spatialstats -- marked-patterns                      (your code)
//   cargo run -p spatialstats --features solution -- marked-patterns  (the model solution)

use imgcourse::Rng;
use crate::marked_patterns::*;
use std::f64::consts::PI;

const A: f64 = 200.0;
const B: f64 = 200.0;

/// Split a pattern by its labels.
fn split(points: &[Point], labels: &[u8]) -> (Vec<Point>, Vec<Point>) {
    let t0 = points.iter().zip(labels).filter(|(_, &l)| l == 0).map(|(p, _)| *p).collect();
    let t1 = points.iter().zip(labels).filter(|(_, &l)| l == 1).map(|(p, _)| *p).collect();
    (t0, t1)
}

/// Cross L(r) - r (type 1 around type 0, translation correction) at r = 2, 4, .., 40 µm.
fn cross_l(t0: &[Point], t1: &[Point]) -> Vec<f64> {
    (1..=20).map(|k| 2.0 * k as f64).map(|r| (cross_k(t0, t1, A, B, r, true) / PI).sqrt() - r).collect()
}

/// The MAD test against the mean of the simulated curves, with 39 simulations from `null`.
/// Returns (observed L(10) - 10, p).
fn test(points: &[Point], labels: &[u8], null: &str, rng: &mut Rng) -> (f64, f64) {
    let (t0, t1) = split(points, labels);
    let observed = cross_l(&t0, &t1);
    let sims: Vec<Vec<f64>> = (0..39)
        .map(|_| match null {
            // relocation: keep type 0, put the type-1 cells anywhere in the window
            "relocate" => cross_l(&t0, &csr(t1.len(), A, B, rng)),
            // random labelling: keep all positions, permute the labels
            _ => {
                let (s0, s1) = split(points, &permute_labels(labels, rng));
                cross_l(&s0, &s1)
            }
        })
        .collect();
    let mean: Vec<f64> = (0..20).map(|k| sims.iter().map(|c| c[k]).sum::<f64>() / 39.0).collect();
    let mad = |c: &[f64]| c.iter().zip(&mean).fold(0.0f64, |m, (v, mu)| m.max((v - mu).abs()));
    let t0 = mad(&observed);
    let extreme = sims.iter().filter(|c| mad(c) >= t0).count();
    (observed[4], (1 + extreme) as f64 / 40.0)
}

pub fn main() {
    let mut rng = Rng::new(3);
    // attraction: tumour nests; half of the immune cells next to a tumour cell, half anywhere
    let tumour = thomas(0.0006, 10.0, 6.0, A, B, &mut rng);
    let mut immune = csr(45, A, B, &mut rng);
    for _ in 0..45 {
        let t = tumour[rng.below(tumour.len())];
        let p = (t.0 + 4.0 * rng.normal() as f64, t.1 + 4.0 * rng.normal() as f64);
        if (0.0..A).contains(&p.0) && (0.0..B).contains(&p.1) {
            immune.push(p);
        }
    }
    let attract: Vec<Point> = [tumour.clone(), immune].concat();
    let attract_labels: Vec<u8> = (0..attract.len()).map(|k| (k >= tumour.len()) as u8).collect();
    // independent: the same tumour nests, immune cells anywhere
    let independent: Vec<Point> = [tumour.clone(), csr(90, A, B, &mut rng)].concat();
    let independent_labels: Vec<u8> = (0..independent.len()).map(|k| (k >= tumour.len()) as u8).collect();
    // packed tissue: cells at least 7.5 µm apart, 30 % immune at random
    let packed = hard_core(220, 7.5, A, B, &mut rng);
    let packed_labels: Vec<u8> = (0..packed.len()).map(|_| (rng.uniform() < 0.3) as u8).collect();
    // segregated: the same packed cells, immune to the right of a wavy line
    let segregated_labels: Vec<u8> = packed.iter().map(|p| (p.0 > 120.0 + 30.0 * (p.1 / 25.0).sin()) as u8).collect();

    println!("cross L(10) - 10 (immune around tumour) and the MAD p-value (r = 2 .. 40 µm, 39 simulations):");
    println!("{:>34} {:>7} {:>10} {:>10}", "", "L - r", "relocate", "permute");
    let scenarios = [
        ("attraction", &attract, &attract_labels),
        ("independent", &independent, &independent_labels),
        ("packed tissue, random types", &packed, &packed_labels),
        ("packed tissue, segregated", &packed, &segregated_labels),
    ];
    for (name, pts, labels) in scenarios {
        let (l, p_reloc) = test(pts, labels, "relocate", &mut rng);
        let (_, p_perm) = test(pts, labels, "permute", &mut rng);
        let n1 = labels.iter().filter(|&&l| l == 1).count();
        println!("{:>34} {l:7.2} {p_reloc:10.3} {p_perm:10.3}", format!("{name} ({}+{})", pts.len() - n1, n1));
    }

    println!();
    println!("mark connection function p01(r) (h = 2 µm) of the packed tissue:");
    for (name, labels) in [("random types", &packed_labels), ("segregated", &segregated_labels)] {
        let p1 = labels.iter().filter(|&&l| l == 1).count() as f64 / labels.len() as f64;
        let values: Vec<String> = [8.0, 10.0, 20.0, 40.0].iter().map(|&r| format!("{:.2}", mark_connection(&packed, labels, r, 2.0))).collect();
        println!("  {name:>12}: 2 p0 p1 = {:.2}; at r = 8, 10, 20, 40 µm: {}", 2.0 * p1 * (1.0 - p1), values.join(", "));
    }

    // an uneven density without interaction: CSR thinned with probability x / A
    let gradient: Vec<Point> = csr(600, A, B, &mut rng).into_iter().filter(|p| (rng.uniform() as f64) < p.0 / A).collect();
    println!();
    println!("gradient ({} cells, no interaction): L(r) - r", gradient.len());
    println!("{:>24} {}", "r =", [5.0, 10.0, 20.0, 30.0].map(|r| format!("{r:>7}")).join(""));
    let plain: Vec<String> = [5.0, 10.0, 20.0, 30.0].iter().map(|&r| {
        let n = gradient.len() as f64;
        let mut s = 0.0;
        for (i, &p) in gradient.iter().enumerate() {
            for (j, &q) in gradient.iter().enumerate() {
                if i != j && dist(p, q) <= r {
                    s += translation_weight(q.0 - p.0, q.1 - p.1, A, B);
                }
            }
        }
        format!("{:7.2}", (A * B / (n * (n - 1.0)) * s / PI).sqrt() - r)
    }).collect();
    println!("{:>24} {}", "plain K", plain.join(""));
    for sigma in [20.0, 40.0, 200.0] {
        let v: Vec<String> = [5.0, 10.0, 20.0, 30.0].iter().map(|&r| format!("{:7.2}", (k_inhom(&gradient, A, B, r, sigma) / PI).sqrt() - r)).collect();
        println!("{:>24} {}", format!("inhomogeneous, σ = {sigma}"), v.join(""));
    }
}
