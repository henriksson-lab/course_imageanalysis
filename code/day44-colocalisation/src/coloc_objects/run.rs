// Object-based colocalisation on simulated spot positions in two cells (pixel 65 nm);
// pictures in out/coloc-objects/.
//   cargo run -p day44 -- coloc-objects                      (your code)
//   cargo run -p day44 --features solution -- coloc-objects  (the model solution)

use crate::coloc_objects::sim::{self, Point};
use crate::coloc_objects::*;
use imgcourse::*;
use std::f32::consts::PI;

/// L(r) − r for every radius.
fn l_minus_r(a: &[Point], b: &[Point], area: f32, radii: &[f32]) -> Vec<f32> {
    radii.iter().map(|&r| (cross_k(a, b, area, r) / PI).sqrt() - r).collect()
}

pub fn main() {
    let dir = "out/coloc-objects";
    let cells = sim::cells(128);
    let area = sim::area(&cells);
    let d_max = 130.0;
    println!("two cells of {:.1} µm², d_max = {d_max} nm, localisation precision 20 nm per axis", area / 1e6);

    // 1. Distances, one-to-one, chance level
    println!("\n{:<26} {:>4} {:>4} {:>9} {:>8} {:>9} {:>10}", "scene", "n_A", "n_B", "observed", "chance", "corrected", "one-to-one");
    let scenes = [
        ("pairs at 0 nm", 40, 0.6, 0.0, 8),
        ("independent, sparse", 40, 0.0, 0.0, 12),
        ("independent, dense", 40, 0.0, 0.0, 120),
        ("pairs at 300 nm", 40, 1.0, 300.0, 0),
    ];
    for (k, &(name, n_a, paired, distance, extra)) in scenes.iter().enumerate() {
        let (a, b) = sim::spots(&cells, n_a, paired, distance, extra, 20.0, 7);
        let f = fraction_within(&a, &b, d_max);
        let chance = chance_level(b.len() as f32 / area, d_max);
        let pairs = one_to_one(&a, &b, d_max) as f32 / a.len() as f32;
        println!("{name:<26} {:>4} {:>4} {:>8.1}% {:>7.1}% {:>8.1}% {:>9.1}%", a.len(), b.len(), 100.0 * f, 100.0 * chance, 100.0 * corrected(f, chance), 100.0 * pairs);
        save_rgb(&sim::picture(&cells, &a, &b, d_max, 4), format!("{dir}/scene{}.png", k + 1));
    }
    let (a, b) = sim::spots(&cells, 40, 0.0, 0.0, 120, 20.0, 7);
    let f = fraction_within(&a, &b, 260.0);
    println!("independent, dense, d_max = 260 nm: observed {:.1}%, chance {:.1}%", 100.0 * f, 100.0 * chance_level(b.len() as f32 / area, 260.0));

    // 2. Cross-L with an envelope of 39 relocations of the B spots
    let radii: Vec<f32> = (1..=10).map(|k| 50.0 * k as f32).collect();
    println!("\nL(r) − r in nm, r = {radii:?}");
    for (name, paired, distance, extra) in [("pairs at 0 nm", 1.0, 0.0, 0), ("pairs at 300 nm", 1.0, 300.0, 0), ("independent", 0.0, 0.0, 40)] {
        let (a, b) = sim::spots(&cells, 40, paired, distance, extra, 20.0, 7);
        let observed = l_minus_r(&a, &b, area, &radii);
        let whole = Image::new(128, 128, true);
        for (label, region) in [("inside the cells", &cells), ("whole image", &whole)] {
            let mut rng = Rng::new(99);
            let mut lo = vec![f32::INFINITY; radii.len()];
            let mut hi = vec![f32::NEG_INFINITY; radii.len()];
            for _ in 0..39 {
                let moved = relocate(b.len(), region, &mut rng);
                for (i, v) in l_minus_r(&a, &moved, area, &radii).into_iter().enumerate() {
                    lo[i] = lo[i].min(v);
                    hi[i] = hi[i].max(v);
                }
            }
            let mark: String = observed.iter().zip(lo.iter().zip(&hi)).map(|(o, (l, h))| if o > h { '+' } else if o < l { '-' } else { '.' }).collect();
            println!("  {name:<16} envelope {label:<17}: {mark}   (+ above, - below, . inside)");
        }
        let line: Vec<String> = observed.iter().map(|v| format!("{v:.0}")).collect();
        println!("  {name:<16} observed: {}", line.join(" "));
    }

    // 3. The distance bias: many pairs, localisation precision 20 nm per axis in each channel
    println!("\nmeasured centroid distances of 2000 pairs, precision 20 nm per axis and channel:");
    for d in [0.0, 60.0, 200.0] {
        let (a, b) = sim::spots(&Image::new(128, 128, true), 2000, 1.0, d, 0, 20.0, 5);
        let mean = a.iter().zip(&b).map(|(&p, &q)| sim::dist(p, q)).sum::<f32>() / 2000.0;
        println!("  true {d:>5.0} nm: mean measured {mean:.1} nm");
    }
    println!("wrote the pictures to {dir}/ (scene1 … scene4: A green dots, B magenta rings, yellow links within d_max)");
}
