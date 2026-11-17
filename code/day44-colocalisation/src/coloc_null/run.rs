// Shift curves, block randomisation and the ICQ on synthetic two-channel images;
// pictures in out/coloc-null/.
//   cargo run -p day44 -- coloc-null                      (your code)
//   cargo run -p day44 --features solution -- coloc-null  (the model solution)

use crate::coloc_null::sim::{self, Scene};
use crate::coloc_null::*;
use imgcourse::*;

/// The half width at half height of a peak at the middle of `curve`, above the level `floor`
/// (linear interpolation between the samples, on both sides, averaged).
fn half_width(curve: &[f32], floor: f32) -> f32 {
    let mid = curve.len() / 2;
    let half = floor + 0.5 * (curve[mid] - floor);
    let side = |dir: isize| {
        let mut i = mid as isize;
        while (i + dir) >= 0 && ((i + dir) as usize) < curve.len() && curve[(i + dir) as usize] > half {
            i += dir;
        }
        let (r1, r2) = (curve[i as usize], curve[(i + dir).clamp(0, curve.len() as isize - 1) as usize]);
        (i - mid as isize).abs() as f32 + if r1 > r2 { (r1 - half) / (r1 - r2) } else { 0.0 }
    };
    0.5 * (side(-1) + side(1))
}

/// Pearson's r of A against `n` block-shuffled copies of B (fixed seed).
fn null_r(a: &Image<f32>, b: &Image<f32>, mask: &Image<bool>, block: usize, n: usize) -> Vec<f32> {
    let mut rng = Rng::new(2024);
    (0..n).map(|_| sim::pearson(a, &block_shuffle(b, mask, block, &mut rng), mask)).collect()
}

pub fn main() {
    let dir = "out/coloc-null";
    let base = Scene::default();

    // 1. Shift curves
    println!("shift curves, r at dx = -10 ... 10:");
    for (name, scene, region) in [
        ("colocalised", Scene { shared: 1.0, ..base.clone() }, "cells"),
        ("shifted 2.4 px", Scene { shared: 1.0, shift: (2.4, 0.0), ..base.clone() }, "cells"),
        ("independent + diffuse", Scene { shared: 0.0, diffuse: 0.15, ..base.clone() }, "image"),
    ] {
        let (a, b, cells) = sim::scene(&scene);
        let mask = if region == "cells" { cells } else { sim::everywhere(a.width(), a.height()) };
        let curve = shift_curve(&a, &b, &mask, 10);
        let line: Vec<String> = curve.iter().map(|r| format!("{r:.2}")).collect();
        println!("  {name} ({region}): {}", line.join(" "));
        let best = (0..curve.len()).max_by(|&i, &j| curve[i].total_cmp(&curve[j])).unwrap();
        let peak = if best > 0 && best + 1 < curve.len() {
            best as f32 - 10.0 + subpixel_peak(curve[best - 1], curve[best], curve[best + 1])
        } else {
            best as f32 - 10.0
        };
        let floor = 0.5 * (curve[0] + curve[curve.len() - 1]);
        println!("    r(0) = {:.3}, peak at {peak:.2} px (r = {:.3}), r at ±10 px: {floor:.3}, half width above that: {:.1} px", curve[10], curve[best], half_width(&curve, floor));
    }

    // 2. Block randomisation: independent structures, blurred with sigma 2 px
    let indep = Scene { shared: 0.0, blur: 2.0, seed: 23, ..base.clone() };
    let (a, b, cells) = sim::scene(&indep);
    let r_obs = sim::pearson(&a, &b, &cells);
    println!("\nrandomisation, independent channels (blur 2 px), r = {r_obs:.3}, 200 shuffles:");
    for block in [1, 2, 4, 8, 16] {
        let null = null_r(&a, &b, &cells, block, 200);
        let (m, sd) = sim::mean_sd(&null);
        let reach = null.iter().filter(|&&r| r >= r_obs).count();
        println!("  blocks of {block:>2} px: null mean {m:+.3}, sd {sd:.3}, {reach:>3} of 200 >= observed, p = {:.3}", permutation_p(r_obs, &null));
    }
    let mut rng = Rng::new(3);
    save_rgb(&sim::merge(&a, &block_shuffle(&b, &cells, 1, &mut rng)), format!("{dir}/shuffled-1px.png"));
    save_rgb(&sim::merge(&a, &block_shuffle(&b, &cells, 8, &mut rng)), format!("{dir}/shuffled-8px.png"));
    save_rgb(&sim::merge(&a, &b), format!("{dir}/independent.png"));

    let leak = Scene { leak: 0.4, ..indep.clone() };
    let (a, b, cells) = sim::scene(&leak);
    let r_leak = sim::pearson(&a, &b, &cells);
    let null = null_r(&a, &b, &cells, 8, 200);
    println!("  with 40 % bleed-through: r = {r_leak:.3}, 8-px blocks: p = {:.3}", permutation_p(r_leak, &null));

    // 3. PDM and ICQ
    println!("\nICQ (observed, and mean ± sd of 200 shuffles in 8-px blocks):");
    for (name, scene) in [("colocalised", Scene { shared: 1.0, ..base.clone() }), ("independent", Scene { shared: 0.0, ..base.clone() })] {
        let (a, b, cells) = sim::scene(&scene);
        let mut rng = Rng::new(11);
        let shuffled: Vec<f32> = (0..200).map(|_| icq(&a, &block_shuffle(&b, &cells, 8, &mut rng), &cells)).collect();
        let (m, sd) = sim::mean_sd(&shuffled);
        println!("  {name}: r = {:.3}, ICQ = {:.3}, shuffled ICQ {m:.3} ± {sd:.3}", sim::pearson(&a, &b, &cells), icq(&a, &b, &cells));
        save_gray_signed(&pdm(&a, &b, &cells), format!("{dir}/pdm-{name}.png"));
    }
    println!("wrote the pictures to {dir}/");
}
