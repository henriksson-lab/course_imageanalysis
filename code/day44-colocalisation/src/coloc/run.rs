// Runs your coefficients on synthetic two-channel images; pictures in out/coloc/.
//   cargo run -p day44 -- coloc                      (your code)
//   cargo run -p day44 --features solution -- coloc  (the model solution)

use crate::common::{self, Scene};
use crate::coloc::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/coloc";
    let base = Scene::default();
    let scenes = [
        ("perfect", Scene { shared: 1.0, ..base.clone() }),
        ("independent", Scene { shared: 0.0, ..base.clone() }),
        ("leak", Scene { shared: 0.0, leak: 0.4, ..base.clone() }),
        ("a-inside-b", Scene { a_inside_b: true, ..base.clone() }),
        ("diffuse", Scene { shared: 0.0, diffuse: 0.15, ..base.clone() }),
    ];
    println!("{:<12} {:>7} {:>7} {:>8} {:>8} {:>13} {:>6} {:>6}", "scene", "r cells", "r image", "overlap", "Spearman", "tA, tB", "M1", "M2");
    for (name, scene) in &scenes {
        let (a, b, cells) = common::scene(scene);
        let whole = common::everywhere(a.width(), a.height());
        let r = pearson(&a, &b, &cells);
        let r_whole = pearson(&a, &b, &whole);
        let (ov, _, _) = overlap(&a, &b, &cells);
        let rho = spearman(&a, &b, &cells);
        // Costes' thresholds, or the 75th percentiles if Costes is not defined (marked *)
        let (t, mark) = match costes(&a, &b, &cells, 256) {
            Some(t) => (t, ' '),
            None => ((common::percentile(&a, &cells, 0.75), common::percentile(&b, &cells, 0.75)), '*'),
        };
        let (m1, m2) = manders(&a, &b, &cells, t.0, t.1);
        println!("{name:<12} {r:>7.3} {r_whole:>7.3} {ov:>8.3} {rho:>8.3} {:>6.3},{:>5.3}{mark} {m1:>6.3} {m2:>6.3}", t.0, t.1);
        save_rgb(&common::merge(&a, &b), format!("{dir}/{name}-merge.png"));
        save_rgb(&common::scatter(&a, &b, &cells, 128, Some(t)), format!("{dir}/{name}-scatter.png"));
    }
    // Without thresholds every pixel is "above": M1 and M2 of independent channels.
    let (a, b, cells) = common::scene(&scenes[1].1);
    let (m1, m2) = manders(&a, &b, &cells, 0.0, 0.0);
    println!("independent, thresholds 0: M1 = {m1:.3}, M2 = {m2:.3}");
    println!("(* = Costes not defined, 75th percentiles used)");
    println!("wrote the merges and scatter plots to {dir}/");
}
