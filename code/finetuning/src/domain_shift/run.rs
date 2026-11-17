// A classifier trained on microscope A, used on microscope B with a shifted intensity scale.
// Accuracy on B against the shift for four remedies, with B's share of mitotic cells as in A
// (one half) and lower (one fifth).
//   cargo run -p finetuning -- domain-shift                      (your code)
//   cargo run -p finetuning --features solution -- domain-shift  (the model solution)

use crate::domain_shift::*;

pub fn main() {
    let a = cells_a(400, 1, 0.5);
    let (ma, sa) = stats(&a);
    println!("microscope A: mean intensity {:.3} (sd {:.3}), mean size {:.3} (sd {:.3})", ma[0], sa[0], ma[1], sa[1]);
    let plain = Linear::fit(&a, 1e-3, [0.0; 3]);
    let normed = Linear::fit(&standardise(&a, ma, sa), 1e-3, [0.0; 3]);
    let augmented = Linear::fit(&augment(&a, 3, 3), 1e-3, [0.0; 3]);
    println!("accuracy on A: plain {:.1} %, trained with augmentation {:.1} %", 100.0 * plain.accuracy(&a), 100.0 * augmented.accuracy(&a));
    println!("weights (constant, intensity, size): plain {:.1?}, augmented {:.1?}", plain.w, augmented.w);
    for frac in [0.5, 0.2] {
        println!("\nmicroscope B, mitotic share {frac}: accuracy on 400 B cells");
        println!("  shift   score    as is   normalised   augmented   fine-tuned (10 B cells)");
        for s in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let b = to_b(&cells_a(400, 2, frac), s);
            let (mb, sb) = stats(&b);
            // 10 labelled B cells, 5 of each class, from another draw
            let extra = to_b(&cells_a(200, 12, 0.5), s);
            let mut tune: Vec<Cell> = extra.iter().filter(|c| c.class == 0).take(5).copied().collect();
            tune.extend(extra.iter().filter(|c| c.class == 1).take(5));
            let tuned = Linear::fit(&tune, 0.05, plain.w);
            println!(
                "  {s:5.2}   {:5.2}   {:5.1} %   {:8.1} %   {:7.1} %   {:8.1} %",
                shift_score(&a, &b),
                100.0 * plain.accuracy(&b),
                100.0 * normed.accuracy(&standardise(&b, mb, sb)),
                100.0 * augmented.accuracy(&b),
                100.0 * tuned.accuracy(&b)
            );
        }
    }
}
