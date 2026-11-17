// The confounds of the microscope and their corrections, in order, on a synthetic scene;
// pictures in out/coloc-confounds/.
//   cargo run -p day44 -- coloc-confounds                      (your code)
//   cargo run -p day44 --features solution -- coloc-confounds  (the model solution)

use crate::coloc_confounds::sim;
use crate::common::{self, Scene};
use crate::coloc_confounds::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/coloc-confounds";
    // Half of B's structures sit on A's. The microscope adds: a chromatic shift of 1.5 px
    // (about 100 nm at 65 nm pixels), 30 % bleed-through of A into B, a shared uneven background.
    let ideal = Scene { shared: 0.5, ..Scene::default() };
    let measured = Scene { shift: (1.5, 0.0), leak: 0.3, background: 0.25, ..ideal.clone() };
    let (a0, b0, cells) = common::scene(&ideal);
    let (a, b, _) = common::scene(&measured);
    let r = |a: &Image<f32>, b: &Image<f32>| common::pearson(a, b, &cells);
    println!("ideal microscope (same structures, no confounds): r = {:.3}", r(&a0, &b0));
    println!("measured:                                          r = {:.3}", r(&a, &b));

    // The single-stained control (only dye A, no haze) gives the leak: subtract its offset, the
    // median outside the cells, then fit over the cells. The beads give the shift.
    let (ac, bc, _) = common::scene(&Scene { b_bright: 0.0, background: 0.0, seed: 77, ..measured.clone() });
    let outside = cells.map(|c| !c);
    let (oa, ob) = (common::percentile(&ac, &outside, 0.5), common::percentile(&bc, &outside, 0.5));
    let s = leak_slope(&ac.map(|v| v - oa), &bc.map(|v| v - ob), &cells);
    let (beads_a, beads_b) = sim::beads(&measured, 25, measured.shift);
    let (tx, ty) = estimate_shift(&beads_a, &beads_b, 4);
    println!("from the controls: leak ŝ = {s:.3} (true 0.3), bead shift = ({tx:.2}, {ty:.2}) px (true (1.5, 0))");

    // The corrections in order.
    let (a1, b1) = (subtract_background(&a, 5), subtract_background(&b, 5));
    println!("1. background subtracted:                          r = {:.3}", r(&a1, &b1));
    let b2 = compensate(&b1, &a1, s);
    println!("2. compensated:                                    r = {:.3}", r(&a1, &b2));
    let b3 = shift_image(&b2, tx, ty);
    println!("3. registered:                                     r = {:.3}", r(&a1, &b3));

    // The wrong order, and wrong factors.
    let b_reg_first = compensate(&shift_image(&b1, tx, ty), &a1, s);
    println!("registered before compensating:                    r = {:.3}", r(&a1, &b_reg_first));
    let b_no_bg = shift_image(&compensate(&b, &a, s), tx, ty);
    println!("compensated and registered without background:     r = {:.3}", r(&a, &b_no_bg));
    for s_wrong in [0.15, 0.6] {
        let bw = shift_image(&compensate(&b1, &a1, s_wrong), tx, ty);
        println!("with ŝ = {s_wrong}:                                     r = {:.3}", r(&a1, &bw));
    }

    // The resolution: overlap of two spots at 520 nm (sigma 78 nm) and 670 nm (100 nm)
    print!("\nGaussian overlap of a 78 nm and a 100 nm spot at distance");
    for d in [0.0, 50.0, 100.0, 150.0, 200.0, 300.0] {
        print!("  {d:.0} nm: {:.2}", gaussian_overlap(78.0, 100.0, d));
    }
    println!();

    save_rgb(&common::merge(&a0, &b0), format!("{dir}/ideal.png"));
    save_rgb(&common::merge(&a, &b), format!("{dir}/measured.png"));
    save_rgb(&common::merge(&a1, &b1), format!("{dir}/1-background.png"));
    save_rgb(&common::merge(&a1, &b2), format!("{dir}/2-compensated.png"));
    save_rgb(&common::merge(&a1, &b3), format!("{dir}/3-registered.png"));
    println!("wrote the merges to {dir}/");
}
