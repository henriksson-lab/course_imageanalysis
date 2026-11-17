// A noisy disk with a thin arm: thresholding against dynamic programming along the rows,
// for several λ. Prints wrong pixels and energies; pictures in out/seg-energy/.
//   cargo run -p day27 -- seg-energy                      (your code)
//   cargo run -p day27 --features solution -- seg-energy  (the model solution)

use imgcourse::*;
use crate::common::disk_with_arm;
use crate::seg_energy::*;

const SIZE: usize = 128;

pub fn main() {
    let dir = "out/seg-energy";
    let (mu0, mu1, s) = (80.0 / 255.0, 170.0 / 255.0, 50.0 / 255.0);
    let truth = disk_with_arm(SIZE);
    let clean = truth.map(|l| if l == 1 { mu1 } else { mu0 });
    let img = samples::add_noise(&clean, s, 3);
    save_gray(&img, format!("{dir}/input.png"));
    let (d0, d1) = data_costs(&img, mu0, mu1, s);
    let wrong = |l: &Image<u32>| l.pixels().iter().zip(truth.pixels()).filter(|(a, b)| a != b).count();
    let to_gray = |l: &Image<u32>| l.map(|v| v as f32);
    let boundaries = |l: &Image<u32>| (energy(l, &d0, &d1, 1.0) - energy(l, &d0, &d1, 0.0)).round();

    let thr = threshold(&d0, &d1);
    save_gray(&to_gray(&thr), format!("{dir}/threshold.png"));
    println!("{SIZE} × {SIZE} image, μ0 = 80, μ1 = 170, noise s = 50 (grey levels of 255)");
    println!("labelling                   wrong pixels  boundaries   data      E (λ = 1.5)");
    let row = |name: &str, l: &Image<u32>| {
        println!(
            "{name:27} {:8}     {:8}   {:8.1}   {:8.1}",
            wrong(l),
            boundaries(l),
            energy(l, &d0, &d1, 0.0),
            energy(l, &d0, &d1, 1.5)
        );
    };
    row("truth", &truth);
    row("threshold (λ = 0)", &thr);
    for lambda in [1.5, 4.0, 10.0] {
        let l = rows_dp(&d0, &d1, lambda);
        row(&format!("rows by DP, λ = {lambda}"), &l);
        save_gray(&to_gray(&l), format!("{dir}/rows-dp-lambda{lambda}.png"));
    }

    // One row through the disk, as a chain (the lecture's line profile).
    let y = 70;
    println!("\nrow {y}, as a chain: wrong samples / boundaries");
    for lambda in [0.0, 1.5, 4.0, 10.0, 100.0] {
        let (l, e) = chain_dp(d0.row(y), d1.row(y), lambda);
        let wrong = (0..SIZE).filter(|&x| l[x] != truth[(x, y)]).count();
        let b = (1..SIZE).filter(|&x| l[x] != l[x - 1]).count();
        println!("  λ = {lambda:5}: {wrong:3} wrong, {b:3} boundaries, E = {e:.1}");
    }
    println!("\nwrote {dir}/input.png, threshold.png and rows-dp-lambda*.png (white = object)");
}
