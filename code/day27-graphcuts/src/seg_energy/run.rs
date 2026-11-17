// A noisy disk with a thin arm: thresholding against dynamic programming along the rows,
// for several λ. Prints wrong pixels and energies; pictures in out/seg-energy/.
//   cargo run -p day27 -- seg-energy                      (your code)
//   cargo run -p day27 --features solution -- seg-energy  (the model solution)

use imgcourse::*;
use crate::seg_energy::*;

const SIZE: usize = 128;

/// The true labels: a disk of radius 30 and an arm 3 pixels wide running up to the right.
fn truth() -> Image<u32> {
    Image::from_fn(SIZE, SIZE, |x, y| {
        let (dx, dy) = (x as f32 - 54.0, y as f32 - 70.0);
        let disk = dx * dx + dy * dy <= 30.0 * 30.0;
        // distance to the line through the centre with direction (1, −1)/√2, beyond the disk
        let along = (dx - dy) / 2f32.sqrt();
        let across = (dx + dy) / 2f32.sqrt();
        let arm = along > 0.0 && along < 85.0 && across.abs() <= 1.5;
        if disk || arm { 1 } else { 0 }
    })
}

pub fn main() {
    let dir = "out/seg-energy";
    let (mu0, mu1, s) = (80.0 / 255.0, 170.0 / 255.0, 50.0 / 255.0);
    let truth = truth();
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
