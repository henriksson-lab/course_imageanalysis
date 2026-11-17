// A synthetic two-colour cell with bleed-through: estimate the spillover from the nucleus
// (a single-stained structure), compensate, and check under-, over- and exact compensation.
// Pictures in out/crosstalk/.
//   cargo run -p unmixing -- crosstalk                      (your code)
//   cargo run -p unmixing --features solution -- crosstalk  (the model solution)

use crate::crosstalk::*;
use imgcourse::*;

/// The cell: true amounts of dye 1 (nucleus, vesicles) and dye 2 (membrane, vesicles), and
/// masks of the nucleus, the membrane, the whole cell and the background.
struct Cell {
    dye1: Image<f32>,
    dye2: Image<f32>,
    nucleus: Image<bool>,
    membrane: Image<bool>,
    inside: Image<bool>,
    background: Image<bool>,
}

fn cell(size: usize) -> Cell {
    let c = size as f32 / 2.0;
    let r = |x: usize, y: usize, cx: f32, cy: f32| ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
    let vesicles = [(60.0, 70.0), (130.0, 60.0), (140.0, 125.0), (70.0, 135.0), (100.0, 150.0), (55.0, 100.0)];
    let in_vesicle = |x: usize, y: usize| vesicles.iter().any(|&(vx, vy)| r(x, y, vx, vy) < 4.0);
    let nucleus = Image::from_fn(size, size, |x, y| r(x, y, c + 5.0, c - 5.0) < 30.0);
    let membrane = Image::from_fn(size, size, |x, y| (66.0..72.0).contains(&r(x, y, c, c)));
    let inside = Image::from_fn(size, size, |x, y| r(x, y, c, c) < 72.0);
    let background = Image::from_fn(size, size, |x, y| r(x, y, c, c) > 80.0);
    let dye1 = Image::from_fn(size, size, |x, y| {
        if nucleus[(x, y)] {
            // some texture: from about 150 to 450
            300.0 + 150.0 * ((x as f32 / 6.0).sin() * (y as f32 / 7.0).cos())
        } else if in_vesicle(x, y) {
            120.0
        } else {
            0.0
        }
    });
    let dye2 = Image::from_fn(size, size, |x, y| if membrane[(x, y)] { 300.0 } else if in_vesicle(x, y) { 120.0 } else { 0.0 });
    Cell { dye1, dye2, nucleus, membrane, inside, background }
}

/// Record the two channels: offset + S · (dye amounts), with photon noise.
fn record(cell: &Cell, s: Mat2, offset: f32, seed: u64) -> (Image<f32>, Image<f32>) {
    let mut rng = Rng::new(seed);
    let mut ch1 = Image::new(cell.dye1.width(), cell.dye1.height(), 0.0);
    let mut ch2 = ch1.clone();
    for (x, y) in ch1.coords() {
        let (a1, a2) = (cell.dye1[(x, y)], cell.dye2[(x, y)]);
        ch1[(x, y)] = rng.poisson(offset + s[0][0] * a1 + s[0][1] * a2);
        ch2[(x, y)] = rng.poisson(offset + s[1][0] * a1 + s[1][1] * a2);
    }
    (ch1, ch2)
}

fn values(img: &Image<f32>, mask: &Image<bool>) -> Vec<f32> {
    img.coords().filter(|&p| mask[p]).map(|p| img[p]).collect()
}
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}
fn sd(v: &[f32]) -> f32 {
    let m = v.iter().sum::<f32>() / v.len() as f32;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
}

pub fn main() {
    let dir = "out/crosstalk";
    let size = 192;
    let offset = 10.0;
    let true_s = [[1.0, 0.05], [0.25, 1.0]];
    let cell = cell(size);
    let (ch1, ch2) = record(&cell, true_s, offset, 1);
    save_gray_stretched(&ch1, format!("{dir}/channel1.png"));
    save_gray_stretched(&ch2, format!("{dir}/channel2.png"));

    // Spillover from single-stained structures, after subtracting the offset.
    let (y1, y2) = (ch1.map(|v| v - offset), ch2.map(|v| v - offset));
    let s21 = slope_through_origin(&values(&y1, &cell.nucleus), &values(&y2, &cell.nucleus));
    let s12 = slope_through_origin(&values(&y2, &cell.membrane), &values(&y1, &cell.membrane));
    println!("estimated spillover: s21 = {s21:.3}, s12 = {s12:.3} (true 0.25 and 0.05)");
    let s21_raw = slope_through_origin(&values(&ch1, &cell.nucleus), &values(&ch2, &cell.nucleus));
    println!("  s21 without subtracting the offset: {s21_raw:.3}");

    // Compensate with the estimate.
    let (a1, a2) = compensate(&y1, &y2, [[1.0, s12], [s21, 1.0]]);
    save_gray_stretched(&a2, format!("{dir}/dye2-compensated.png"));
    println!("Pearson's r inside the cell: {:.2} measured, {:.2} compensated", pearson(&ch1, &ch2, &cell.inside), pearson(&a1, &a2, &cell.inside));

    // Under, right, over: the nucleus's median in the dye-2 image, and its slope.
    for g in [0.15, 0.25, 0.35] {
        let (b1, b2) = compensate(&y1, &y2, [[1.0, 0.05], [g, 1.0]]);
        let slope = slope_through_origin(&values(&b1, &cell.nucleus), &values(&b2, &cell.nucleus));
        println!(
            "guess s21 = {g}: nucleus median in dye 2 = {:6.1} (background {:4.1}), slope {slope:+.3}",
            median(values(&b2, &cell.nucleus)),
            median(values(&b2, &cell.background))
        );
        if g == 0.35 {
            save_gray_stretched(&b2, format!("{dir}/dye2-over.png"));
        }
    }

    // Similar dyes: the exact compensation amplifies the noise.
    let similar = [[1.0, 0.9], [0.9, 1.0]];
    let (c1, c2) = record(&cell, similar, offset, 2);
    let (z1, z2) = (c1.map(|v| v - offset), c2.map(|v| v - offset));
    let (_, d2) = compensate(&z1, &z2, similar);
    save_gray_stretched(&d2, format!("{dir}/similar-dye2.png"));
    println!(
        "similar dyes (0.9, 0.9): background noise sd {:.1} in channel 2, {:.1} after compensation",
        sd(&values(&z2, &cell.background)),
        sd(&values(&d2, &cell.background))
    );
    println!("wrote the pictures to {dir}/");
}
