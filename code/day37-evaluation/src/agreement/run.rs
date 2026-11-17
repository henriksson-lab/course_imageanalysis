// Boundary distances against IoU for discs of different sizes and for a thin spike, and
// Bland–Altman against correlation for simulated cell counts. Results in out/agreement/.
//   cargo run -p day37 -- agreement                      (your code)
//   cargo run -p day37 --features solution -- agreement  (the model solution)

use crate::agreement::*;
use imgcourse::*;

/// A disc of radius `r` around the centre of a `size × size` image.
fn disc(size: usize, r: f32) -> Image<bool> {
    let c = size as f32 / 2.0;
    Image::from_fn(size, size, |x, y| (x as f32 - c).powi(2) + (y as f32 - c).powi(2) <= r * r)
}

fn iou(a: &Image<bool>, b: &Image<bool>) -> f32 {
    let (mut both, mut either) = (0, 0);
    for (x, y) in a.coords() {
        both += (a[(x, y)] && b[(x, y)]) as usize;
        either += (a[(x, y)] || b[(x, y)]) as usize;
    }
    both as f32 / either as f32
}

fn area(a: &Image<bool>) -> f32 {
    a.pixels().iter().filter(|&&v| v).count() as f32
}

pub fn main() {
    let dir = "out/agreement";

    println!("A disc and the same disc drawn 1 pixel larger:");
    println!("  radius   IoU   Hausdorff  HD95   ASD   area ratio");
    let mut true_areas = Vec::new();
    let mut auto_areas = Vec::new();
    for r in [3.0, 5.0, 8.0, 15.0, 30.0, 40.0] {
        let (t, p) = (disc(100, r), disc(100, r + 1.0));
        let s = boundary_scores(&boundary(&t), &boundary(&p));
        println!("  {r:5.0}   {:.3}   {:.2}      {:.2}   {:.2}   {:.2}", iou(&t, &p), s.hausdorff, s.hd95, s.asd, area(&p) / area(&t));
        true_areas.push(area(&t));
        auto_areas.push(area(&p));
    }

    // A thin spike, 3 pixels wide, sticking out of a disc of radius 30.
    let t = disc(140, 30.0); // centre (70, 70), right edge at x = 100
    println!("\nA disc of radius 30 with a spike 3 px wide:");
    println!("  spike   IoU   Hausdorff  HD95   ASD");
    for length in [7, 14, 28] {
        let mut p = t.clone();
        for x in 100..100 + length {
            for y in 69..72 {
                p[(x, y)] = true;
            }
        }
        let s = boundary_scores(&boundary(&t), &boundary(&p));
        println!("  {length:5}   {:.3}   {:5.2}     {:.2}   {:.2}", iou(&t, &p), s.hausdorff, s.hd95, s.asd);
        if length == 14 {
            let mut pic = Image::new(140, 140, [0.0f32; 3]);
            for (x, y) in boundary(&p) {
                pic[(x as usize, y as usize)] = [0.3, 0.5, 1.0];
            }
            for (x, y) in boundary(&t) {
                pic[(x as usize, y as usize)] = [1.0, 1.0, 1.0];
            }
            let big = Image::from_fn(280, 280, |x, y| pic[(x / 2, y / 2)]);
            save_rgb(&big, format!("{dir}/spike-outlines.png"));
        }
    }

    // Counts of 40 images: manual counts, and three automated methods.
    let mut rng = Rng::new(3);
    let manual: Vec<f32> = (0..40).map(|_| (17.0 + 130.0 * rng.uniform()).round()).collect();
    let scaled: Vec<f32> = manual.iter().map(|m| (0.8 * m + 2.0 * rng.normal()).round()).collect();
    let merges: Vec<f32> = manual.iter().map(|m| (m - 0.002 * m * m + 2.0 * rng.normal()).round()).collect();
    let scatter: Vec<f32> = manual.iter().map(|m| (m + 9.0 * rng.normal()).round()).collect();
    println!("\nCounts of 40 images against the manual count:");
    println!("  method                  r      bias    limits of agreement");
    for (name, auto) in [("80 % of the count", &scaled), ("merges in dense images", &merges), ("unbiased, scattered", &scatter)] {
        let ba = bland_altman(&manual, auto);
        println!("  {name:22}  {:.3}  {:6.1}   {:6.1} to {:5.1}", correlation(&manual, auto), ba.bias, ba.lower, ba.upper);
    }
    // the merges: is the error the same for sparse and dense images?
    let (sparse, dense): (Vec<usize>, Vec<usize>) = (0..40).partition(|&i| manual[i] < 80.0);
    let mean_diff = |idx: &[usize]| idx.iter().map(|&i| merges[i] - manual[i]).sum::<f32>() / idx.len() as f32;
    println!("  merges: mean difference {:.1} in images below 80 cells, {:.1} above", mean_diff(&sparse), mean_diff(&dense));

    let ba = bland_altman(&true_areas, &auto_areas);
    println!("\nAreas of the six discs, 1 px too large: bias {:.0} px², limits {:.0} to {:.0} px²", ba.bias, ba.lower, ba.upper);
    println!("wrote {dir}/spike-outlines.png");
}
