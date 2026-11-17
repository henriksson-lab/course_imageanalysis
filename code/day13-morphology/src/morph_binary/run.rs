// Runs your erosion and dilation on a thresholded image of synthetic nuclei; results in
// out/morph-binary/.
//   cargo run -p day13 -- morph-binary                      (your code)
//   cargo run -p day13 --features solution -- morph-binary  (the model solution)

use imgcourse::*;
use crate::morph_binary::*;
use std::time::Instant;

fn to_gray(a: &Image<bool>) -> Image<f32> {
    a.map(|b| if b { 1.0 } else { 0.0 })
}

fn count(a: &Image<bool>) -> usize {
    a.pixels().iter().filter(|&&b| b).count()
}

pub fn main() {
    let dir = "out/morph-binary";
    // A noisy threshold: specks in the background, holes in the nuclei.
    let (img, _) = samples::nuclei(256, 25, 4);
    let img = samples::add_noise(&img, 0.12, 6);
    let a = img.map(|v| v > 0.45);
    save_gray(&img, format!("{dir}/nuclei.png"));
    save_gray(&to_gray(&a), format!("{dir}/mask.png"));
    println!("mask: {} pixels", count(&a));

    for r in [1, 3] {
        let e = erode(&a, &disc(r), false);
        let d = dilate(&a, &disc(r), false);
        println!("disc of radius {r} ({} pixels): erosion {} pixels, dilation {} pixels", disc(r).len(), count(&e), count(&d));
        save_gray(&to_gray(&e), format!("{dir}/erode-r{r}.png"));
        save_gray(&to_gray(&d), format!("{dir}/dilate-r{r}.png"));
    }

    // Inner boundaries of a cleaner mask (no noise), drawn in red on the image.
    let (clean, _) = samples::nuclei(256, 25, 4);
    let b = clean.map(|v| v > 0.45);
    for (name, se) in [("cross", CROSS), ("square", SQUARE)] {
        let edge = inner_boundary(&b, se);
        println!("inner boundary with the {name}: {} pixels", count(&edge));
        let mut rgb = gray_to_rgb(&clean);
        for (x, y) in edge.coords() {
            if edge[(x, y)] {
                rgb[(x, y)] = [1.0, 0.1, 0.1];
            }
        }
        save_rgb(&rgb, format!("{dir}/boundary-{name}.png"));
    }

    // A 15 × 15 square directly, and as a row of 15 followed by a column of 15.
    let square: Vec<(isize, isize)> = (-7..=7).flat_map(|dy| (-7..=7).map(move |dx| (dx, dy))).collect();
    let row: Vec<(isize, isize)> = (-7..=7).map(|dx| (dx, 0)).collect();
    let column: Vec<(isize, isize)> = (-7..=7).map(|dy| (0, dy)).collect();
    let start = Instant::now();
    let direct = dilate(&a, &square, false);
    let t_direct = start.elapsed();
    let start = Instant::now();
    let two = dilate(&dilate(&a, &row, false), &column, false);
    let t_two = start.elapsed();
    println!(
        "15 × 15 square: directly {:.1} ms, as a row and a column {:.1} ms, same result: {}",
        t_direct.as_secs_f64() * 1000.0,
        t_two.as_secs_f64() * 1000.0,
        direct == two
    );
    println!("wrote the results to {dir}/");
}
