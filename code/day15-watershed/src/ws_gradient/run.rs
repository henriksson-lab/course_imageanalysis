// The watershed of the gradient of the micrograph; pictures in out/ws-gradient/.
//   cargo run -p day15 -- ws-gradient                      (your code)
//   cargo run -p day15 --features solution -- ws-gradient  (the model solution)

use imgcourse::*;
use crate::ws_gradient::*;

/// The image with the region boundaries in yellow.
fn overlay(img: &Image<f32>, labels: &Image<u32>) -> Image<[f32; 3]> {
    let b = boundaries(labels);
    img.map_xy(|x, y, v| if b[(x, y)] { [1.0, 0.9, 0.0] } else { [v, v, v] })
}

pub fn main() {
    let dir = "out/ws-gradient";
    let img = samples::cell();
    let grey = img.map(|v| 255.0 * v); // grey levels 0..255, as in the lecture
    save_gray(&img, format!("{dir}/cell.png"));

    // 1. Smoothing alone: the number of regional minima of the gradient.
    print!("regional minima of the gradient:");
    for sigma in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0] {
        let g = gradient_l1(&gaussian_blur(&grey, sigma));
        print!("  σ = {sigma}: {}", regional_minima(&g).1);
    }
    println!();
    let g1 = gradient_l1(&gaussian_blur(&grey, 1.0));
    save_gray_stretched(&g1, format!("{dir}/gradient-sigma1.png"));
    let raw = watershed(&g1, &regional_minima(&g1).0);
    save_rgb(&overlay(&img, &raw), format!("{dir}/all-minima-sigma1.png"));

    // 2. h-minima at σ = 2: only minima deeper than h are markers.
    let g = gradient_l1(&gaussian_blur(&grey, 2.0));
    save_gray_stretched(&g, format!("{dir}/gradient-sigma2.png"));
    print!("σ = 2, regions after h-minima:");
    for h in [0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 160.0] {
        print!("  h = {h}: {}", regional_minima(&h_minima(&g, h)).1);
    }
    println!();
    for h in [20.0, 80.0] {
        let markers = regional_minima(&h_minima(&g, h)).0;
        let labels = watershed(&g, &markers);
        save_rgb(&overlay(&img, &labels), format!("{dir}/hminima-{h}.png"));
    }

    // 3. The hierarchy: flood once from all minima, then merge across dams lower than t.
    let fine = watershed(&g, &regional_minima(&g).0);
    let dams = dam_heights(&fine, &g);
    print!("σ = 2, regions after merging across dams ≤ t:");
    for t in [0.0, 10.0, 20.0, 40.0, 80.0, 160.0] {
        print!("  t = {t}: {}", merge_below(&fine, &dams, t).1);
    }
    println!();
    let (merged, n) = merge_below(&fine, &dams, 80.0);
    save_rgb(&overlay(&img, &merged), format!("{dir}/merged-80.png"));
    println!("merged at t = 80: {n} regions");
    println!("wrote the results to {dir}/");
}
