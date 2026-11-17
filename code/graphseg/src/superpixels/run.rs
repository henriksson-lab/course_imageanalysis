// SLIC superpixels on the micrograph for three compactness values, and the three measures on a
// synthetic nuclei image against its true labels. Pictures in out/superpixels/.
//   cargo run -p graphseg -- superpixels                      (your code)
//   cargo run -p graphseg --features solution -- superpixels  (the model solution)

use imgcourse::*;
use std::time::Instant;
use crate::superpixels::*;

/// The image with the superpixel boundaries in yellow.
fn boundaries(img: &Image<f32>, sp: &Image<u32>) -> Image<[f32; 3]> {
    let (w, h) = (img.width(), img.height());
    img.map_xy(|x, y, v| {
        let l = sp[(x, y)];
        if (x + 1 < w && sp[(x + 1, y)] != l) || (y + 1 < h && sp[(x, y + 1)] != l) {
            [1.0, 0.9, 0.0]
        } else {
            [v, v, v]
        }
    })
}

fn count(sp: &Image<u32>) -> usize {
    let mut seen = std::collections::HashSet::new();
    sp.pixels().iter().for_each(|&l| {
        seen.insert(l);
    });
    seen.len()
}

pub fn main() {
    let dir = "out/superpixels";
    // grey levels scaled to 0–100, like lightness, so that m means the same as for colour
    let cell = samples::cell();
    let scaled = cell.map(|v| 100.0 * v);
    println!("micrograph 256 × 256, K = 256, 10 iterations:");
    for m in [5.0, 20.0, 80.0] {
        let start = Instant::now();
        let raw = slic(&scaled, 256, m, 10);
        let ms = 1e3 * start.elapsed().as_secs_f64();
        let (_, pieces_before) = pieces(&raw);
        let sp = enforce_connectivity(&raw);
        println!("  m = {m:4}: {} pieces before the clean-up, {} superpixels after ({ms:.0} ms for SLIC)", pieces_before.len(), count(&sp));
        save_rgb(&boundaries(&cell, &sp), format!("{dir}/cell-m{m}.png"));
        if m == 5.0 {
            save_rgb(&boundaries(&cell, &raw), format!("{dir}/cell-m5-no-cleanup.png"));
        }
    }

    // The measures against K on synthetic nuclei (truth: background and 18 nuclei).
    let (nuc, truth) = samples::nuclei(256, 18, 4);
    let scaled = nuc.map(|v| 100.0 * v);
    println!("\nsynthetic nuclei, m = 20: K, superpixels, boundary recall (2 px), under-segmentation error, achievable accuracy");
    for k in [25, 50, 100, 200, 400, 800] {
        let sp = enforce_connectivity(&slic(&scaled, k, 20.0, 10));
        println!(
            "  K = {k:3}: {:3}   {:.3}   {:.3}   {:.4}",
            count(&sp),
            boundary_recall(&sp, &truth, 2),
            undersegmentation_error(&sp, &truth),
            achievable_accuracy(&sp, &truth)
        );
        if k == 50 || k == 400 {
            save_rgb(&boundaries(&nuc, &sp), format!("{dir}/nuclei-k{k}.png"));
        }
    }
    println!("\nwrote the pictures to {dir}/");
}
