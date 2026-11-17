// Splits touching nuclei with the distance transform; pictures in out/ws-distance/.
//   cargo run -p watershed -- ws-distance                      (your code)
//   cargo run -p watershed --features solution -- ws-distance  (the model solution)

use imgcourse::*;
use crate::ws_distance::*;

/// Two discs of radius r with centres d apart, on the row y = cy; returns the mask and the
/// pixel positions of the two centres.
fn two_discs(r: f32, d: f32) -> (Image<bool>, (usize, usize), (usize, usize)) {
    let w = (2.0 * r + d) as usize + 10;
    let h = 2 * r as usize + 10;
    let (c1, cy) = (5 + r as usize, h / 2);
    let c2 = c1 + d as usize;
    let mask = Image::from_fn(w, h, |x, y| {
        let (dy2, dx1, dx2) = ((y as f32 - cy as f32).powi(2), (x as f32 - c1 as f32).powi(2), (x as f32 - c2 as f32).powi(2));
        dx1 + dy2 <= r * r || dx2 + dy2 <= r * r
    });
    (mask, (c1, cy), (c2, cy))
}

/// The labels in colour, with the watershed line (0 inside the mask) in white.
fn picture(labels: &Image<u32>, mask: &Image<bool>) -> Image<[f32; 3]> {
    let c = io::label_colours(labels);
    labels.map_xy(|x, y, l| if l == 0 && mask[(x, y)] { [1.0; 3] } else { c[(x, y)] })
}

pub fn main() {
    let dir = "out/ws-distance";

    // 1. Synthetic nuclei: threshold, count the components, then split with the distance transform.
    let (img, truth) = samples::nuclei(256, 30, 7);
    let mask = img.map(|v| v > 0.35);
    let (_, components) = label_components(&mask);
    let true_count = {
        let mut seen = [false; 31];
        truth.pixels().iter().for_each(|&l| seen[l as usize] = true);
        seen[1..].iter().filter(|&&s| s).count()
    };
    println!("nuclei: {true_count} nuclei; the mask has {components} components");
    save_gray(&img, format!("{dir}/nuclei.png"));
    save_labels(&truth, format!("{dir}/nuclei-true.png"));

    let metrics: [(&str, Image<f32>); 3] =
        [("Euclidean", distance_euclidean(&mask)), ("chamfer 3-4", distance_chamfer34(&mask)), ("city-block", distance_city_block(&mask))];
    save_gray_stretched(&metrics[0].1, format!("{dir}/nuclei-dt.png"));
    let hs = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0];
    println!("objects after splitting, for h = {hs:?}:");
    for (name, dt) in &metrics {
        let counts: Vec<u32> = hs.iter().map(|&h| split(dt, &mask, h, false).1).collect();
        println!("  {name:12} {counts:?}");
    }
    for h in [0.0, 2.0] {
        let (labels, _) = split(&metrics[0].1, &mask, h, true);
        save_rgb(&picture(&labels, &mask), format!("{dir}/nuclei-split-h{h}.png"));
    }

    // 2. Two discs of radius 30 moving together: peaks, neck, depth, and the result with h = 2.
    println!("\ntwo discs, r = 30, h = 2 (Euclidean): d, peak, neck, measured depth, formula, objects");
    for d in [20.0, 24.0, 26.0, 28.0, 30.0, 34.0, 40.0, 50.0] {
        let (m, a, b) = two_discs(30.0, d);
        let dt = distance_euclidean(&m);
        let peak = dt[a].min(dt[b]);
        let neck = dt[((a.0 + b.0) / 2, a.1)];
        let formula = 30.0 - (900.0 - d * d / 4.0f32).sqrt();
        let objects = split(&dt, &m, 2.0, false).1;
        println!("{d:6} {peak:6.2} {neck:6.2} {:6.2} {formula:6.2} {objects:3}", peak - neck);
        if d == 40.0 {
            save_gray_stretched(&dt, format!("{dir}/two-discs-dt.png"));
            save_rgb(&picture(&split(&dt, &m, 2.0, true).0, &m), format!("{dir}/two-discs-split.png"));
        }
    }
    println!("wrote the results to {dir}/");
}
