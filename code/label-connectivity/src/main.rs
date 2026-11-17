// Labels a synthetic nuclei image and the micrograph, and writes the results to out/label-connectivity/.
//   cargo run -p label-connectivity                      (your code)
//   cargo run -p label-connectivity --features solution  (the model solution)

use imgcourse::*;
use label_connectivity::*;
use std::time::Instant;

fn main() {
    let dir = "out/label-connectivity";

    // 1. Synthetic nuclei: 30 bright ellipses, some touching. Threshold, then label.
    let (img, truth) = samples::nuclei(256, 30, 7);
    let true_count = truth.pixels().iter().copied().collect::<std::collections::BTreeSet<u32>>().len() - 1;
    let mask = img.map(|v| v > 0.35);
    save_gray(&img, format!("{dir}/nuclei.png"));
    save_gray(&mask.map(|m| if m { 1.0 } else { 0.0 }), format!("{dir}/nuclei-mask.png"));
    let (lab8, n8) = label_flood_fill(&mask, true);
    let (_, n4) = label_flood_fill(&mask, false);
    save_labels(&lab8, format!("{dir}/nuclei-labels.png"));
    println!("nuclei: {true_count} nuclei visible in the true labels; the mask has {n8} components with 8-connectivity, {n4} with 4");

    // 2. The micrograph: dark regions (below 95 of 255) as objects, as in the demo of the last unit.
    let cell = samples::cell();
    let dark = cell.map(|v| v < 95.0 / 255.0);
    save_gray(&dark.map(|m| if m { 1.0 } else { 0.0 }), format!("{dir}/cell-mask.png"));
    for eight in [false, true] {
        let t = Instant::now();
        let (lab, n) = label_flood_fill(&dark, eight);
        let t_fill = t.elapsed();
        let t = Instant::now();
        let (prov, lab2, n2) = label_two_pass(&dark, eight);
        let t_two = t.elapsed();
        let provisional = prov.pixels().iter().max().copied().unwrap_or(0);
        assert!(n == n2 && lab == lab2, "flood fill and two passes disagree");
        println!(
            "cell, {}-connectivity: {n} objects; flood fill {:.1} ms, two passes {:.1} ms ({provisional} provisional labels)",
            if eight { 8 } else { 4 },
            t_fill.as_secs_f64() * 1e3,
            t_two.as_secs_f64() * 1e3
        );
        if eight {
            save_labels(&lab, format!("{dir}/cell-labels.png"));
        }
    }

    // 3. The order of a flood fill: fill the largest component of the nuclei mask from the
    //    pixel nearest its centre, and colour the pixels by when they came out of the queue,
    //    in 16 bands of equal size. Left: 4-connectivity, right: 8. Enlarged 3 times.
    let mut sizes = vec![0usize; n8 as usize + 1];
    for &l in lab8.pixels() {
        sizes[l as usize] += 1;
    }
    let biggest = (1..sizes.len()).max_by_key(|&l| sizes[l]).unwrap() as u32;
    let pixels: Vec<(usize, usize)> = lab8.coords().filter(|&p| lab8[p] == biggest).collect();
    let (sx, sy) = pixels.iter().fold((0, 0), |(a, b), &(x, y)| (a + x, b + y));
    let (cx, cy) = (sx / pixels.len(), sy / pixels.len());
    let seed = *pixels.iter().min_by_key(|&&(x, y)| x.abs_diff(cx).pow(2) + y.abs_diff(cy).pow(2)).unwrap();
    let x0 = pixels.iter().map(|p| p.0).min().unwrap().saturating_sub(2);
    let y0 = pixels.iter().map(|p| p.1).min().unwrap().saturating_sub(2);
    let w = (pixels.iter().map(|p| p.0).max().unwrap() + 3).min(256) - x0;
    let h = (pixels.iter().map(|p| p.1).max().unwrap() + 3).min(256) - y0;
    let only = lab8.map(|l| l == biggest);
    let zoom = 3;
    let mut picture = Image::new(2 * w * zoom + 4, h * zoom, [1.0f32; 3]);
    for (k, eight) in [false, true].into_iter().enumerate() {
        let mut lab = Image::new(256, 256, 0u32);
        let order = flood_fill(&only, &mut lab, seed, 1, eight);
        let mut band = Image::new(256, 256, [0.0f32; 3]);
        for (i, &p) in order.iter().enumerate() {
            let b = i * 16 / order.len();
            band[p] = if b % 2 == 1 { [0.95, 0.75, 0.3] } else { [0.3, 0.45, 0.85] };
        }
        band[seed] = [1.0, 0.0, 0.0];
        for (x, y) in picture.coords() {
            let (u, v) = (x as isize - (k * (w * zoom + 4)) as isize, y);
            if u >= 0 && (u as usize) < w * zoom {
                picture[(x, y)] = band[(x0 + u as usize / zoom, y0 + v / zoom)];
            }
        }
    }
    save_rgb(&picture, format!("{dir}/fill-order.png"));
    println!("wrote the results to {dir}/");
}
