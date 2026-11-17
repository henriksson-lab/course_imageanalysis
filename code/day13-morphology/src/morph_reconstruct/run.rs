// Runs your reconstruction on synthetic nuclei: h-maxima as markers, fill holes and clear
// border on a mask; results in out/morph-reconstruct/.
//   cargo run -p day13 -- morph-reconstruct                      (your code)
//   cargo run -p day13 --features solution -- morph-reconstruct  (the model solution)

use imgcourse::*;
use crate::morph_reconstruct::*;

/// The number of 8-connected groups of true pixels (a small flood fill; labelling all
/// components is the next day's topic).
fn count_groups(a: &Image<bool>) -> usize {
    let mut seen = Image::new(a.width(), a.height(), false);
    let mut groups = 0;
    for (x, y) in a.coords() {
        if a[(x, y)] && !seen[(x, y)] {
            groups += 1;
            seen[(x, y)] = true;
            let mut stack = vec![(x, y)];
            while let Some((px, py)) = stack.pop() {
                for (dx, dy) in neighbourhood(true) {
                    let (u, v) = (px as isize + dx, py as isize + dy);
                    if a.contains(u, v) && a[(u as usize, v as usize)] && !seen[(u as usize, v as usize)] {
                        seen[(u as usize, v as usize)] = true;
                        stack.push((u as usize, v as usize));
                    }
                }
            }
        }
    }
    groups
}

pub fn main() {
    let dir = "out/morph-reconstruct";
    let (img, labels) = samples::nuclei(256, 22, 8);
    // the number of nuclei that can be seen (where two overlap, the later one hides part of the other)
    let mut seen: Vec<u32> = labels.pixels().iter().copied().filter(|&l| l > 0).collect();
    seen.sort();
    seen.dedup();
    let n_nuclei = seen.len();
    // grey levels 0–255, rounded, so that neighbouring levels differ by 1
    let img = img.map(|v| (v * 255.0).round().clamp(0.0, 255.0));
    save_gray(&img.map(|v| v / 255.0), format!("{dir}/nuclei.png"));

    // Regional maxima of the h-maxima transform: one dot per nucleus?
    for h in [0.0, 10.0, 20.0, 30.0, 40.0, 60.0] {
        let peaks = regional_maxima(&h_maxima(&img, h), 1.0);
        println!("h = {h:2}: {} regional maxima ({n_nuclei} nuclei)", count_groups(&peaks));
        let mut rgb = gray_to_rgb(&img.map(|v| v / 255.0));
        for (x, y) in peaks.coords() {
            if peaks[(x, y)] {
                rgb[(x, y)] = [1.0, 0.1, 0.1];
            }
        }
        save_rgb(&rgb, format!("{dir}/maxima-h{h}.png"));
    }

    // A mask with holes and objects cut by the border: fill holes, clear border.
    // Cut out the middle, so that some nuclei touch the border; noise makes holes.
    let a = samples::add_noise(&img, 25.0, 9).crop(28, 28, 200, 200).map(|v| v > 115.0);
    let filled = fill_holes(&a);
    let cleared = clear_border(&filled);
    let count = |m: &Image<bool>| m.pixels().iter().filter(|&&b| b).count();
    println!(
        "mask: {} pixels in {} objects; fill holes adds {} pixels; clear border removes {} objects ({} pixels)",
        count(&a),
        count_groups(&a),
        count(&filled) - count(&a),
        count_groups(&filled) - count_groups(&cleared),
        count(&filled) - count(&cleared)
    );
    // dark: kept, green: filled hole, red: removed (touches the border)
    let picture = Image::from_fn(200, 200, |x, y| match (a[(x, y)], filled[(x, y)], cleared[(x, y)]) {
        (_, true, false) => [0.9, 0.2, 0.2],
        (false, true, true) => [0.2, 0.8, 0.2],
        (true, _, _) => [1.0, 1.0, 1.0],
        _ => [0.0, 0.0, 0.0],
    });
    save_rgb(&picture, format!("{dir}/fill-clear.png"));

    // How many geodesic dilations does the flood of the background need?
    let background = a.map(|p| if p { 0.0 } else { 1.0 });
    let seeds = background.map_xy(|x, y, v| if x == 0 || y == 0 || x == 199 || y == 199 { v } else { 0.0 });
    let (_, iterations) = reconstruct(&seeds, &background, false);
    println!("the flood of the background from the frame takes {iterations} geodesic dilations");
    println!("wrote the results to {dir}/");
}
