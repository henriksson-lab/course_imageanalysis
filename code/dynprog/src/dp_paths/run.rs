// Runs your minimal paths on the course micrograph; results in out/dp-paths/.
//   cargo run -p dynprog -- dp-paths                      (your code)
//   cargo run -p dynprog --features solution -- dp-paths  (the model solution)

use crate::dp_paths::*;
use imgcourse::*;

/// Draw a path (one column per row) on a colour image, 3 pixels wide.
fn draw(rgb: &mut Image<[f32; 3]>, columns: &[usize], colour: [f32; 3]) {
    for (y, &x) in columns.iter().enumerate() {
        for dx in -1..=1isize {
            let xx = x as isize + dx;
            if rgb.contains(xx, y as isize) {
                rgb[(xx as usize, y)] = colour;
            }
        }
    }
}

/// The edge strength by central differences (not part of this unit's exercises).
fn gradient(img: &Image<f32>) -> Image<f32> {
    img.map_xy(|x, y, _| {
        let (x, y) = (x as isize, y as isize);
        let gx = img.get_border(x + 1, y, Border::Clamp) - img.get_border(x - 1, y, Border::Clamp);
        let gy = img.get_border(x, y + 1, Border::Clamp) - img.get_border(x, y - 1, Border::Clamp);
        0.5 * (gx * gx + gy * gy).sqrt()
    })
}

pub fn main() {
    let dir = "out/dp-paths";
    let img = samples::cell();
    let (w, h) = (img.width(), img.height());
    let (orange, blue) = ([1.0, 0.5, 0.0], [0.2, 0.5, 1.0]);

    // 1. The darkest path from top to bottom: the cost is the grey value.
    let start = std::time::Instant::now();
    let (m, pointer) = dp_table(&img);
    let best = backtrack(&m, &pointer);
    let time = start.elapsed();
    let greedy = greedy_path(&img);
    let mut rgb = gray_to_rgb(&img);
    draw(&mut rgb, &greedy, blue);
    draw(&mut rgb, &best, orange);
    save_rgb(&rgb, format!("{dir}/dark-path.png"));
    save_gray_stretched(&m, format!("{dir}/table-m.png"));
    let n = count_paths(w, h);
    let paths: f64 = (0..w).map(|x| n[(x, h - 1)]).sum();
    println!("darkest path: DP {:.2}, greedy {:.2} (sum of grey values along the path)", path_cost(&img, &best), path_cost(&img, &greedy));
    println!("  {paths:.2e} paths, {} candidate evaluations, DP took {:.2} ms", (h - 1) * (3 * w - 2), time.as_secs_f64() * 1e3);

    // 2. A seam that avoids edges: the cost is the gradient magnitude.
    let g = gradient(&img);
    let (m, pointer) = dp_table(&g);
    let seam = backtrack(&m, &pointer);
    let greedy = greedy_path(&g);
    let mut rgb = gray_to_rgb(&img);
    draw(&mut rgb, &greedy, blue);
    draw(&mut rgb, &seam, orange);
    save_rgb(&rgb, format!("{dir}/seam.png"));
    println!("seam through the gradient: DP {:.3}, greedy {:.3}", path_cost(&g, &seam), path_cost(&g, &greedy));

    // 3. The city-block distance to the letter F: one pass and two passes.
    let f = samples::letter_f(128);
    let bg = f.map(|v| v > 0.5);
    let one = distance_forward(&bg);
    let two = distance_two_passes(&bg);
    let finite_max = two.max();
    save_gray(&one.map(|d| d.min(finite_max) / finite_max), format!("{dir}/distance-one-pass.png"));
    save_gray(&two.map(|d| d / finite_max), format!("{dir}/distance-two-passes.png"));
    let wrong = one.pixels().iter().zip(two.pixels()).filter(|(a, b)| a != b).count();
    println!("distance to the letter F: {wrong} of {} pixels wrong after one pass", f.pixels().len());
    println!("wrote the results to {dir}/");
}
