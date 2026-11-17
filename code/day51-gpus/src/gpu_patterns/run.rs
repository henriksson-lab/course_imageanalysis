// The patterns at work on images: a sum by a loop and by a tree, the integral image as two
// passes of scans, compaction of the foreground pixels, and an atomic histogram with and
// without private block histograms. Numbers on the terminal, a picture in out/gpu-patterns/.
//   cargo run -p day51 -- gpu-patterns                      (your code)
//   cargo run -p day51 --features solution -- gpu-patterns  (the model solution)

use crate::gpu_patterns::*;
use imgcourse::*;

pub fn main() {
    let dir = "out/gpu-patterns";
    let cell = samples::cell();

    // The sum of a big image (the micrograph tiled to 4096 × 4096) in float32.
    let big = Image::from_fn(4096, 4096, |x, y| cell[(x % 256, y % 256)]);
    let exact: f64 = big.pixels().iter().map(|&v| v as f64).sum();
    let looped = big.pixels().iter().fold(0.0f32, |s, &v| s + v);
    let rows = reduce_tree(big.pixels());
    let tree = rows.last().unwrap()[0];
    println!("Sum of 4096 × 4096 pixels: exact {exact:.1}");
    println!("  loop in float32: {looped:.1} (relative error {:.1e}), {} steps", (looped as f64 - exact).abs() / exact, big.pixels().len() - 1);
    println!("  tree in float32: {tree:.1} (relative error {:.1e}), {} steps", (tree as f64 - exact).abs() / exact, rows.len());

    // The integral image: a scan along every row, then along every column.
    let mut integral = cell.clone();
    let mut additions = 0;
    for y in 0..cell.height() {
        let (row, adds) = scan_hillis_steele(cell.row(y));
        additions += adds;
        for x in 0..cell.width() {
            integral[(x, y)] = row[x];
        }
    }
    for x in 0..cell.width() {
        let column: Vec<f32> = (0..cell.height()).map(|y| integral[(x, y)]).collect();
        let (column, adds) = scan_blelloch(&column); // exclusive: add the pixel itself back
        additions += adds;
        for y in 0..cell.height() {
            integral[(x, y)] += column[y];
        }
    }
    let (w, h) = (cell.width(), cell.height());
    let exact: f64 = cell.pixels().iter().map(|&v| v as f64).sum();
    println!("Integral image of the 256 × 256 micrograph: bottom right {:.2} (the sum of all pixels, in float64, {exact:.2}), {additions} additions in 2 · 256 scans", integral[(w - 1, h - 1)]);
    save_gray_stretched(&integral, format!("{dir}/integral.png"));

    // Compaction: the list of the foreground pixels of synthetic nuclei.
    let (_, labels) = samples::nuclei(256, 30, 5);
    let coords: Vec<(usize, usize)> = labels.coords().collect();
    let keep: Vec<bool> = labels.pixels().iter().map(|&l| l > 0).collect();
    let list = compact(&coords, &keep);
    println!("Compaction: {} of {} pixels are foreground; the first are {:?}", list.len(), coords.len(), &list[..3]);

    // Histogram of the micrograph (256 bins), one thread per pixel, blocks of 256 threads.
    for (name, img) in [("micrograph", cell.clone()), ("dark image (micrograph / 8)", cell.map(|v| v / 8.0))] {
        let values: Vec<usize> = img.pixels().iter().map(|&v| ((v * 255.0).round() as usize).min(255)).collect();
        let hist = atomic_histogram(&values, 256, 256);
        let busiest = hist.counts.iter().enumerate().max_by_key(|c| c.1).unwrap();
        println!(
            "Histogram of the {name}: busiest bin {} with {} pixels; global atomics {} cycles, private block histograms {} cycles",
            busiest.0, busiest.1, hist.global_cycles, hist.private_cycles
        );
    }
    println!("wrote the picture to {dir}/");
}
