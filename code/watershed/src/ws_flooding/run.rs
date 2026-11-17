// Runs the flooding on the exercise grid and on a large image; pictures in out/ws-flooding/.
//   cargo run -p watershed -- ws-flooding                      (your code)
//   cargo run -p watershed --features solution -- ws-flooding  (the model solution)

use imgcourse::*;
use std::time::Instant;
use crate::ws_flooding::*;

/// The labels in colour, with the watershed line (label 0) in white.
fn picture(labels: &Image<u32>) -> Image<[f32; 3]> {
    let c = io::label_colours(labels);
    labels.map_xy(|x, y, l| if l == 0 { [1.0; 3] } else { c[(x, y)] })
}

pub fn main() {
    let dir = "out/ws-flooding";

    // 1. The 5 × 5 grid of the exercise (A = 1, B = 2, line = 0).
    let f = Image::from_rows(&[[3., 2., 3., 5., 6.], [2., 1., 2., 4., 5.], [3., 2., 4., 3., 2.], [5., 4., 3., 2., 1.], [6., 5., 3., 1., 0.]]);
    let mut markers = Image::new(5, 5, 0u32);
    markers[(1, 1)] = 1;
    markers[(4, 4)] = 2;
    for eight in [false, true] {
        let (labels, order) = watershed(&f, &markers, eight, true, &mut HeapQueue::new());
        println!("exercise grid, {} neighbours, {} pops (0 = line):\n{labels}", if eight { 8 } else { 4 }, order.len());
    }

    // 2. A large landscape: the inverted synthetic nuclei image, 1024 × 1024, in 256 grey levels,
    //    flooded from all its regional minima, with a heap and with a bucket queue.
    let (img, _) = samples::nuclei(1024, 60, 1);
    let f = img.map(|v| ((1.0 - v).clamp(0.0, 1.0) * 255.0).round());
    let t = Instant::now();
    let (minima, n) = regional_minima(&f, true);
    println!("1024 × 1024 landscape: {n} regional minima (8 neighbours), found in {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let (a, _) = watershed(&f, &minima, true, true, &mut HeapQueue::new());
    let t_heap = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (b, _) = watershed(&f, &minima, true, true, &mut BucketQueue::new());
    let t_bucket = t.elapsed().as_secs_f64();
    println!("flooding with a heap: {:.0} ms; with a bucket queue: {:.0} ms; same labels: {}", t_heap * 1e3, t_bucket * 1e3, a == b);
    let lines = a.pixels().iter().filter(|&&l| l == 0).count();
    println!("{lines} line pixels ({:.1} % of the image)", 100.0 * lines as f32 / (1024.0 * 1024.0));
    save_gray(&img.crop(0, 0, 256, 256), format!("{dir}/nuclei-crop.png"));
    save_rgb(&picture(&a.crop(0, 0, 256, 256)), format!("{dir}/all-minima-crop.png"));

    println!("wrote the results to {dir}/");
}
