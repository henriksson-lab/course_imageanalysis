// Single linkage and Felzenszwalb–Huttenlocher on the micrograph (grey levels 0–255, smoothed
// with σ = 0.8, 8-neighbours). Region pictures in random colours in out/mst-seg/.
//   cargo run -p mst-seg                      (your code)
//   cargo run -p mst-seg --features solution  (the model solution)

use imgcourse::*;
use mst_seg::*;
use std::time::Instant;

/// Gaussian smoothing with standard deviation `sigma` (separable, mirrored at the border).
fn smooth(img: &Image<f32>, sigma: f32) -> Image<f32> {
    let r = (3.0 * sigma).ceil() as isize;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    let pass = |src: &Image<f32>, along_x: bool| {
        src.map_xy(|x, y, _| {
            (-r..=r)
                .map(|i| {
                    let (xx, yy) = if along_x { (x as isize + i, y as isize) } else { (x as isize, y as isize + i) };
                    k[(i + r) as usize] * src.get_border(xx, yy, Border::Mirror)
                })
                .sum()
        })
    };
    pass(&pass(img, true), false)
}

fn to_image(labels: &[usize], w: usize, h: usize) -> Image<u32> {
    Image::from_fn(w, h, |x, y| labels[y * w + x] as u32 + 1)
}

fn count(labels: &[usize]) -> usize {
    labels.iter().max().map_or(0, |m| m + 1)
}

fn main() {
    let dir = "out/mst-seg";
    let img = smooth(&samples::cell().map(|v| 255.0 * v), 0.8);
    let (w, h) = (img.width(), img.height());
    let n = w * h;
    let start = Instant::now();
    let edges = grid_edges(&img, true);
    let tree = kruskal(n, &edges);
    println!("micrograph {w} × {h}: {} edges, minimum spanning tree with {} edges ({:.0} ms)", edges.len(), tree.len(), 1e3 * start.elapsed().as_secs_f64());
    let largest = tree.iter().fold(0.0f32, |m, e| m.max(e.2));
    println!("  the heaviest tree edge: {largest:.1}");

    println!("\nsingle linkage:");
    for t in [3.0, 6.0, 12.0] {
        let l = single_linkage(n, &edges, t);
        println!("  t = {t:4}: {:6} regions", count(&l));
        save_labels(&to_image(&l, w, h), format!("{dir}/single-{t}.png"));
    }

    println!("\nFelzenszwalb–Huttenlocher, then a minimum size of 20 pixels:");
    for k in [50.0, 300.0, 1500.0] {
        let start = Instant::now();
        let fh = felzenszwalb(n, &edges, k);
        let clean = merge_small(n, &edges, &fh, 20);
        let ms = 1e3 * start.elapsed().as_secs_f64();
        println!("  k = {k:6}: {:5} regions, {:4} after the clean-up ({ms:.0} ms)", count(&fh), count(&clean));
        save_labels(&to_image(&clean, w, h), format!("{dir}/fh-{k}.png"));
    }
    println!("\nwrote the region pictures to {dir}/");
}
