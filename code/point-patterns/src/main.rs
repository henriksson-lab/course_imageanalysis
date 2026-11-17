// Quadrat counts and intensity maps of five simulated patterns in a 200 × 200 µm window;
// results in out/point-patterns/.
//   cargo run -p point-patterns                      (your code)
//   cargo run -p point-patterns --features solution  (the model solution)

use imgcourse::*;
use point_patterns::*;

/// Merge points closer than `d` (a detection artefact): repeatedly replace the closest pair
/// by its midpoint.
fn merge_close(mut pts: Vec<Point>, d: f64) -> Vec<Point> {
    loop {
        let mut best = (f64::INFINITY, 0, 0);
        for i in 0..pts.len() {
            for j in i + 1..pts.len() {
                let dist = (pts[i].0 - pts[j].0).hypot(pts[i].1 - pts[j].1);
                if dist < best.0 {
                    best = (dist, i, j);
                }
            }
        }
        if best.0 >= d {
            return pts;
        }
        let (p, q) = (pts[best.1], pts[best.2]);
        pts[best.1] = ((p.0 + q.0) / 2.0, (p.1 + q.1) / 2.0);
        pts.swap_remove(best.2);
    }
}

fn main() {
    let dir = "out/point-patterns";
    let (a, b) = (200.0, 200.0);
    let mut rng = Rng::new(1);
    let random = csr(150, a, b, &mut rng);
    let clusters = thomas(0.001, 5.0, 5.0, a, b, &mut rng);
    let regular = hard_core(150, 12.0, a, b, &mut rng);
    // more cells to the right: CSR, each point kept with probability x / a
    let gradient: Vec<Point> = csr(300, a, b, &mut rng).into_iter().filter(|p| (rng.uniform() as f64) < p.0 / a).collect();
    let merged = merge_close(random.clone(), 8.0);
    let patterns = [("CSR", &random), ("clusters", &clusters), ("hard core 12", &regular), ("gradient", &gradient), ("CSR, merged 8", &merged)];

    let sides = [5, 10, 20, 25, 50];
    println!("VMR of quadrat counts, by the side of the quadrats (µm):");
    println!("{:>14} {:>5} {}", "pattern", "n", sides.map(|s| format!("{s:>7}")).join(""));
    let mut strip = Image::new(5 * 302, 300, 0.5f32);
    for (k, (name, pts)) in patterns.iter().enumerate() {
        let vmrs = sides.map(|s| {
            let (_, _, vmr, _) = dispersion(&quadrat_counts(pts, a, b, (a / s as f64) as usize));
            format!("{vmr:7.2}")
        });
        println!("{name:>14} {:>5} {}", pts.len(), vmrs.join(""));
        let pic = draw(pts, a, b, 1.5);
        for (x, y) in pic.coords() {
            strip[(k * 302 + x, y)] = pic[(x, y)];
        }
    }
    // the χ² test of B7 for 8 × 8 quadrats of 25 µm (63 degrees of freedom)
    println!();
    for (name, pts) in &patterns[..3] {
        let (mean, var, vmr, x2) = dispersion(&quadrat_counts(pts, a, b, 8));
        println!("{name:>14}, 8 × 8 quadrats: mean {mean:.2}, variance {var:.2}, VMR {vmr:.2}, X² = {x2:.1} on 63 df");
    }
    save_gray(&strip, format!("{dir}/patterns.png"));

    // the intensity of the gradient pattern with three bandwidths
    let mut maps = Image::new(3 * 202, 200, 0.0f32);
    for (k, sigma) in [3.0, 15.0, 60.0].into_iter().enumerate() {
        let map = intensity_map(&gradient, a, b, sigma);
        let column_means: Vec<f32> = [10, 100, 190].iter().map(|&x| (0..200).map(|y| map[(x, y)]).sum::<f32>() / 200.0).collect();
        println!("gradient, σ = {sigma:>2} µm: mean intensity at x = 10, 100, 190 µm: {:.4}, {:.4}, {:.4} per µm² (true: {:.4}, {:.4}, {:.4})",
            column_means[0], column_means[1], column_means[2], 0.0075 * 10.0 / a, 0.0075 * 100.0 / a, 0.0075 * 190.0 / a);
        for (x, y) in map.coords() {
            maps[(k * 202 + x, y)] = map[(x, y)] / 0.012;
        }
    }
    save_gray(&maps, format!("{dir}/intensity.png"));
    println!("wrote {dir}/patterns.png (CSR, clusters, hard core, gradient, merged) and {dir}/intensity.png (σ = 3, 15, 60 µm; white = 0.012 per µm²)");
}
