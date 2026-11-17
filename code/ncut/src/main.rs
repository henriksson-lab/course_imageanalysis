// Normalized cuts on point clouds (two blobs, blobs + outlier, three blobs, two rings) and on
// a small image. Prints μ₂, the splits and their Ncut; pictures in out/ncut/.
//   cargo run -p ncut                      (your code)
//   cargo run -p ncut --features solution  (the model solution)

use imgcourse::*;
use ncut::*;

/// `n` points around `centre`, Gaussian with standard deviation `sd`.
fn blob(rng: &mut Rng, n: usize, centre: [f64; 2], sd: f64) -> Vec<[f64; 2]> {
    (0..n).map(|_| [centre[0] + sd * rng.normal() as f64, centre[1] + sd * rng.normal() as f64]).collect()
}

/// `n` points on a ring of radius `r` around (0.5, 0.5), a little noisy.
fn ring(rng: &mut Rng, n: usize, r: f64) -> Vec<[f64; 2]> {
    (0..n)
        .map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            [0.5 + r * a.cos() + 0.01 * rng.normal() as f64, 0.5 + r * a.sin() + 0.01 * rng.normal() as f64]
        })
        .collect()
}

/// Points in the unit square, coloured by the Fiedler vector: blue negative, orange positive.
fn draw(points: &[[f64; 2]], x: &[f64], path: String) {
    let mut img = Image::new(400, 400, [1.0f32; 3]);
    for (p, v) in points.iter().zip(x) {
        let t = v.abs().min(1.0) as f32;
        let colour = if *v < 0.0 { [1.0 - 0.9 * t, 1.0 - 0.6 * t, 1.0 - 0.1 * t] } else { [1.0 - 0.05 * t, 1.0 - 0.5 * t, 1.0 - 0.95 * t] };
        let (cx, cy) = ((p[0] * 400.0) as isize, (p[1] * 400.0) as isize);
        for dy in -4..=4isize {
            for dx in -4..=4isize {
                if dx * dx + dy * dy <= 16 && img.contains(cx + dx, cy + dy) {
                    // a dark rim, so that pale points can be seen
                    let rim = dx * dx + dy * dy > 9;
                    img[((cx + dx) as usize, (cy + dy) as usize)] = if rim { [0.3, 0.3, 0.3] } else { colour };
                }
            }
        }
    }
    save_rgb(&img, path);
}

/// μ₂, the split at 0, the best split of the sweep, and the cheapest single node to cut off.
/// Small numbers in powers of ten, the others with 4 decimals.
fn num(v: f64) -> String {
    if v != 0.0 && v.abs() < 0.01 { format!("{v:.2e}") } else { format!("{v:.4}") }
}

fn report(name: &str, w: &[Vec<f64>], iterations: usize) -> Vec<f64> {
    let n = w.len();
    let (mu, x) = fiedler(w, iterations);
    let at_zero: Vec<bool> = x.iter().map(|v| *v < 0.0).collect();
    let k = at_zero.iter().filter(|&&a| a).count();
    let (cut0, nc0) = ncut(w, &at_zero);
    let (best, nc_best) = sweep(w, &x);
    let kb = best.iter().filter(|&&a| a).count();
    let (single, p) = (0..n)
        .map(|p| (ncut(w, &(0..n).map(|q| q == p).collect::<Vec<_>>()), p))
        .fold(((f64::INFINITY, 0.0), 0), |a, b| if b.0 .0 < a.0 .0 { b } else { a });
    println!("{name}: {n} nodes, μ₂ = {mu:.2e}");
    println!("  threshold 0: {k} | {}, cut {}, Ncut {};  best sweep: {kb} | {}, Ncut {}", n - k, num(cut0), num(nc0), n - kb, num(nc_best));
    println!("  cheapest single node to cut off: node {p}, cut {}, Ncut {}", num(single.0), num(single.1));
    x
}

fn main() {
    let dir = "out/ncut";
    let mut rng = Rng::new(6);

    let two = [blob(&mut rng, 25, [0.3, 0.5], 0.06), blob(&mut rng, 25, [0.7, 0.5], 0.06)].concat();
    let x = report("two blobs (σ = 0.08)", &affinities(&two, 0.08), 3000);
    draw(&two, &x, format!("{dir}/two-blobs.png"));
    report("two blobs (σ = 0.15)", &affinities(&two, 0.15), 3000);

    let outlier = [two.clone(), vec![[0.5, 0.95]]].concat();
    let x = report("blobs + outlier (σ = 0.08)", &affinities(&outlier, 0.08), 3000);
    draw(&outlier, &x, format!("{dir}/blobs-outlier.png"));

    let three = [blob(&mut rng, 20, [0.2, 0.3], 0.05), blob(&mut rng, 20, [0.45, 0.35], 0.05), blob(&mut rng, 20, [0.8, 0.7], 0.05)].concat();
    let x = report("three blobs (σ = 0.08)", &affinities(&three, 0.08), 3000);
    draw(&three, &x, format!("{dir}/three-blobs.png"));

    let rings = [ring(&mut rng, 20, 0.12), ring(&mut rng, 44, 0.35)].concat();
    for sigma in [0.05, 0.12] {
        let x = report(&format!("two rings (σ = {sigma})"), &affinities(&rings, sigma), 3000);
        draw(&rings, &x, format!("{dir}/rings-sigma{sigma}.png"));
    }

    // A 16 × 16 image: a bright disk and a grey square on a dark background, with noise.
    // Pixels within 2 of each other are joined, w = exp(−ΔI²/2σ²) · exp(−d²/2 · 1.5²).
    let size = 16;
    let clean = Image::from_fn(size, size, |x, y| {
        let disk = (x as f32 - 4.5).powi(2) + (y as f32 - 5.0).powi(2) <= 12.0;
        let square = (9..14).contains(&x) && (8..14).contains(&y);
        if disk { 190.0 } else if square { 120.0 } else { 60.0 }
    });
    let img = samples::add_noise(&clean, 10.0, 2);
    save_gray(&img.map(|v| v / 255.0), format!("{dir}/image.png"));
    for sigma in [30.0, 60.0] {
        let n = size * size;
        let mut w = vec![vec![0.0; n]; n];
        for (x, y) in img.coords() {
            for (u, v) in img.coords() {
                let d2 = ((x as isize - u as isize).pow(2) + (y as isize - v as isize).pow(2)) as f64;
                if d2 > 0.0 && d2 <= 4.0 {
                    let di = (img[(x, y)] - img[(u, v)]) as f64;
                    w[y * size + x][v * size + u] = (-di * di / (2.0 * sigma * sigma)).exp() * (-d2 / 4.5).exp();
                }
            }
        }
        let start = std::time::Instant::now();
        let x = report(&format!("image (σ = {sigma})"), &w, 3000);
        println!("  ({:.1} s)", start.elapsed().as_secs_f32());
        let vector = Image::from_fn(size * 8, size * 8, |u, v| x[(v / 8) * size + u / 8] as f32);
        save_gray_signed(&vector, format!("{dir}/image-fiedler-sigma{sigma}.png"));
    }
    println!("\nwrote the pictures to {dir}/ (points: blue negative, orange positive; image: the vector, 0 = grey)");
}
