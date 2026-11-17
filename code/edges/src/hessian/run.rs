// Runs the Hessian on a test scene and on the course micrograph; results in out/hessian/.
//   cargo run -p edges -- hessian                      (your code)
//   cargo run -p edges --features solution -- hessian  (the model solution)

use crate::hessian::*;
use imgcourse::*;

/// A 128 × 128 test scene in grey levels (0 … 255), like the demo's: a bright Gaussian spot at
/// (28, 28), a thin bright line (width 1.5) down x = 86, a thick bright line (width 4.5) along
/// y = 70 (left half), a thin dark line along y = 40 (right half), a thin bright line along
/// y = 104 that crosses the vertical one (a junction at (86, 104)), and a bright square at the
/// bottom left (a step edge).
fn scene() -> Image<f32> {
    let g = |d: f32, s: f32| (-d * d / (2.0 * s * s)).exp();
    let clean = Image::from_fn(128, 128, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let mut v = 40.0;
        v += 120.0 * g(xf - 28.0, 4.0) * g(yf - 28.0, 4.0);
        v += 100.0 * g(xf - 86.0, 1.5);
        if x < 70 {
            v += 100.0 * g(yf - 70.0, 4.5);
        }
        if x > 64 {
            v -= 30.0 * g(yf - 40.0, 1.5);
        }
        if (60..112).contains(&x) {
            v += 100.0 * g(yf - 104.0, 1.5);
        }
        if x < 25 && y > 100 {
            v += 80.0;
        }
        v
    });
    samples::add_noise(&clean, 3.0, 5)
}

fn class_colour(c: Class) -> [f32; 3] {
    match c {
        Class::Flat => [0.15, 0.15, 0.15],
        Class::BrightBlob => [1.0, 0.85, 0.1],
        Class::DarkBlob => [0.1, 0.5, 1.0],
        Class::BrightRidge => [1.0, 0.3, 0.1],
        Class::DarkRidge => [0.2, 0.8, 1.0],
        Class::Saddle => [0.7, 0.3, 0.9],
    }
}

pub fn main() {
    let dir = "out/hessian";
    let img = scene();
    save_gray(&img.map(|v| v / 255.0), format!("{dir}/scene.png"));

    // Classes, trace and determinant at σ = 3.
    let h = hessian(&img, 3.0);
    let eig = h.map(eigen_sym);
    save_rgb(&eig.map(|(l1, l2, _)| class_colour(classify(l1, l2, 4.0))), format!("{dir}/classes.png"));
    save_gray_signed(&eig.map(|(l1, l2, _)| l1 + l2), format!("{dir}/trace.png"));
    save_gray_signed(&eig.map(|(l1, l2, _)| l1 * l2), format!("{dir}/determinant.png"));
    println!("σ = 3 (values × σ²):");
    for (name, x, y) in [("spot", 28, 28), ("thin line", 86, 27), ("thick line", 30, 70), ("dark line", 100, 40), ("junction", 86, 104), ("inside the edge", 12, 104)] {
        let (l1, l2, u) = eig[(x, y)];
        println!(
            "  {name:16} ({x:3}, {y:3}): λ = ({l1:7.2}, {l2:7.2}), along ({:5.2}, {:5.2}), trace {:7.2}, det {:7.1}, {:?}, vesselness {:.2}",
            u[0],
            u[1],
            l1 + l2,
            l1 * l2,
            classify(l1, l2, 4.0),
            vesselness(l1, l2, 0.5, 15.0)
        );
    }

    // Vesselness: one small σ, then the maximum over 8 scales.
    let (one, _) = multiscale_vesselness(&img, &[1.5], 0.5, 15.0, true);
    save_gray(&one, format!("{dir}/vesselness-sigma1.5.png"));
    let sigmas: Vec<f32> = (0..8).map(|k| 8f32.powf(k as f32 / 7.0)).collect();
    let (multi, best) = multiscale_vesselness(&img, &sigmas, 0.5, 15.0, true);
    save_gray(&multi, format!("{dir}/vesselness-multi.png"));
    save_gray(&best.map(|s| s / 8.0), format!("{dir}/best-sigma.png"));
    for (name, x, y) in [("thin line", 86, 27), ("thick line", 30, 70), ("spot", 28, 28), ("junction", 86, 104)] {
        println!("  {name:10}: vesselness {:.2} at σ = 1.5; multi-scale {:.2}, best σ = {:.2}", one[(x, y)], multi[(x, y)], best[(x, y)]);
    }

    // The membranes of the micrograph are dark lines.
    let cell = samples::cell().map(|v| 255.0 * v);
    let (membranes, _) = multiscale_vesselness(&cell, &[1.0, 1.4, 2.0, 2.8, 4.0], 0.5, 20.0, false);
    save_gray(&membranes, format!("{dir}/cell-dark-lines.png"));
    println!("micrograph, dark lines, σ = 1 … 4, c = 20: vesselness {:.2} on the nuclear envelope at (94, 126), {:.2} inside the nucleus at (150, 150)", membranes[(94, 126)], membranes[(150, 150)]);
    println!("wrote the results to {dir}/");
}
