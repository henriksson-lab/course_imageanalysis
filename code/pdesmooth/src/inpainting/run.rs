// Fills holes in the micrograph and in a dot texture with your code; results in out/inpainting/.
//   cargo run -p pdesmooth -- inpainting                      (your code)
//   cargo run -p pdesmooth --features solution -- inpainting  (the model solution)

use imgcourse::*;
use crate::inpainting::*;

/// The mean absolute error inside the hole, in grey levels (0-255).
fn hole_error(a: &Image<f32>, truth: &Image<f32>, missing: &Image<bool>) -> f32 {
    let (mut sum, mut n) = (0.0, 0.0);
    for (x, y) in a.coords() {
        if missing[(x, y)] {
            sum += (a[(x, y)] - truth[(x, y)]).abs();
            n += 1.0;
        }
    }
    255.0 * sum / n
}

/// The hole shown black on the image, for the pictures.
fn with_hole(img: &Image<f32>, missing: &Image<bool>) -> Image<f32> {
    img.zip_map(missing, |v, m| if m { 0.0 } else { v })
}

pub fn main() {
    let dir = "out/inpainting";
    let cell = samples::cell();

    // A dead column (x = 100) and a 3-pixel-wide scratch, filled harmonically.
    let missing = Image::from_fn(256, 256, |x, y| x == 100 || (y as isize - x as isize / 2 - 100).abs() <= 1);
    save_gray(&with_hole(&cell, &missing), format!("{dir}/defects.png"));
    let (filled, sweeps) = harmonic_inpaint(&cell, &missing, 1e-4);
    save_gray(&filled, format!("{dir}/defects-harmonic.png"));
    println!("dead column and scratch: {sweeps} Gauss–Seidel sweeps, mean error in the hole {:.1} grey levels", hole_error(&filled, &cell, &missing));

    // Square holes of width 8, 16, 32: sweeps until every pixel is within 1 grey level of
    // the exact fill, for Jacobi and Gauss–Seidel (both starting from 0 in the hole).
    println!("sweeps until within 1 grey level of the solution:");
    for w in [8, 16, 32] {
        let missing = Image::from_fn(256, 256, |x, y| (100..100 + w).contains(&x) && (100..100 + w).contains(&y));
        let (exact, _) = harmonic_inpaint(&cell, &missing, 1e-7);
        let start = with_hole(&cell, &missing);
        let far = |u: &Image<f32>| 255.0 * u.zip_map(&exact, |a, b| (a - b).abs()).max() > 1.0;
        let mut u = start.clone();
        let mut jacobi = 0;
        while far(&u) {
            u = jacobi_sweep(&u, &missing);
            jacobi += 1;
        }
        let mut u = start.clone();
        let mut gs = 0;
        while far(&u) {
            gauss_seidel_sweep(&mut u, &missing);
            gs += 1;
        }
        println!("  hole {w:2} × {w:2}: Jacobi {jacobi:4}, Gauss–Seidel {gs:4}");
        if w == 32 {
            save_gray(&with_hole(&cell, &missing), format!("{dir}/hole32.png"));
            save_gray(&exact, format!("{dir}/hole32-harmonic.png"));
        }
    }

    // A dot texture with a 28 × 28 hole: harmonic against exemplar inpainting.
    let truth = Image::from_fn(96, 96, |x, y| {
        let (dx, dy) = ((x % 8) as f32 - 3.5, (y % 8) as f32 - 3.5);
        if dx * dx + dy * dy < 6.0 { 0.85 } else { 0.25 }
    });
    let truth = samples::add_noise(&truth, 0.02, 5);
    let hole = Image::from_fn(96, 96, |x, y| (34..62).contains(&x) && (34..62).contains(&y));
    save_gray(&with_hole(&truth, &hole), format!("{dir}/texture-hole.png"));
    let (harmonic, _) = harmonic_inpaint(&truth, &hole, 1e-4);
    save_gray(&harmonic, format!("{dir}/texture-harmonic.png"));

    // Exemplar: fill, patch by patch, the front pixel whose 9 × 9 patch has the most known
    // pixels (the "confidence" of the lecture, without the data term).
    let half = 4;
    let (mut img, mut missing) = (with_hole(&truth, &hole), hole.clone());
    let mut patches = 0;
    while missing.pixels().iter().any(|&m| m) {
        let mut best = (0, (0, 0));
        for (x, y) in img.coords() {
            if !missing[(x, y)] || x < half || y < half || x + half >= 96 || y + half >= 96 {
                continue;
            }
            let known = (y - half..=y + half).map(|yy| (x - half..=x + half).filter(|&xx| !missing[(xx, yy)]).count()).sum::<usize>();
            if known > best.0 {
                best = (known, (x, y));
            }
        }
        fill_best_patch(&mut img, &mut missing, best.1, half, 40);
        patches += 1;
    }
    save_gray(&img, format!("{dir}/texture-exemplar.png"));
    println!(
        "dot texture, 28 × 28 hole: harmonic error {:.1}, exemplar error {:.1} grey levels ({patches} patches)",
        hole_error(&harmonic, &truth, &hole),
        hole_error(&img, &truth, &hole)
    );
    println!("wrote the pictures to {dir}/");
}
