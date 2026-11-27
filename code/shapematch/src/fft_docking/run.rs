// Docking a key-shaped ligand into a receptor with a pocket, by one FFT correlation per angle.
// The shapes are synthetic 2D stand-ins for two molecules (the same as in the dock2d demo):
// the receptor is an ellipse with the key's shape cut out (plus a clearance of half a cell).
// Pictures in out/fft-docking/.
//   cargo run -p shapematch -- fft-docking                      (your code)
//   cargo run -p shapematch --features solution -- fft-docking  (the model solution)

use crate::fft_docking::*;
use imgcourse::fft::fft2;
use imgcourse::*;
use std::time::Instant;

const M: usize = 128; // FFT grid
const BOX: usize = 64; // the receptor's box, in the middle of the grid
const O: usize = (M - BOX) / 2;
const L: usize = 33; // ligand image size (centre cell 16)
const TRUE_ANGLE: f32 = 20.0;
const TRUE_X: f32 = 32.5; // world position of the key's centroid at the true pose
const TRUE_Y: f32 = 28.5;

/// The key, with its centroid at (0, 0): a cap with a stem that has a tooth on its right.
fn key() -> Vec<(f32, f32)> {
    let p = [(-8., -5.), (8., -5.), (8., 0.), (3., 0.), (3., 5.), (5.5, 5.), (5.5, 8.), (3., 8.), (3., 11.), (-3., 11.), (-3., 0.), (-8., 0.)];
    let (mut a, mut cx, mut cy) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..p.len() {
        let ((x0, y0), (x1, y1)): ((f64, f64), (f64, f64)) = (p[i], p[(i + 1) % p.len()]);
        let c = x0 * y1 - x1 * y0;
        a += c;
        cx += (x0 + x1) * c;
        cy += (y0 + y1) * c;
    }
    let (cx, cy) = (cx / (3.0 * a), cy / (3.0 * a));
    p.iter().map(|&(x, y)| ((x - cx) as f32, (y - cy) as f32)).collect()
}

fn inside(p: &[(f32, f32)], x: f32, y: f32) -> bool {
    let mut c = false;
    let mut j = p.len() - 1;
    for i in 0..p.len() {
        let ((xi, yi), (xj, yj)) = (p[i], p[j]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            c = !c;
        }
        j = i;
    }
    c
}

fn dist_to_outline(p: &[(f32, f32)], x: f32, y: f32) -> f32 {
    let mut best = f32::INFINITY;
    for i in 0..p.len() {
        let ((x0, y0), (x1, y1)) = (p[i], p[(i + 1) % p.len()]);
        let (dx, dy) = (x1 - x0, y1 - y0);
        let t = (((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
        best = best.min((x - x0 - t * dx).hypot(y - y0 - t * dy));
    }
    best
}

/// The receptor mask on the 128 × 128 grid: an ellipse minus the key at its true pose.
fn receptor_mask() -> Image<f32> {
    let (c, s) = (TRUE_ANGLE.to_radians().cos(), TRUE_ANGLE.to_radians().sin());
    let posed: Vec<(f32, f32)> = key().iter().map(|&(x, y)| (c * x - s * y + TRUE_X, s * x + c * y + TRUE_Y)).collect();
    Image::from_fn(M, M, |gx, gy| {
        if gx < O || gy < O || gx >= O + BOX || gy >= O + BOX {
            return 0.0;
        }
        let (x, y) = ((gx - O) as f32 + 0.5, (gy - O) as f32 + 0.5);
        let (ex, ey) = ((x - 32.0) / 24.0, (y - 40.0) / 14.0);
        let in_body = ex * ex + ey * ey <= 1.0;
        let in_pocket = inside(&posed, x, y) || dist_to_outline(&posed, x, y) < 0.5;
        if in_body && !in_pocket { 1.0 } else { 0.0 }
    })
}

/// The key as a 33 × 33 image, its centroid on the centre cell, at angle 0.
fn ligand_image() -> Image<f32> {
    let k = key();
    let c = (L / 2) as f32;
    Image::from_fn(L, L, |i, j| if inside(&k, i as f32 - c, j as f32 - c) { 1.0 } else { 0.0 })
}

/// World position of a grid cell's centre.
fn world(x: usize, y: usize) -> (f32, f32) {
    (x as f32 - O as f32 + 0.5, y as f32 - O as f32 + 0.5)
}

fn is_true(p: &Pose) -> bool {
    let (x, y) = world(p.x, p.y);
    let da = ((p.angle - TRUE_ANGLE + 540.0) % 360.0 - 180.0).abs();
    (x - TRUE_X).hypot(y - TRUE_Y) <= 1.5 && da <= 10.0
}

fn best_of(poses: &[Pose]) -> Pose {
    *poses.iter().max_by(|a, b| a.score.total_cmp(&b.score)).unwrap()
}

fn describe(p: &Pose) -> String {
    let (x, y) = world(p.x, p.y);
    format!("{:>5.0}° at ({x:.1}, {y:.1}), score {:.1}{}", p.angle, p.score, if is_true(p) { "  <- the true fit" } else { "" })
}

/// Contacts and clashes of a pose: ligand cells on the surface layer and in the core.
fn count(grid: &Image<f32>, lig: &Image<f32>, x: usize, y: usize) -> (usize, usize) {
    let (mut contact, mut clash) = (0, 0);
    for (i, j) in lig.coords() {
        if lig[(i, j)] > 0.0 {
            let u = (x + i + M - L / 2) % M;
            let w = (y + j + M - L / 2) % M;
            if grid[(u, w)] > 0.0 {
                contact += 1;
            } else if grid[(u, w)] < 0.0 {
                clash += 1;
            }
        }
    }
    (contact, clash)
}

/// The box of the grid (the receptor's 64 × 64 cells).
fn crop_box(img: &Image<f32>) -> Image<f32> {
    img.crop(O, O, BOX, BOX)
}

pub fn main() {
    let dir = "out/fft-docking";

    // By hand (B1): eight cells, ρ = −9, a two-cell ligand.
    let rho = -9.0;
    let a1 = receptor_grid(&Image::from_rows(&[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), 1, rho);
    let s1 = score_1d(a1.row(0), &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    println!("By hand: receptor a = {:?}", a1.row(0));
    println!("         ligand on cells d, d+1: S(d) = {s1:?}\n");

    // The receptor and the ligand.
    let mask = receptor_mask();
    let grid = receptor_grid(&mask, 2, rho);
    let lig = ligand_image();
    let core = grid.pixels().iter().filter(|&&v| v < 0.0).count();
    let surf = grid.pixels().iter().filter(|&&v| v > 0.0).count();
    let cells = lig.pixels().iter().filter(|&&v| v > 0.0).count();
    println!("Receptor: {core} core cells (ρ = {rho}), {surf} surface cells (2 cells thick), in a {M} × {M} grid");
    println!("Ligand: {cells} cells. True pose: {TRUE_ANGLE}° with its centre at ({TRUE_X}, {TRUE_Y})");
    let (tx, ty) = (O + TRUE_X as usize, O + TRUE_Y as usize);
    let at_true = rotate_ligand(&lig, TRUE_ANGLE);
    let (c, k) = count(&grid, &at_true, tx, ty);
    println!("  score at the true pose: {:.0} ({c} contacts, {k} clashes)\n", score_at(&grid, &at_true, tx, ty));

    // The search, 5° steps.
    let t0 = Instant::now();
    let poses = dock(&grid, &lig, 5.0);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let best = best_of(&poses);
    println!("Search in 5° steps: {} angles, {} FFTs of {M} × {M}, {ms:.0} ms", poses.len(), 1 + 2 * poses.len());
    println!("  best pose: {}", describe(&best));
    let other = poses.iter().filter(|p| ((p.angle - best.angle + 540.0) % 360.0 - 180.0).abs() > 20.0).copied().collect::<Vec<_>>();
    println!("  best pose more than 20° away from it: {}", describe(&best_of(&other)));
    print!("  best score every 30°:");
    for p in poses.iter().filter(|p| p.angle % 30.0 == 0.0) {
        print!(" {:.0}°: {:.0}", p.angle, p.score);
    }
    println!("\n");

    // The score maps at the right and at a wrong angle.
    let fa = fft2(&grid.map(Complex::from), false);
    for angle in [20.0, 110.0] {
        let map = score_map(&fa, &rotate_ligand(&lig, angle));
        println!("Score map at {angle:>3}°: highest {:.0}, lowest {:.0}", crop_box(&map).max(), crop_box(&map).min());
        save_gray_signed(&crop_box(&map).map(|v| v.max(-30.0)), format!("{dir}/map-{angle}.png"));
    }

    // Softer and harder cores, coarser angle steps.
    println!();
    for r in [-9.0, -1.0, -0.1] {
        let g = receptor_grid(&mask, 2, r);
        let b = best_of(&dock(&g, &lig, 5.0));
        let rot = rotate_ligand(&lig, b.angle);
        let (c, k) = count(&g, &rot, b.x, b.y);
        println!("ρ = {r:>4}: best pose {}  ({c} contacts, {k} clashes)", describe(&b));
    }
    for step in [15.0, 30.0] {
        let b = best_of(&dock(&grid, &lig, step));
        println!("step {step:>2}°: best pose {}", describe(&b));
    }
    let flat = best_of(&dock(&receptor_grid(&mask, 0, rho), &lig, 5.0));
    println!("no surface layer (thickness 0): the highest score anywhere is {:.1}", flat.score);

    // Brute force for one angle: the direct sum over the ligand's cells, for every shift.
    let rot = rotate_ligand(&lig, 20.0);
    let t0 = Instant::now();
    let offsets: Vec<(usize, usize)> = rot.coords().filter(|&(i, j)| rot[(i, j)] > 0.0).collect();
    let direct = Image::from_fn(M, M, |x, y| {
        let mut s = 0.0;
        for &(i, j) in &offsets {
            s += grid[((x + i + M - L / 2) % M, (y + j + M - L / 2) % M)];
        }
        s
    });
    let t_direct = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let fast = score_map(&fa, &rot);
    let t_fft = t0.elapsed().as_secs_f64() * 1000.0;
    let diff = direct.zip_map(&fast, |p, q| (p - q).abs()).max();
    println!("\nOne angle, all {} shifts: direct over the {cells} ligand cells {t_direct:.1} ms, FFT route {t_fft:.1} ms (largest difference {diff:.1e})", M * M);

    // Counting (B8 and the 3D slides).
    let flops = |n: f64| 5.0 * n * n.log2();
    let m2 = (M * M) as f64;
    let angles2d = 72.0;
    let fft_route = flops(m2) + angles2d * (2.0 * flops(m2) + 6.0 * m2);
    println!("\nCounting, 2D: {M}² grid, 5° steps = 72 angles");
    println!("  FFT route ≈ {fft_route:.2e} flops; dense brute force 72 · 2 · M² = {:.2e}; over the ligand cells only 72 · M · {cells} = {:.2e}",
        angles2d * 2.0 * m2 * m2, angles2d * m2 * cells as f64);
    let m3 = 128f64.powi(3);
    println!("Counting, 3D: one 128³ FFT ≈ {:.2e} flops", flops(m3));
    for step in [15.0f64, 10.0, 6.0] {
        let n = 8.0 * std::f64::consts::PI.powi(2) / step.to_radians().powi(3);
        println!("  {step:>2}° steps: 8π²/Δθ³ ≈ {n:.0} orientations, 2 FFTs each: {:.1e} flops", n * 2.0 * flops(m3));
    }
    println!("  10° at 20 Å from the centre moves an atom by {:.2} Å", 20.0 * 10f64.to_radians());

    // Pictures: the receptor's layers and the best pose.
    let rgb = Image::from_fn(BOX, BOX, |x, y| {
        let v = grid[(x + O, y + O)];
        if v < 0.0 { [0.29, 0.42, 0.66] } else if v > 0.0 { [0.96, 0.67, 0.27] } else { [0.98, 0.98, 0.98] }
    });
    save_rgb(&rgb, format!("{dir}/receptor.png"));
    let rot = rotate_ligand(&lig, best.angle);
    let mut pose = rgb.clone();
    for (i, j) in rot.coords() {
        if rot[(i, j)] > 0.0 {
            let (x, y) = (best.x + i - L / 2, best.y + j - L / 2);
            if x >= O && y >= O && x < O + BOX && y < O + BOX {
                let v = grid[(x, y)];
                pose[(x - O, y - O)] = if v < 0.0 { [0.84, 0.15, 0.16] } else if v > 0.0 { [0.17, 0.63, 0.17] } else { [0.6, 0.85, 0.6] };
            }
        }
    }
    save_rgb(&pose, format!("{dir}/best-pose.png"));
    println!("\nwrote the pictures to {dir}/");
}
