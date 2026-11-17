// Moves curves with your level set steps: merging, splitting, a too large step and curvature
// flow; pictures in out/levelset-eq/.
//   cargo run -p day31 -- levelset-eq                      (your code)
//   cargo run -p day31 --features solution -- levelset-eq  (the model solution)

use imgcourse::*;
use crate::levelset_eq::*;
use crate::common::{draw_contour, pieces, sheet};

const SIZE: usize = 128;

fn disk(cx: f32, cy: f32, r: f32) -> Image<f32> {
    Image::from_fn(SIZE, SIZE, |x, y| ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() - r)
}

/// Signed distance of an axis-parallel box with centre `(cx, cy)` and half-sizes `hx`, `hy`.
fn rect(cx: f32, cy: f32, hx: f32, hy: f32) -> Image<f32> {
    Image::from_fn(SIZE, SIZE, |x, y| {
        let (dx, dy) = ((x as f32 - cx).abs() - hx, (y as f32 - cy).abs() - hy);
        (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0)
    })
}

fn union(a: &Image<f32>, b: &Image<f32>) -> Image<f32> {
    a.zip_map(b, |p, q| p.min(q))
}

fn area(phi: &Image<f32>) -> usize {
    phi.pixels().iter().filter(|&&v| v < 0.0).count()
}

const COLOURS: [[f32; 3]; 4] = [[0.0, 0.0, 0.0], [0.1, 0.3, 0.9], [0.0, 0.6, 0.2], [0.9, 0.1, 0.1]];

/// Move φ with constant speed `f` and step `dt` up to time `t_end`; report when the number of
/// pieces changes, and draw the contour at the times `snapshots`.
fn grass_fire(name: &str, phi0: &Image<f32>, f: f32, dt: f32, t_end: f32, snapshots: &[f32]) -> Image<[f32; 3]> {
    let speed = Image::new(SIZE, SIZE, f);
    let mut phi = phi0.clone();
    let mut pic = Image::new(SIZE, SIZE, [1.0f32; 3]);
    let mut n = pieces(&phi, 0.0);
    let steps = (t_end / dt).round() as usize;
    println!("{name}: {n} piece(s) at t = 0");
    for step in 0..=steps {
        let t = step as f32 * dt;
        if let Some(k) = snapshots.iter().position(|&s| (s - t).abs() < dt / 2.0) {
            draw_contour(&mut pic, &phi, COLOURS[k % 4]);
        }
        if step == steps {
            break;
        }
        phi = level_set_step(&phi, &speed, dt);
        if (step + 1) % 10 == 0 {
            phi = reinitialise(&phi);
        }
        let m = pieces(&phi, 0.0);
        if m != n {
            println!("{name}: {m} piece(s) from t = {:.1}", t + dt);
            n = m;
        }
    }
    pic
}

pub fn main() {
    let dir = "out/levelset-eq";
    let limit = 1.0 / 2f32.sqrt(); // CFL in 2D for |F| = 1

    // Two disks of radius 18, centres 48 apart, growing with F = 1 (B6).
    let two = union(&disk(40.0, 64.0, 18.0), &disk(88.0, 64.0, 18.0));
    let merge = grass_fire("two disks, F = 1", &two, 1.0, 0.5, 12.0, &[0.0, 4.0, 8.0, 12.0]);
    // A dumbbell with a bar 12 pixels thick, shrinking with F = −1.
    let dumbbell = union(&union(&disk(36.0, 64.0, 22.0), &disk(92.0, 64.0, 22.0)), &rect(64.0, 64.0, 30.0, 6.0));
    let split = grass_fire("dumbbell, F = -1", &dumbbell, -1.0, 0.5, 12.0, &[0.0, 4.0, 8.0, 12.0]);

    // A square of side 64 grown with F = 1 for t = 16: rounded corners (B4).
    let square = rect(64.0, 64.0, 32.0, 32.0);
    let grown = grass_fire("square, F = 1", &square, 1.0, 0.5, 16.0, &[0.0, 8.0, 16.0]);

    // A too large step: 1.5 times the CFL limit.
    let mut phi = disk(64.0, 64.0, 30.0);
    let speed = Image::new(SIZE, SIZE, 1.0);
    for step in 1..=30 {
        phi = level_set_step(&phi, &speed, 1.5 * limit);
        if step % 5 == 0 {
            let m = phi.pixels().iter().fold(0.0f32, |m, v| m.max(v.abs()));
            println!("dt = 1.5 x limit, step {step}: largest |phi| = {m:.0}");
        }
    }
    let mut explode = gray_to_rgb(&phi.map(|v| if v.abs() > 1e3 { 0.0 } else { 1.0 }));
    draw_contour(&mut explode, &phi, COLOURS[3]);
    let mut ok = disk(64.0, 64.0, 30.0);
    for _ in 1..=30 {
        ok = level_set_step(&ok, &speed, limit);
    }
    println!("dt = limit, step 30: largest |phi| = {:.0}", ok.pixels().iter().fold(0.0f32, |m, v| m.max(v.abs())));

    // Curvature flow (ε = 1, Δt = 0.25) of a five-armed star, reinitialised every 20 steps.
    let star = Image::from_fn(SIZE, SIZE, |x, y| {
        let (dx, dy) = (x as f32 - 64.0, y as f32 - 64.0);
        (dx * dx + dy * dy).sqrt() - 36.0 * (1.0 + 0.3 * (5.0 * dy.atan2(dx)).cos())
    });
    let mut phi = reinitialise(&star);
    let mut flow = Image::new(SIZE, SIZE, [1.0f32; 3]);
    let a0 = area(&phi) as f32;
    for step in 0..=1000 {
        let t = step as f32 * 0.25;
        if let Some(k) = [0.0, 30.0, 100.0, 250.0].iter().position(|&s| s == t) {
            draw_contour(&mut flow, &phi, COLOURS[k]);
            println!("star, curvature flow, t = {t:.0}: {} pixels inside ({} piece(s)); area lost per unit time so far {:.2}", area(&phi), pieces(&phi, 0.0), if t > 0.0 { (a0 - area(&phi) as f32) / t } else { 0.0 });
        }
        phi = curvature_step(&phi, 1.0, 0.25);
        if (step + 1) % 20 == 0 {
            phi = reinitialise(&phi);
        }
    }

    save_rgb(&sheet(&[merge, split, grown], 0.5), format!("{dir}/fronts.png"));
    save_rgb(&sheet(&[explode, flow], 0.5), format!("{dir}/cfl-curvature.png"));
    println!("wrote the pictures to {dir}/");
}
