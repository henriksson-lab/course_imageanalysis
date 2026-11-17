// Runs your TV flow on plateaus, discs and a square; results in out/tv-flow/.
// This program works in grey levels 0-255, as the lecture does: under TV flow a plateau's
// height changes by (perimeter / area) grey levels per unit of time, so the time scale
// depends on the grey-value scale.
//   cargo run -p pdesmooth -- tv-flow                      (your code)
//   cargo run -p pdesmooth --features solution -- tv-flow  (the model solution)

use imgcourse::*;
use crate::tv_flow::*;

pub fn main() {
    let dir = "out/tv-flow";

    // 1D: plateaus of width 2, 4, 8 at 120 on 60; explicit steps with dt = eps / 2.5.
    for eps in [0.5f32, 0.05] {
        let mut row = vec![60.0f32; 200];
        for (start, w) in [(40, 2), (90, 4), (140, 8)] {
            row[start..start + w].iter_mut().for_each(|v| *v = 120.0);
        }
        let dt = eps / 2.5;
        let steps = (20.0 / dt).round() as usize;
        for _ in 0..steps {
            row = tv_step_1d(&row, dt, eps);
        }
        println!("1D, eps = {eps}: {steps} steps to t = 20; plateaus of width 2, 4, 8 at {:.1}, {:.1}, {:.1} (predicted 100, 110, 115)", row[41], row[92], row[144]);
    }

    // 2D: discs of radius 3, 4.5, 6, 9, 12 (contrast 40 on 60) and a square of side 48.
    let radii = [3.0f32, 4.5, 6.0, 9.0, 12.0];
    let centres: Vec<(f32, f32)> = (0..5).map(|i| (20.0 + 36.0 * i as f32, 24.0)).collect();
    let img = Image::from_fn(192, 112, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let in_disc = radii.iter().zip(&centres).any(|(r, (cx, cy))| (xf - cx).powi(2) + (yf - cy).powi(2) <= r * r);
        let in_square = (56..104).contains(&x) && (56..104).contains(&y);
        if in_disc || in_square { 100.0 } else { 60.0 }
    });
    let show = |u: &Image<f32>, name: &str| save_gray(&u.map(|v| v / 255.0), format!("{dir}/{name}.png"));
    show(&img, "shapes-t0");

    let (eps, dt) = (0.25, 0.05); // dt = eps / 5
    let mut u = img.clone();
    let mut t = 0.0;
    for step in 1..=4800 {
        u = tv_step(&u, dt, eps);
        t = step as f32 * dt;
        if [800, 1600, 2400, 4800].contains(&step) {
            show(&u, &format!("shapes-t{t}"));
            println!("t = {t:3}: disc contrast (predicted 40 - 2t/r, 0 when gone)");
            for (r, (cx, cy)) in radii.iter().zip(&centres) {
                let (x, y) = (*cx as usize, *cy as usize);
                let contrast = u[(x, y)] - u[(x, 4)]; // centre against the background above it
                println!("  radius {r:4}: {contrast:5.1}  ({:5.1})", (40.0 - 2.0 * t / r).max(0.0));
            }
        }
    }
    // The square's corner against its centre.
    println!("square at t = {t}: centre {:.1}, 4 px inside a corner {:.1}, background {:.1}", u[(80, 80)], u[(60, 60)], u[(80, 4)]);
    println!("wrote the pictures to {dir}/");
}
