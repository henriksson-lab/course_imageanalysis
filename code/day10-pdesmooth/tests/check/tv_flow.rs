// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day10 tv_flow::

use imgcourse::Image;
use day10::tv_flow::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_step_or_ramp_b3() {
    let step = [0.0, 10.0];
    let ramp: Vec<f32> = (0..=10).map(|i| i as f32).collect();
    let cases = [
        ("heat", Penalty::Heat, 50.0, 5.0),
        ("TV", Penalty::Tv, 10.0, 10.0),
        ("Perona–Malik, K = 2", Penalty::PeronaMalik(2.0), 6.516, 4.463),
    ];
    for (name, p, e_step, e_ramp) in cases {
        let (a, b) = (energy_1d(&step, p), energy_1d(&ramp, p));
        assert!(close(a, e_step, 1e-2), "{name}, one step of 10: got {a}, expected {e_step}");
        assert!(close(b, e_ramp, 1e-2), "{name}, ten steps of 1: got {b}, expected {e_ramp}");
    }
}

/// Run explicit TV flow on a row up to time `t`, with `dt = eps / 2.5`.
fn flow_1d(row: &[f32], t: f32, eps: f32) -> Vec<f32> {
    let dt = eps / 2.5;
    let mut u = row.to_vec();
    for _ in 0..(t / dt).round() as usize {
        u = tv_step_1d(&u, dt, eps);
    }
    u
}

#[test]
fn c2_by_hand_b1() {
    let y = [0.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0, 0.0];
    let u = flow_1d(&y, 2.0, 0.01);
    let expected = [2. / 3., 2. / 3., 2. / 3., 4., 4., 2. / 3., 2. / 3., 2. / 3.];
    let ok = u.iter().zip(expected).all(|(a, b)| close(*a, b, 0.05));
    assert!(ok, "B1, t = 2: got {u:?}, expected {expected:?}");
    let mean: f32 = u.iter().sum::<f32>() / 8.0;
    assert!(close(mean, 1.5, 1e-3), "the mean must stay 1.5, got {mean}");
}

#[test]
fn c2_plateaus_sink_at_2_over_width() {
    // Plateaus of width 2, 4 and 8 at 120 on 60, far apart (demo E1): at t = 40 they are at
    // 120 - 2t/w = 80, 100, 110.
    let mut row = vec![60.0; 200];
    for (start, w) in [(40, 2), (90, 4), (140, 8)] {
        for v in &mut row[start..start + w] {
            *v = 120.0;
        }
    }
    let u = flow_1d(&row, 40.0, 0.5);
    for (start, w, expected) in [(40, 2, 80.0), (90, 4, 100.0), (140, 8, 110.0)] {
        let got = u[start + w / 2];
        assert!(close(got, expected, 1.5), "plateau of width {w}: got {got:.1}, expected {expected}");
    }
}

#[test]
fn c3_disc_fades_at_2_over_r() {
    // A disc of radius 8 and contrast 40: after t = 40 the contrast should be about 40 - 2t/r = 30.
    let img = Image::from_fn(40, 40, |x, y| {
        let (dx, dy) = (x as f32 - 19.5, y as f32 - 19.5);
        if dx * dx + dy * dy <= 64.0 { 100.0 } else { 60.0 }
    });
    let (eps, t) = (1.0, 40.0);
    let dt = eps / 5.0;
    let mut u = img.clone();
    for _ in 0..(t / dt) as usize {
        u = tv_step(&u, dt, eps);
    }
    let contrast = u[(20, 20)] - u[(2, 2)];
    assert!(close(contrast, 30.0, 4.0), "disc of radius 8 after t = 40: contrast {contrast:.1}, expected about 30");
    let edge = u[(20, 20)] - u[(20, 31)]; // 3 pixels outside the disc
    assert!(edge > 0.8 * contrast, "the edge must stay sharp: centre - 3 px outside = {edge:.1} of a contrast {contrast:.1}");
}

#[test]
fn c3_total_kept() {
    let img = Image::from_fn(16, 12, |x, y| ((x * 7 + y * 13) % 10) as f32);
    let u = tv_step(&img, 0.1, 0.5);
    assert!(close(u.sum(), img.sum(), 1e-2), "no flow across the border: total {} before, {} after", img.sum(), u.sum());
}
