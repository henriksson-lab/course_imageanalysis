// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day26 track_analysis::

use imgcourse::Rng;
use day26::track_analysis::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn c1_path_stats_b5() {
    let track = [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0], [5.0, 2.0]];
    let (path, net, straight) = path_stats(&track);
    assert!(close(path, 3.0 + 2.0 * 2f32.sqrt(), 1e-4), "B5(a): path length {path}, expected 3 + 2√2 = 5.83");
    assert!(close(net, 29f32.sqrt(), 1e-4), "B5(a): net displacement {net}, expected √29 = 5.39");
    assert!(close(straight, 0.923, 1e-3), "B5(a): straightness {straight}, expected 0.92");
}

#[test]
fn c2_msd_b1() {
    // B1: the 1D track 0, 1, 3, 2, 4
    let track: Vec<Point> = [0.0, 1.0, 3.0, 2.0, 4.0].iter().map(|&x| [x, 0.0]).collect();
    let m = msd(&track, 4);
    let expected = [0.0, 2.5, 11.0 / 3.0, 6.5, 16.0];
    for lag in 0..5 {
        assert!(close(m[lag], expected[lag], 1e-4), "B1(a): MSD at lag {lag}: got {}, expected {}", m[lag], expected[lag]);
    }
}

#[test]
fn c3_fits_b4() {
    // B4(b): the exact MSD with σ = 30 nm, D = 0.01 µm²/s, Δt = 0.05 s, lags 1–4
    let tau = [0.05, 0.10, 0.15, 0.20];
    let m: Vec<f32> = tau.iter().map(|t| 4.0 * 0.01 * t + 4.0 * 0.03 * 0.03).collect();
    let slope = fit_through_origin(&tau, &m);
    assert!(close(slope / 4.0, 0.016, 5e-4), "B4(b): through the origin D = {}, expected 0.016", slope / 4.0);
    let (slope, intercept) = fit_line(&tau, &m);
    assert!(close(slope / 4.0, 0.010, 1e-5), "B4(b): free intercept D = {}, expected 0.010", slope / 4.0);
    assert!(close(intercept, 0.0036, 1e-6), "B4(b): intercept {intercept}, expected 4σ² = 0.0036");
}

#[test]
fn c4_random_walk_msd_is_4_d_dt() {
    // MSD(1) of a 2D random walk is 4 D Δt; with localisation error, 4 D Δt + 4 σ².
    let mut rng = Rng::new(1);
    let (d, dt) = (0.1, 0.1);
    for sigma in [0.0, 0.1] {
        let track = random_walk(20000, d, dt, sigma, &mut rng);
        assert_eq!(track.len(), 20000, "random_walk should return n positions");
        let m1 = msd(&track, 1)[1];
        let expected = 4.0 * d * dt + 4.0 * sigma * sigma;
        assert!(close(m1, expected, 0.05 * expected), "σ = {sigma}: MSD(1) = {m1}, expected about {expected}");
    }
}

#[test]
fn c5_speed() {
    // steps of length 1, 1 and √2 every 0.5 s
    let s = apparent_speed(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]], 0.5);
    assert!(close(s, (2.0 + 2f32.sqrt()) / 3.0 / 0.5, 1e-4), "mean step (2 + √2)/3 per 0.5 s: got {s}");
}

#[test]
fn c5_speed_trap_b3() {
    // B3: D = 0.1 at Δt = 0.1 s gives about √(π D/Δt) = 1.77 µm/s
    let mut rng = Rng::new(2);
    let track = random_walk(20000, 0.1, 0.1, 0.0, &mut rng);
    let s = apparent_speed(&track, 0.1);
    assert!(close(s, 1.77, 0.03), "B3: apparent speed {s}, expected about 1.77 µm/s");
}
