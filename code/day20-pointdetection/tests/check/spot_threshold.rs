// Checks for the coding exercises, using the numbers of the pen-and-paper exercises and the lecture.
// Run: cargo test -p day20 spot_threshold::

use imgcourse::{Image, Rng};
use day20::spot_threshold::sim::{filter_noise_factor, gaussian_blur, noise_image, poisson};
use day20::spot_threshold::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn c1_mad_b1() {
    let v = [3., -1., 0., 2., -2., 1., 25., -1., 0.];
    let s = mad_sigma(&v);
    assert!(close(s, 1.0 / 0.6745, 1e-3), "B1(a): got {s}, expected 1/0.6745 = 1.48");
    // B1(c): four of nine values may be spots without breaking the estimate.
    let w = [3., -1., 0., 2., 900., 800., 700., 600., 0.];
    let s = mad_sigma(&w);
    assert!(s < 10.0, "B1(c): with four huge values the estimate should stay small (the ordinary sd is about 380), got {s}");
    let even = [1., 2., 3., 4.];
    assert!(close(mad_sigma(&even), 1.0 / 0.6745, 1e-3), "even count: median of 1, 2, 3, 4 is 2.5, of the deviations 1");
}

#[test]
fn c2_noise_from_the_image() {
    // noise sd 0.05 on a steep ramp with a few bright squares: the estimate should ignore both
    let noise = noise_image(128, 0.0, 0.05, 7);
    let img = Image::from_fn(128, 128, |x, y| {
        let spot = if (x / 16 + y / 16) % 5 == 0 && x % 16 < 3 && y % 16 < 3 { 0.5 } else { 0.0 };
        0.2 + 0.004 * x as f32 + spot + noise[(x, y)]
    });
    let s = estimate_noise(&img);
    assert!(close(s, 0.05, 0.003), "noise sd 0.05 on a ramp with spots: got {s}");
}

#[test]
fn c3_maxima_small() {
    let z = Image::from_rows(&[
        [0., 0., 0., 0., 9.],
        [0., 5., 1., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 0., 3., 0.],
        [0., 0., 0., 0., 0.],
    ]);
    assert_eq!(maxima_above(&z, f32::NEG_INFINITY), vec![(1, 1), (3, 3)], "maxima (the 9 on the border is skipped)");
    assert_eq!(maxima_above(&z, 4.0), vec![(1, 1)], "maxima above 4");
}

#[test]
fn c3_noise_maxima_lecture() {
    // The lecture: white noise smoothed with σ_f = 1.5 has about one maximum per 24 σ_f² pixels,
    // about 1 220 per 256 × 256 image.
    let mut total = 0;
    for seed in 0..4 {
        let z = gaussian_blur(&noise_image(256, 0.0, 1.0, seed), 1.5);
        total += maxima_above(&z, f32::NEG_INFINITY).len();
    }
    let per_image = total as f32 / 4.0;
    assert!((1150.0..1300.0).contains(&per_image), "maxima per 256² noise image: got {per_image}, expected about 1 220");
}

#[test]
fn c3_false_spots_at_k3() {
    // About 26 false spots per 256² image at k = 3 (lecture demo, 16 seeded images).
    let mut above = 0;
    for seed in 0..8 {
        let z = gaussian_blur(&noise_image(256, 0.0, 1.0, 100 + seed), 1.5).map(|v| v / filter_noise_factor(1.5, 0.0));
        above += maxima_above(&z, 3.0).len();
    }
    let per_image = above as f32 / 8.0;
    assert!((15.0..40.0).contains(&per_image), "false spots per 256² image at k = 3: got {per_image}, expected about 26");
}

#[test]
fn c4_euler_lecture_and_b8() {
    let a = 512.0 * 512.0;
    for (u, e, tol) in [(3.0, 123.0, 1.0), (4.0, 5.0, 0.1), (5.0, 0.07, 0.01)] {
        let got = euler_expected(a, 1.5, u);
        assert!(close(got, e, tol), "512², σ_f = 1.5, u = {u}: got {got}, expected {e}");
    }
    let a = 1024.0 * 1024.0;
    for (u, e, tol) in [(4.0, 11.2, 0.1), (4.5, 1.50, 0.02), (5.0, 0.16, 0.01)] {
        let got = euler_expected(a, 2.0, u);
        assert!(close(got, e, tol), "B8, 1024², σ_f = 2, u = {u}: got {got}, expected {e}");
    }
}

#[test]
fn c5_anscombe_b9() {
    let mut rng = Rng::new(5);
    for (mu, tol) in [(100.0f32, 0.05), (20.0, 0.06)] {
        let counts = Image::from_fn(200, 100, |_, _| poisson(&mut rng, mu));
        let t = anscombe(&counts);
        let mean = t.mean();
        let var = t.pixels().iter().map(|v| (v - mean).powi(2)).sum::<f32>() / t.pixels().len() as f32;
        assert!(close(var, 1.0, tol), "B9: Poisson counts with mean {mu}: variance after Anscombe got {var}, expected about 1");
    }
    assert!(close(anscombe(&Image::new(1, 1, 0.0))[(0, 0)], 2.0 * 0.375f32.sqrt(), 1e-6), "anscombe(0) = 2 √(3/8)");
}
