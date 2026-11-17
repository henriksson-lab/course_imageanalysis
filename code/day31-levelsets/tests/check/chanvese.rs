// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day31 chanvese::

use day31::chanvese::*;
use imgcourse::Image;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

/// The eight pixels of B1.
fn eight() -> Image<f32> {
    Image::from_rows(&[[1.0, 1.0, 2.0, 6.0, 7.0, 7.0, 6.0, 1.0]])
}

/// "Inside" = the pixels `from..=to`, counted from 1 as in the exercise.
fn inside(from: usize, to: usize) -> Image<bool> {
    Image::from_fn(8, 1, |x, _| (from - 1..to).contains(&x))
}

fn as_phi(inside: &Image<bool>) -> Image<f32> {
    inside.map(|i| if i { -1.0 } else { 1.0 })
}

#[test]
fn c1_means_b1() {
    let (c1, c2) = region_means(&eight(), &as_phi(&inside(4, 7)));
    assert!(close(c1, 6.5, 1e-5) && close(c2, 1.25, 1e-5), "B1(a): got c1 = {c1}, c2 = {c2}, expected 6.5 and 1.25");
    let (c1, c2) = region_means(&eight(), &as_phi(&inside(3, 7)));
    assert!(close(c1, 5.6, 1e-5) && close(c2, 1.0, 1e-5), "B1(b): got c1 = {c1}, c2 = {c2}, expected 5.6 and 1");
}

#[test]
fn c2_energy_b1() {
    let img = eight();
    let all = Image::new(8, 1, true);
    for (name, region, e0, e1) in [
        ("(a) 4–7", inside(4, 7), 1.75, 3.75),
        ("(b) 3–7", inside(3, 7), 17.2, 19.2),
        ("(c) 4–6", inside(4, 6), 19.467, 21.467),
        ("(d) all", all, 56.875, 56.875),
    ] {
        let (g0, g1) = (energy(&img, &region, 0.0), energy(&img, &region, 1.0));
        assert!(close(g0, e0, 2e-3) && close(g1, e1, 2e-3), "B1{name}: E = {g0} (λ = 0) and {g1} (λ = 1), expected {e0} and {e1}");
    }
}

#[test]
fn c2_energy_b3() {
    let img = Image::from_rows(&[[1.0, 1.0, 1.0, 6.0, 6.0, 3.0, 6.0, 1.0]]);
    let without = Image::from_fn(8, 1, |x, _| [3, 4, 6].contains(&x)); // pixels 4, 5, 7
    let with = inside(4, 7);
    for lambda in [0.0, 1.0, 2.0] {
        let (a, b) = (energy(&img, &without, lambda), energy(&img, &with, lambda));
        assert!(close(a, 3.2 + 4.0 * lambda, 1e-4), "B3(a), λ = {lambda}, the 3 left out: E = {a}, expected {}", 3.2 + 4.0 * lambda);
        assert!(close(b, 6.75 + 2.0 * lambda, 1e-4), "B3(a), λ = {lambda}, the 3 filled in: E = {b}, expected {}", 6.75 + 2.0 * lambda);
    }
}

#[test]
fn c3_which_way_b2() {
    // Flat φ (no curvature), λ = ν = 0: a pixel of grey level 4 joins the inside (φ goes down),
    // one of grey level 2 leaves it (φ goes up).
    let img = Image::from_rows(&[[4.0, 2.0]]);
    let phi = Image::new(2, 1, 0.5);
    let new = chan_vese_step(&img, &phi, 6.5, 1.25, Params { lambda: 0.0, nu: 0.0, dt: 0.1, eps: 1.0 });
    assert!(new[(0, 0)] < 0.5, "B2, grey level 4: φ went from 0.5 to {}, expected it to go down", new[(0, 0)]);
    assert!(new[(1, 0)] > 0.5, "B2, grey level 2: φ went from 0.5 to {}, expected it to go up", new[(1, 0)]);
    // Size of the step: δ_1(0.5) = 1 / (π · 1.25), times −1.3125, times Δt = 0.1.
    let expected = 0.5 + 0.1 * (1.0 / (std::f32::consts::PI * 1.25)) * (6.25 - 7.5625);
    assert!(close(new[(0, 0)], expected, 1e-5), "B2, grey level 4: got {}, expected {expected}", new[(0, 0)]);
}

#[test]
fn c4_lambda_zero_is_thresholding_b4() {
    // Start with only pixel 5 inside; with λ = 0 the result is the threshold at (c1 + c2)/2:
    // pixels 4–7 inside, with the means 6.5 and 1.25.
    let start = Image::from_fn(8, 1, |x, _| if x == 4 { -1.0 } else { 1.0 });
    let (phi, c1, c2) = chan_vese(&eight(), &start, Params { lambda: 0.0, nu: 0.0, dt: 0.5, eps: 1.0 }, 300);
    let got: Vec<bool> = phi.row(0).iter().map(|&v| v < 0.0).collect();
    let expected = [false, false, false, true, true, true, true, false];
    assert_eq!(got, expected, "B4: inside after 300 iterations (pixels 1 to 8), expected pixels 4–7");
    assert!(close(c1, 6.5, 1e-4) && close(c2, 1.25, 1e-4), "B4: means {c1} and {c2}, expected 6.5 and 1.25");
}
