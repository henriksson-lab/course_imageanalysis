// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p day28 gmrf::

use day28::gmrf::*;
use imgcourse::{samples, Image};

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn assert_signal(got: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: got {} values, expected {}", got.len(), expected.len());
    for p in 0..got.len() {
        assert!(close(got[p], expected[p], tol), "{what}: got {got:.3?}, expected {expected:?}");
    }
}

#[test]
fn c1_penalties_b2() {
    // from 0 to 3 in one jump, in 3 steps of 1, in 30 steps of 0.1
    let cost = |steps: usize, rho| (0..steps).map(|_| penalty(3.0 / steps as f32, rho)).sum::<f32>();
    let quad = [cost(1, Penalty::Quadratic), cost(3, Penalty::Quadratic), cost(30, Penalty::Quadratic)];
    let tv = [cost(1, Penalty::Tv), cost(3, Penalty::Tv), cost(30, Penalty::Tv)];
    assert_signal(&quad, &[4.5, 1.5, 0.15], 1e-4, "B2, quadratic cost of 1, 3, 30 steps");
    assert_signal(&tv, &[3.0, 3.0, 3.0], 1e-4, "B2, TV cost of 1, 3, 30 steps");
    let hub = [penalty(0.5, Penalty::Huber(1.0)), penalty(-3.0, Penalty::Huber(1.0))];
    assert_signal(&hub, &[0.125, 2.5], 1e-6, "Huber δ = 1 at d = 0.5 and d = −3");
}

#[test]
fn c1_energy_b1_b3() {
    let y = [0.0, 3.0, 0.0];
    let e = [energy_1d(&[0.75, 1.5, 0.75], &y, 1.0, Penalty::Quadratic), energy_1d(&y, &y, 1.0, Penalty::Quadratic), energy_1d(&[1.0, 1.0, 1.0], &y, 1.0, Penalty::Quadratic)];
    assert_signal(&e, &[2.25, 9.0, 3.0], 1e-5, "B1(c), energies of the solution, of y and of (1, 1, 1)");
    let y = [0.0, 0.0, 3.0, 3.0];
    let e = [energy_1d(&[0.5, 0.5, 2.5, 2.5], &y, 1.0, Penalty::Tv), energy_1d(&y, &y, 1.0, Penalty::Tv)];
    assert_signal(&e, &[2.5, 3.0], 1e-5, "B3(a), TV energies of the solution and of y");
}

#[test]
fn c2_quadratic_b1() {
    assert_signal(&quadratic_1d(&[0.0, 3.0, 0.0], 1.0), &[0.75, 1.5, 0.75], 1e-4, "B1(a), λ = 1");
    assert_signal(&quadratic_1d(&[0.0, 3.0, 0.0], 2.0), &[0.857, 1.286, 0.857], 1e-3, "B1(b), λ = 2");
    assert_signal(&quadratic_1d(&[0.0, 0.0, 3.0, 3.0], 1.0), &[0.43, 0.86, 2.14, 2.57], 0.006, "B3(c), quadratic on the step");
}

#[test]
fn c2_quadratic_keeps_the_mean_and_is_the_minimum() {
    let y: Vec<f32> = (0..50).map(|p| ((p * 37) % 11) as f32 / 10.0).collect();
    let x = quadratic_1d(&y, 3.0);
    let (sy, sx) = (y.iter().sum::<f32>(), x.iter().sum::<f32>());
    assert!(close(sx, sy, 1e-3), "the sum should stay {sy}, got {sx}");
    let e = energy_1d(&x, &y, 3.0, Penalty::Quadratic);
    for p in [0, 17, 49] {
        for d in [-0.01, 0.01] {
            let mut x2 = x.clone();
            x2[p] += d;
            assert!(energy_1d(&x2, &y, 3.0, Penalty::Quadratic) >= e - 1e-5, "moving sample {p} by {d} lowered the energy: not the minimum");
        }
    }
}

#[test]
fn c3_tv_step_b3() {
    let y = [0.0, 0.0, 3.0, 3.0];
    assert_signal(&tv_1d(&y, 1.0, 2000), &[0.5, 0.5, 2.5, 2.5], 1e-3, "B3(a), TV λ = 1");
    assert_signal(&tv_1d(&y, 3.5, 2000), &[1.5, 1.5, 1.5, 1.5], 1e-3, "B3(b), TV λ = 3.5 (above 3: flat)");
}

#[test]
fn c3_tv_peak_b4() {
    let y = [0.0, 3.0, 0.0];
    assert_signal(&tv_1d(&y, 0.5, 2000), &[0.5, 2.0, 0.5], 1e-3, "B4(a), TV λ = 0.5");
    assert_signal(&tv_1d(&y, 1.0, 2000), &[1.0, 1.0, 1.0], 1e-3, "B4(a), TV λ = 1");
}

#[test]
fn c4_quadratic_2d_equals_1d_on_a_row() {
    let y: Vec<f32> = (0..20).map(|p| if p < 8 { 0.0 } else { 1.0 }).collect();
    let img = Image::from_vec(20, 1, y.clone());
    let x2 = quadratic_2d(&img, 2.0, 500);
    assert_signal(x2.pixels(), &quadratic_1d(&y, 2.0), 1e-4, "a one-row image should give the 1D result");
}

#[test]
fn c4_quadratic_2d_keeps_the_mean() {
    let img = samples::cell().crop(60, 60, 24, 24);
    let x = quadratic_2d(&img, 1.0, 200);
    assert!(close(x.mean(), img.mean(), 1e-4), "the mean should stay {}, got {}", img.mean(), x.mean());
    assert!(x.max() < img.max() && x.min() > img.min(), "smoothing should narrow the range of values");
}

#[test]
fn c5_tv_2d_equals_1d_on_a_row() {
    let y = [0.0, 0.0, 3.0, 3.0];
    let x = tv_2d(&Image::from_rows(&[y]), 1.0, 4000);
    assert_signal(x.pixels(), &[0.5, 0.5, 2.5, 2.5], 1e-3, "B3(a) as a one-row image");
}

#[test]
fn c5_tv_2d_keeps_a_square_sharp() {
    // a bright 4 × 4 square in an 8 × 8 image: TV lowers it by λ · perimeter / area = λ · 16/16
    let y = Image::from_fn(8, 8, |x, y| if (2..6).contains(&x) && (2..6).contains(&y) { 4.0 } else { 0.0 });
    let x = tv_2d(&y, 0.5, 4000);
    assert!(close(x[(3, 3)], 3.5, 0.01), "inside the square: got {}, expected 4 − 0.5 = 3.5", x[(3, 3)]);
    assert!(close(x[(2, 2)], x[(3, 3)], 0.01), "the square should stay flat: corner {} centre {}", x[(2, 2)], x[(3, 3)]);
    assert!(x[(1, 3)] < 0.2, "outside next to the edge: got {}, expected a value near 0 (the edge stays sharp)", x[(1, 3)]);
}
