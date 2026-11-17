// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p pdesmooth heat_equation::

use pdesmooth::heat_equation::*;
use imgcourse::{samples, Border, Image};

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

fn assert_row(got: &[f32], expected: &[f32], what: &str) {
    let ok = got.len() == expected.len() && got.iter().zip(expected).all(|(g, e)| close(*g, *e));
    assert!(ok, "{what}: got {got:?}, expected {expected:?}");
}

#[test]
fn c1_one_step_b1() {
    let row = [0.0, 0.0, 4.0, 0.0, 0.0];
    assert_row(&heat_step_1d(&row, 0.25), &[0.0, 1.0, 2.0, 1.0, 0.0], "r = 1/4");
    assert_row(&heat_step_1d(&row, 0.5), &[0.0, 2.0, 0.0, 2.0, 0.0], "r = 1/2");
    let once = heat_step_1d(&row, 1.0);
    assert_row(&once, &[0.0, 4.0, -4.0, 4.0, 0.0], "r = 1");
    assert_row(&heat_step_1d(&once, 1.0), &[4.0, -8.0, 12.0, -8.0, 4.0], "r = 1, second step (B1 b)");
}

#[test]
fn c1_total_kept() {
    let row = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
    let total: f32 = heat_step_1d(&row, 0.3).iter().sum();
    assert!(close(total, 25.0), "mirror border: the total must stay 25, got {total}");
}

#[test]
fn c2_one_pixel_b2() {
    let mut img = Image::new(3, 3, 0.0);
    img[(1, 1)] = 8.0;
    let a = heat_step(&img, 0.2, Border::Zero)[(1, 1)];
    let b = heat_step(&img, 0.3, Border::Zero)[(1, 1)];
    assert!(close(a, 1.6), "dt = 0.2: got {a}, expected 1.6");
    assert!(close(b, -1.6), "dt = 0.3: got {b}, expected -1.6");
}

#[test]
fn c2_grid_b6() {
    // 7 × 7 zeros, 100 in the centre (3, 3), 20 above (3, 2) and left (2, 3) of it.
    let mut img = Image::new(7, 7, 0.0);
    img[(3, 3)] = 100.0;
    img[(3, 2)] = 20.0;
    img[(2, 3)] = 20.0;
    let out = heat_step(&img, 0.2, Border::Zero);
    assert!(close(out[(3, 3)], 28.0), "centre: got {}, expected 28", out[(3, 3)]);
    assert!(close(out[(3, 2)], 24.0), "above: got {}, expected 24", out[(3, 2)]);
    assert!(close(out[(4, 3)], 20.0), "right: got {}, expected 20", out[(4, 3)]);
    assert!(close(out.sum(), 140.0), "total: got {}, expected 140", out.sum());
    let c = heat_step(&img, 0.3, Border::Zero)[(3, 3)];
    assert!(close(c, -8.0), "dt = 0.3, centre: got {c}, expected -8");
}

#[test]
fn c2_mirror_keeps_total() {
    let img = samples::cell().crop(60, 60, 40, 30);
    let (before, after) = (img.sum(), heat_step(&img, 0.2, Border::Mirror).sum());
    assert!((before - after).abs() < 1e-2, "mirror border: total {before} before, {after} after");
}

#[test]
fn c3_heat_is_gaussian() {
    // 40 steps of 0.2: t = 8, so sigma = sqrt(2 t) = 4.
    let img = samples::cell().crop(64, 64, 96, 96);
    let heat = heat_flow(&img, 0.2, 40, Border::Mirror);
    let gauss = gaussian_blur(&img, 4.0);
    let diff = heat.zip_map(&gauss, |a, b| (a - b).abs()).max();
    assert!(diff < 2.0 / 255.0, "after t = 8 heat flow and a Gaussian of sigma 4 differ by {:.2} grey levels, expected below 2", diff * 255.0);
}

#[test]
fn c4_implicit_b5() {
    assert_row(&implicit_step_1d(&[0.0, 3.0, 0.0], 1.0), &[0.75, 1.5, 0.75], "(0, 3, 0) with dt = 1");
}

#[test]
fn c4_implicit_undoes_to_input() {
    // Plug the answer in (B5 b): u - dt * (Laplacian of u) must give the input back.
    let y = [5.0, 0.0, 2.0, 8.0, 1.0, 1.0];
    let dt = 3.0;
    let u = implicit_step_1d(&y, dt);
    let lap = heat_step_1d(&u, 1.0).iter().zip(&u).map(|(a, b)| a - b).collect::<Vec<_>>();
    let back: Vec<f32> = u.iter().zip(&lap).map(|(a, l)| a - dt * l).collect();
    assert_row(&back, &y, "u - dt * Laplacian(u)");
    assert!(u.iter().all(|&v| (0.0..=8.0).contains(&v)), "a large implicit step must stay between the old min and max: {u:?}");
}
