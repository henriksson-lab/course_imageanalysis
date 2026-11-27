// Checks for the coding exercises of "Rotation and scale become shifts: Fourier–Mellin",
// partly with the numbers of the pen-and-paper exercises.
// Run: cargo test -p shapematch fourier_mellin::

use imgcourse::{samples, Image};
use shapematch::fourier_mellin::*;

const BASE: Settings = Settings { taper: true, highpass: true, both: true, n_theta: 256, n_r: 128, r_min: 2.0, r_max: 62.0 };

fn wrap_deg(d: f32) -> f32 {
    (d.rem_euclid(360.0) + 180.0).rem_euclid(360.0) - 180.0
}

#[test]
fn c1_polar_reads_along_rays() {
    // f(x, y) = x − 16: on the ray of angle θ at radius r the value is r cos θ.
    let img = Image::from_fn(33, 33, |x, _| x as f32 - 16.0);
    let p = polar(&img, 16.0, 16.0, 12.0, 13, 8);
    assert_eq!((p.width(), p.height()), (8, 13), "n_theta columns, n_r rows");
    for j in 0..8 {
        let theta = 2.0 * std::f32::consts::PI * j as f32 / 8.0;
        for i in 0..13 {
            let expected = i as f32 * theta.cos();
            assert!((p[(j, i)] - expected).abs() < 1e-3, "column {j} row {i}: got {}, expected r cos θ = {expected}", p[(j, i)]);
        }
    }
}

#[test]
fn c1_a_quarter_turn_shifts_the_columns() {
    // Turning the image by 90° about the centre moves every ring by n_theta / 4 columns.
    let cell = samples::cell().crop(60, 60, 65, 65);
    let turned = Image::from_fn(65, 65, |x, y| cell[(y, 64 - x)]); // 90° clockwise on screen
    let (pa, pb) = (polar(&cell, 32.0, 32.0, 30.0, 31, 16), polar(&turned, 32.0, 32.0, 30.0, 31, 16));
    for i in 0..31 {
        for j in 0..16 {
            assert!((pb[((j + 4) % 16, i)] - pa[(j, i)]).abs() < 1e-4, "ring {i}: B at column {} should equal A at column {j}", (j + 4) % 16);
        }
    }
}

#[test]
fn c2_one_ring_by_hand() {
    // B4: A = (2, 0, −1, −1), B = A turned by one column (90°): score (−1, 6, −1, −4).
    let a = Image::from_rows(&[[2.0, 0.0, -1.0, -1.0]]);
    let b = Image::from_rows(&[[-1.0, 2.0, 0.0, -1.0]]);
    let s = rotational_correlation(&a, &b, &[1.0]);
    for (t, e) in [-1.0, 6.0, -1.0, -4.0].iter().enumerate() {
        assert!((s[t] - e).abs() < 1e-4, "score[{t}] = {}, expected {e}", s[t]);
    }
}

#[test]
fn c2_finds_the_angle_of_a_turned_disc() {
    let cell = samples::cell();
    let a = warp_similarity(&cell, 98, 98, 0.0, 1.0, 0.0, 0.0, 112.0, 80.0);
    let b = warp_similarity(&cell, 98, 98, 45.0, 1.0, 0.0, 0.0, 112.0, 80.0);
    let (pa, pb) = (polar(&a, 49.0, 49.0, 48.0, 48, 256), polar(&b, 49.0, 49.0, 48.0, 48, 256));
    let s = rotational_correlation(&pa, &pb, &[1.0; 48]);
    let best = (0..256).max_by(|&i, &j| s[i].total_cmp(&s[j])).unwrap();
    assert_eq!(best, 32, "45° is 32 of 256 columns; the peak is at column {best}");
}

#[test]
fn c3_magnitudes_ignore_a_circular_shift() {
    let a = samples::cell().crop(50, 70, 64, 64);
    let b = Image::from_fn(64, 64, |x, y| a[((x + 9) % 64, (y + 50) % 64)]);
    let (ma, mb) = (magnitude_spectrum(&a, false, false), magnitude_spectrum(&b, false, false));
    for (x, y) in ma.coords() {
        assert!((ma[(x, y)] - mb[(x, y)]).abs() < 1e-3 * (1.0 + ma[(x, y)]), "({x}, {y}): {} vs {}", ma[(x, y)], mb[(x, y)]);
    }
    assert!(ma[(32, 32)].abs() < 1e-3, "the mean is removed: |F| at the centre (frequency 0) must be 0, got {}", ma[(32, 32)]);
    // A single wave of 4 cycles across: its two peaks sit 4 pixels left and right of the centre.
    let wave = Image::from_fn(64, 64, |x, _| (2.0 * std::f32::consts::PI * 4.0 * x as f32 / 64.0).cos());
    let mw = magnitude_spectrum(&wave, false, false);
    assert!(mw[(36, 32)] > 1000.0 && mw[(28, 32)] > 1000.0 && mw[(33, 32)] < 1e-2, "frequency 0 belongs at (32, 32): peaks expected at (28, 32) and (36, 32)");
    let hp = magnitude_spectrum(&wave, false, true);
    let x = (std::f32::consts::PI * 4.0 / 64.0).cos();
    assert!((hp[(36, 32)] / mw[(36, 32)] - (1.0 - x) * (2.0 - x)).abs() < 1e-4, "high-pass weight at 4/64 cycles per pixel");
}

#[test]
fn c4_log_polar_rows_and_columns() {
    // A spectrum equal to the radius: row i must read r_i = 30^(i / 9).
    let radius = Image::from_fn(64, 64, |x, y| (x as f32 - 32.0).hypot(y as f32 - 32.0));
    let lp = log_polar(&radius, 1.0, 30.0, 10, 8);
    assert_eq!((lp.width(), lp.height()), (8, 10), "n_theta columns, n_r rows");
    for i in 0..10 {
        let r = 30f32.powf(i as f32 / 9.0);
        for j in 0..8 {
            assert!((lp[(j, i)] - r).abs() < 0.25, "row {i}: got {}, expected the radius {r}", lp[(j, i)]);
        }
    }
    // Half a turn: column j is the angle 180° · j / n_theta.
    let x_minus = Image::from_fn(64, 64, |x, _| x as f32 - 32.0);
    let lp = log_polar(&x_minus, 1.0, 30.0, 10, 8);
    let expected = 30.0 * (std::f32::consts::PI * 6.0 / 8.0).cos();
    assert!((lp[(6, 9)] - expected).abs() < 1e-2, "column 6 of 8 is 135°: got {}, expected {expected}", lp[(6, 9)]);
}

#[test]
fn c5_recovers_30_degrees_and_scale_1_5() {
    let cell = samples::cell();
    let a = warp_similarity(&cell, 128, 128, 0.0, 1.0, 0.0, 0.0, 128.0, 128.0);
    let b = warp_similarity(&cell, 128, 128, 30.0, 1.5, 12.0, -8.0, 128.0, 128.0);
    let e = fourier_mellin(&a, &b, BASE);
    assert!(wrap_deg(e.angle - 30.0).abs() < 1.0, "angle {}°, expected 30° within 1°", e.angle);
    assert!((e.scale / 1.5 - 1.0).abs() < 0.02, "scale {}, expected 1.5 within 2 %", e.scale);
    assert!((e.tx - 12.0).hypot(e.ty + 8.0) < 1.0, "shift ({}, {}), expected (12, −8) within 1 px", e.tx, e.ty);
}

#[test]
fn c5_resolves_the_180_degree_ambiguity() {
    let cell = samples::cell();
    let a = warp_similarity(&cell, 128, 128, 0.0, 1.0, 0.0, 0.0, 128.0, 128.0);
    let b = warp_similarity(&cell, 128, 128, 200.0, 1.0, 10.0, 5.0, 128.0, 128.0);
    let e = fourier_mellin(&a, &b, BASE);
    assert!(wrap_deg(e.angle - 200.0).abs() < 1.0, "angle {}°, expected 200° (= −160°)", e.angle);
    let one = fourier_mellin(&a, &b, Settings { both: false, ..BASE });
    assert!(wrap_deg(one.angle - 20.0).abs() < 1.0, "with only one candidate the angle is 20° (mod 180°), got {}", one.angle);
}
