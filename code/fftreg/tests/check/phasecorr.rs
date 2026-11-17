// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p fftreg phasecorr::

use imgcourse::{samples, Image};
use fftreg::phasecorr::*;

fn assert_all_close(got: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: {} values, expected {}", got.len(), expected.len());
    for (g, e) in got.iter().zip(expected) {
        assert!((g - e).abs() < tol, "{what}: got {got:?}, expected {expected:?}");
    }
}

const PHASE: Settings = Settings { p: 1.0, taper: false, lowpass_sd: 0.0 };

#[test]
fn c1_delta_b1() {
    assert_all_close(&phase_correlation_1d(&[0., 1., 2., 0.], &[1., 2., 0., 0.]), &[0., 1., 0., 0.], 1e-4, "B1(b)");
}

#[test]
fn c1_gain_and_offset_do_not_matter_b2() {
    assert_all_close(&phase_correlation_1d(&[0., 1., 2., 0.], &[3., 5., 1., 1.]), &[0., 1., 0., 0.], 1e-4, "B2, B' = 2B + 1");
}

#[test]
fn c2_hann_b5() {
    assert_all_close(&hann(5), &[0., 0.5, 1., 0.5, 0.], 1e-5, "B5(a), N = 5");
    assert_all_close(&hann(8), &[0., 0.19, 0.61, 0.95, 0.95, 0.61, 0.19, 0.], 0.01, "B5(a), N = 8");
}

#[test]
fn c3_circular_shift_gives_a_spike_of_height_1() {
    let a = samples::cell().crop(70, 60, 64, 64);
    let b = Image::from_fn(64, 64, |x, y| a[((x + 64 - 6) % 64, (y + 11) % 64)]); // B(x) = A(x − 6, y + 11)
    let r = phase_correlation(&a, &b, PHASE);
    let (tx, ty, height, _) = peak_and_ratio(&r);
    assert!((tx, ty) == (-6, 11), "peak at ({tx}, {ty}), expected (−6, 11)");
    assert!((height - 1.0).abs() < 1e-3, "peak height {height}, expected 1 for a circular shift");
    let rest = r.coords().filter(|&c| c != (58, 11)).map(|c| r[c].abs()).fold(0.0f32, f32::max);
    assert!(rest < 1e-3, "away from the spike the surface must be 0, found {rest}");
}

#[test]
fn c3_brightness_of_b_does_not_matter() {
    let cell = samples::cell();
    let (a, b) = (cell.crop(40, 50, 64, 64), cell.crop(49, 45, 64, 64));
    let s = Settings { p: 1.0, taper: true, lowpass_sd: 0.0 };
    let r1 = phase_correlation(&a, &b, s);
    let r2 = phase_correlation(&a, &b.map(|v| 3.0 * v + 0.4), s);
    for (x, y) in r1.coords() {
        assert!((r1[(x, y)] - r2[(x, y)]).abs() < 1e-3, "({x}, {y}): {} for B, {} for 3B + 0.4", r1[(x, y)], r2[(x, y)]);
    }
    let (tx, ty, _, _) = peak_and_ratio(&r1);
    assert!((tx, ty) == (9, -5), "two crops 9 px apart across and 5 px up: peak at ({tx}, {ty}), expected (9, −5)");
}

#[test]
fn c3_p_0_is_cross_correlation() {
    let cell = samples::cell();
    let (a, b) = (cell.crop(40, 50, 32, 32), cell.crop(43, 52, 32, 32));
    let r = phase_correlation(&a, &b, Settings { p: 0.0, taper: false, lowpass_sd: 0.0 });
    // direct circular cross-correlation of the mean-free crops at shift (5, 30)
    let (ma, mb) = (a.mean(), b.mean());
    let mut direct = 0.0;
    for (x, y) in b.coords() {
        direct += (a[((x + 5) % 32, (y + 30) % 32)] - ma) * (b[(x, y)] - mb);
    }
    assert!((r[(5, 30)] - direct).abs() < 1e-3 * direct.abs().max(1.0), "p = 0, shift (5, 30): got {}, the cross-correlation is {direct}", r[(5, 30)]);
}

#[test]
fn c4_peak_and_ratio() {
    let mut c = Image::new(8, 8, 0.0f32);
    c[(6, 1)] = 0.8; // shift (−2, 1)
    c[(2, 5)] = 0.2; // a second, smaller local maximum
    c[(7, 1)] = 0.5; // a neighbour of the peak: not a local maximum
    let (tx, ty, h, ratio) = peak_and_ratio(&c);
    assert!((tx, ty) == (-2, 1), "peak at ({tx}, {ty}), expected (−2, 1)");
    assert!((h - 0.8).abs() < 1e-6, "height {h}, expected 0.8");
    assert!((ratio - 4.0).abs() < 1e-5, "ratio {ratio}, expected 0.8 / 0.2 = 4 (0.5 is next to the peak, not a separate peak)");
}
