// Checks for the coding exercises, using the numbers of the pen-and-paper exercises.
// Run: cargo test -p xcorr-fft

use imgcourse::{samples, Image};
use xcorr_fft::*;

fn assert_all_close(got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: {} values, expected {}", got.len(), expected.len());
    for (t, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!((g - e).abs() < 1e-3, "{what}: got {got:?}, expected {expected:?} (first difference at index {t})");
    }
}

#[test]
fn c1_direct_b1() {
    assert_all_close(&xcorr_direct(&[0., 1., 2., 0.], &[1., 2., 0., 0.]), &[2., 5., 2., 0.], "B1(a)");
    assert_all_close(&xcorr_direct(&[0., 1., 2., 0.], &[0., 0., 1., 2.]), &[2., 0., 2., 5.], "B1(c)");
}

#[test]
fn c2_fft_b1_b2() {
    assert_all_close(&xcorr_fft(&[0., 1., 2., 0.], &[1., 2., 0., 0.]), &[2., 5., 2., 0.], "B1(b)");
    // B2: with the means (3/4 each) subtracted, every value drops by 9/4.
    let a = [0., 1., 2., 0.].map(|v| v - 0.75);
    let b = [1., 2., 0., 0.].map(|v| v - 0.75);
    assert_all_close(&xcorr_fft(&a, &b), &[-0.25, 2.75, -0.25, -2.25], "B2, means subtracted");
}

#[test]
fn c2_fft_equals_direct() {
    let row = samples::cell().row(128).to_vec();
    for n in [16, 12] {
        let a = &row[40..40 + n];
        let b = &row[100..100 + n];
        assert_all_close(&xcorr_fft(a, b), &xcorr_direct(a, b), &format!("length {n}, FFT against direct"));
    }
}

#[test]
fn c3_linear_equals_sliding_matching_b1() {
    // The matching unit's B1: B = (1, 3, 1) at the positions 0..5 of A gives CC 6, 11, 6, 3, 12, 22.
    let a = [0., 1., 3., 1., 0., 2., 6., 2.];
    let b = [1., 3., 1.];
    let c = xcorr_linear(&a, &b);
    assert_eq!(c.len(), 10, "shifts −2 .. 7: expected 10 values, got {}", c.len());
    assert_all_close(&c[2..8], &[6., 11., 6., 3., 12., 22.], "shifts 0..5");
    // the partial overlaps: t = −2 (B's last sample on A's first), t = −1, t = 6, t = 7
    let edges = [c[0], c[1], c[8], c[9]];
    assert_all_close(&edges, &[0., 1., 12., 2.], "shifts −2, −1, 6, 7 (partial overlaps)");
}

#[test]
fn c4_xcorr2_finds_a_circular_shift() {
    let a = samples::cell().crop(60, 40, 64, 32);
    // B(x, y) = A(x + 9, y − 5), wrapping around
    let b = Image::from_fn(64, 32, |x, y| a[((x + 9) % 64, (y + 32 - 5) % 32)]);
    let (tx, ty, _) = peak(&xcorr2(&a, &b));
    assert!((tx, ty) == (9, -5), "peak at ({tx}, {ty}), expected (9, −5)");
}

#[test]
fn c4_xcorr2_ignores_the_means() {
    let a = samples::cell().crop(60, 40, 16, 16);
    let b = samples::cell().crop(63, 41, 16, 16);
    let c1 = xcorr2(&a, &b);
    let c2 = xcorr2(&a.map(|v| v + 0.5), &b.map(|v| 2.0 * v - 0.3));
    for (x, y) in c1.coords() {
        assert!((2.0 * c1[(x, y)] - c2[(x, y)]).abs() < 1e-3, "({x}, {y}): an offset must not change the correlation, a gain of 2 doubles it");
    }
}
