// Checks for the coding exercises, using the numbers of the lecture and the exercises.
// Run: cargo test -p day01 fourier::

use day01::fourier::*;
use imgcourse::{samples, Complex, Image};

fn c(re: f32, im: f32) -> Complex {
    Complex::new(re, im)
}

fn real(v: &[f32]) -> Vec<Complex> {
    v.iter().map(|&x| Complex::from(x)).collect()
}

fn assert_close(got: &[Complex], expected: &[Complex], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: wrong length");
    for (g, e) in got.iter().zip(expected) {
        assert!((*g - *e).abs() < 1e-3, "{what}: got {got:?}, expected {expected:?}");
    }
}

#[test]
fn c1_dft_lecture_table() {
    assert_close(&dft(&real(&[1., 1., 1., 1.])), &[c(4., 0.), c(0., 0.), c(0., 0.), c(0., 0.)], "DFT of 1 1 1 1");
    assert_close(&dft(&real(&[1., 0., -1., 0.])), &[c(0., 0.), c(2., 0.), c(0., 0.), c(2., 0.)], "DFT of 1 0 -1 0");
    assert_close(&dft(&real(&[0., 1., 0., -1.])), &[c(0., 0.), c(0., -2.), c(0., 0.), c(0., 2.)], "DFT of 0 1 0 -1");
}

#[test]
fn c1_dft_b1() {
    let x = real(&[1., 0., -1., 0., 1., 0., -1., 0.]);
    let mut expected = vec![Complex::ZERO; 8];
    expected[2] = c(4., 0.);
    expected[6] = c(4., 0.);
    assert_close(&dft(&x), &expected, "B1: two cycles in 8 samples");
}

#[test]
fn c2_idft_undoes_dft() {
    let x: Vec<Complex> = (0..12).map(|i| c((i as f32 * 0.7).sin(), (i * i % 5) as f32)).collect();
    assert_close(&idft(&dft(&x)), &x, "idft(dft(x)) should be x");
}

#[test]
fn c3_frequency_response_b3() {
    // Waves k = 0 (flat), 2 (the slow wave, period 4) and 4 (the fastest) on length 8.
    let blur = frequency_response(&[0.25, 0.5, 0.25], 8);
    let second = frequency_response(&[1., -2., 1.], 8);
    let pick = |r: &[Complex]| vec![r[0], r[2], r[4]];
    assert_close(&pick(&blur), &[c(1., 0.), c(0.5, 0.), c(0., 0.)], "blur (1 2 1)/4 at k = 0, 2, 4");
    assert_close(&pick(&second), &[c(0., 0.), c(-2., 0.), c(-4., 0.)], "second difference at k = 0, 2, 4");
    // A kernel that is not symmetric: the shift 'take the right neighbour' only turns phases.
    let shift = frequency_response(&[0., 0., 1.], 4);
    assert_close(&shift, &[c(1., 0.), c(0., 1.), c(-1., 0.), c(0., -1.)], "take the right neighbour, length 4");
}

#[test]
fn c4_dft2_of_stripes() {
    // Vertical stripes 4 pixels apart in a 16 × 8 image: dots at (u, v) = (±4, 0), i.e.
    // at the indices (4, 0) and (12, 0); everything else is 0.
    let img = Image::from_fn(16, 8, |x, _| Complex::from((2.0 * std::f32::consts::PI * x as f32 / 4.0).cos()));
    let s = dft2(&img, false);
    for (x, y) in s.coords() {
        let expected = if y == 0 && (x == 4 || x == 12) { 64.0 } else { 0.0 };
        assert!((s[(x, y)].abs() - expected).abs() < 1e-2, "|X| at ({x}, {y}): got {}, expected {expected}", s[(x, y)].abs());
    }
    let back = dft2(&s, true);
    for (x, y) in img.coords() {
        assert!((back[(x, y)] - img[(x, y)]).abs() < 1e-3, "the inverse 2D DFT should give the image back");
    }
}

#[test]
fn c5_lowpass() {
    let img = samples::cell().crop(60, 60, 32, 32);
    // A huge radius keeps everything; radius 0 keeps only the mean.
    let all = lowpass(&img, 100.0);
    let mean = lowpass(&img, 0.0);
    for (x, y) in img.coords() {
        assert!((all[(x, y)] - img[(x, y)]).abs() < 1e-3, "radius 100 should keep the image");
        assert!((mean[(x, y)] - img.mean()).abs() < 1e-3, "radius 0 should leave only the mean");
    }
    // A stripe pattern with frequency 8 survives radius 10 and is removed by radius 5.
    let stripes = Image::from_fn(32, 32, |x, _| (2.0 * std::f32::consts::PI * 8.0 * x as f32 / 32.0).cos());
    assert!(lowpass(&stripes, 10.0).max() > 0.99, "radius 10 should keep frequency 8");
    assert!(lowpass(&stripes, 5.0).max() < 1e-3, "radius 5 should remove frequency 8");
}
